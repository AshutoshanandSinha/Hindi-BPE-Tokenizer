import torch
from collections import defaultdict, Counter
import re
import json
from typing import Dict, List, Tuple, Set
from tqdm.auto import tqdm
import unicodedata
import os
import pickle
import numpy as np

class HindiBPE:
    def __init__(self, vocab_size: int = 8000, min_freq: int = 2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.vocab = {}
        self.merges = {}
        
        current_id = 0
        
        # First initialize with ASCII characters (0-255)
        for i in range(256):
            self.vocab[chr(i)] = i
            current_id = i + 1
        
        # Add special tokens
        self.special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
        for token in self.special_tokens:
            self.vocab[token] = current_id
            current_id += 1
        
        # Add Hindi punctuation with unique IDs
        for punct in ["।", "?", "!", ","]:
            if punct not in self.vocab:
                self.vocab[punct] = current_id
                current_id += 1
        
        # Add common complete words
        self.common_words = {
            "में", "का", "की", "के", "है", "से", "को", "और", "ने", "पर",
            "कर", "था", "थी", "थे", "हैं", "गया", "गयी", "गये", "रहा", "रही",
            "एक", "यह", "वह", "कि", "जो", "तो", "भी", "हो", "कुछ", "अब",
            "लिए", "साथ", "बाद","लिया", "गये", "दिया", "करने", "किया", "होता", "करते",
            "बात", "लोग", "काम", "देश", "समय", "दिन", "कहा", "होने", "बार", "जाता"
        }
        for word in self.common_words:
            if word not in self.vocab:  # Only add if not already in ASCII range
                self.vocab[word] = current_id
                current_id += 1
        
        # Finally add Hindi characters
        self.base_chars = set("अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह" + 
                             "़" +  # Add nukta
                             "।?!,")
        self.matra_chars = set("ािीुूृेैोौंःँ्")
        self.all_chars = self.base_chars.union(self.matra_chars)
        for char in self.base_chars:
            if char.strip() and char not in self.vocab:
                self.vocab[char] = current_id
                current_id += 1
        
        # Common Hindi word endings and prefixes
        self.common_suffixes = ["ों", "ाएं", "ाओं", "ाता", "ाती", "ाते", "ाना", "ाने", "ेगा", "ेगी", 
                                "कर", "िया", "ियों", "वाला", "वाले", "वाली", "कार", "ता", "त्व", "मान"]
        self.common_prefixes = ["अन", "अध", "उप", "प्र", "सम", "अभि", "परि", "विश", "सर्व",
                                "महा", "अति", "सु", "कु", "नि", "दुर्", "स्व", "अनु"]
        
        # Add normalization mappings
        self.char_mappings = {
            # Common variations of a
            'ऍ': 'ए',
            'ॲ': 'अ',
            # Normalize chandrabindu
            'ॐ': 'ओम्',
            # Normalize nukta variations
            'क़': 'क',
            'ख़': 'ख',
            'ग़': 'ग',
            'ज़': 'ज',
            'ड़': 'ड',
            'ढ़': 'ढ',
            'फ़': 'फ',
            # Normalize traditional conjuncts
            'क्ष': 'क्ष',
            'त्र': 'त्र',
            'ज्ञ': 'ज्ञ',
            'श्र': 'श्र',
            # Normalize different types of spaces
            '\u200b': ' ',  # Zero width space
            '\u200c': '',   # Zero width non-joiner
            '\u200d': '',   # Zero width joiner
            '\xa0': ' ',    # Non-breaking space
        }
        
        # Common Hindi abbreviations
        self.abbreviations = {
            'डॉ': 'डॉक्टर',
            'श्री': 'श्रीमान',
            'सु': 'सुश्री',
            'कु': 'कुमारी',
            'प्रो': 'प्रोफेसर',
        }
        
        # Add common syllable combinations
        self.syllables = set()
        for char in self.base_chars:
            if char in "।?!,":
                continue
            # Add base character
            self.syllables.add(char)
            # Add character + matra combinations
            for matra in "ािीुूृेैोौं":
                self.syllables.add(char + matra)
            # Add common conjuncts
            for halant_char in self.base_chars:
                if halant_char not in "।?!,":
                    self.syllables.add(char + '्' + halant_char)
        
        # Add syllables to vocab
        for syllable in self.syllables:
            if syllable not in self.vocab:
                self.vocab[syllable] = len(self.vocab)
        
    def _tokenize_word(self, word: str) -> List[str]:
        word = self._normalize_text(word)
        
        # First check if it's a complete word in vocab
        if word in self.vocab:
            return [word]
        
        tokens = []
        i = 0
        while i < len(word):
            # Try to match longest possible sequence first
            matched = False
            max_length = min(len(word) - i, 12)  # Increased from 8 to 12
            
            # First try matching complete syllables or words
            for length in range(max_length, 0, -1):
                subword = word[i:i+length]
                
                # Check if it's a valid token, common word, or known syllable
                if (subword in self.vocab or 
                    subword in self.common_words or 
                    subword in self.syllables):
                    tokens.append(subword)
                    i += length
                    matched = True
                    break
                
                # Check for valid syllable structure with conjuncts
                if len(subword) >= 3 and '्' in subword:
                    parts = subword.split('्')
                    if all(p in self.base_chars or p in self.matra_chars for p in parts):
                        tokens.append(subword)
                        i += length
                        matched = True
                        break
            
            if not matched:
                # Handle consonant clusters with following matras
                if i < len(word) - 1:
                    cluster = word[i]
                    next_pos = i + 1
                    
                    # Collect halant + consonant sequences
                    while (next_pos < len(word) - 1 and 
                           word[next_pos] == '्' and 
                           word[next_pos + 1] in self.base_chars):
                        cluster += word[next_pos:next_pos + 2]
                        next_pos += 2
                    
                    # Add following matras
                    while next_pos < len(word) and word[next_pos] in self.matra_chars:
                        cluster += word[next_pos]
                        next_pos += 1
                    
                    if cluster in self.vocab or cluster in self.syllables:
                        tokens.append(cluster)
                        i = next_pos
                    else:
                        # Fallback: add base character with its matras
                        char = word[i]
                        i += 1
                        while i < len(word) and word[i] in self.matra_chars:
                            char += word[i]
                            i += 1
                        tokens.append(char)
                else:
                    # Handle single character at end of word
                    tokens.append(word[i])
                    i += 1
        
        return tokens
    
    def _normalize_text(self, text: str) -> str:
        """Enhanced normalization for Hindi text"""
        # 1. Basic cleanup
        text = text.strip()
        
        # 2. NFKC normalization
        text = unicodedata.normalize('NFKC', text)
        
        # 3. Replace variations with standard forms
        for old, new in self.char_mappings.items():
            text = text.replace(old, new)
        
        # 4. Normalize spaces only
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = text.strip()  # Remove leading/trailing spaces
        text = re.sub(r'[-_]+', '-', text)
        text = re.sub(r'[""'']', '"', text)
        text = re.sub(r'\.{2,}', '...', text)
        
        return text
    
    def _get_stats(self, words: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        pairs_dict = {}
        
        for word, freq in words.items():
            symbols = word.split()
            if len(symbols) < 2:
                continue
            
            for j in range(len(symbols) - 1):
                pair = (symbols[j], symbols[j + 1])
                combined = ''.join(pair)
                
                # Enhanced boosting strategy
                boost = 1
                
                # Boost complete words
                if combined in self.common_words:
                    boost = 20
                # Boost for syllables with matras
                elif any(c in self.matra_chars for c in combined):
                    boost = 15
                # Boost for verb forms
                elif any(combined.endswith(suffix) for suffix in ["ना", "ता", "ते", "ती", "गा", "गी", "या", "ये", "कर"]):
                    boost = 12
                # Boost for common suffixes
                elif any(combined.endswith(suffix) for suffix in self.common_suffixes):
                    boost = 10
                # Boost for common prefixes
                elif any(combined.startswith(prefix) for prefix in self.common_prefixes):
                    boost = 8
                # Boost for consonant clusters
                elif '्' in combined:
                    boost = 6
                
                pairs_dict[pair] = pairs_dict.get(pair, 0) + freq * boost
        
        return pairs_dict
    
    def _merge_pair(self, pair: Tuple[str, str], words: Dict[str, int]) -> Dict[str, int]:
        """Optimized batch merge operation"""
        new_words = {}
        replacement = ''.join(pair)
        batch_size = 1000
        
        # Convert to arrays for faster processing
        items = list(words.items())
        for i in range(0, len(items), batch_size):
            batch = items[i:min(i+batch_size, len(items))]
            word_list, freq_list = zip(*batch)
            
            # Process batch
            for word, freq in zip(word_list, freq_list):
                if freq < self.min_freq:
                    continue
                    
                parts = word.split()
                if len(parts) < 2:
                    new_words[word] = freq
                    continue
                
                # Fast string join and split
                new_word = ' '.join(parts).replace(f"{pair[0]} {pair[1]}", replacement)
                new_words[new_word] = freq
        
        return new_words
    
    def train(self, input_file: str, batch_size: int = 10000):
        """Optimized training with parallel processing and improved merging strategy"""
        # Load and preprocess text in larger batches
        words = []
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
            words = text.split()
        
        # Process in larger batches
        word_freqs = Counter()
        for i in range(0, len(words), batch_size):
            batch = words[i:i + batch_size]
            word_freqs.update(batch)
        
        # Initialize vocabulary with character sequences
        vocab = {}
        with tqdm(total=len(word_freqs), desc="Processing words") as pbar:
            # Process most frequent words first
            for word, freq in word_freqs.most_common():
                if freq >= self.min_freq:
                    # Try to keep common words intact
                    if word in self.common_words or freq > self.min_freq * 5:
                        if word not in self.vocab:
                            self.vocab[word] = len(self.vocab)
                    else:
                        chars = ' '.join(self._tokenize_word(word))
                        vocab[chars] = freq
                    pbar.update(1)
        
        # BPE training with improved merging strategy
        num_merges = self.vocab_size - len(self.vocab)
        with tqdm(total=num_merges, desc="Training BPE") as pbar:
            while len(self.vocab) < self.vocab_size:
                pairs = self._get_stats(vocab)
                if not pairs:
                    break
                
                # Get best pairs considering frequency and length
                best_pairs = []
                for pair, freq in sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:50]:
                    merged = ''.join(pair)
                    # Prioritize longer sequences that form meaningful units
                    if (len(merged) > 3 and 
                        any(merged.endswith(suffix) for suffix in self.common_suffixes) or
                        any(merged.startswith(prefix) for prefix in self.common_prefixes) or
                        '्' in merged):
                        best_pairs.append((pair, freq))
                    if len(best_pairs) >= 10:
                        break
                
                if not best_pairs:
                    break
                    
                # Apply merges
                for best_pair, _ in best_pairs:
                    if len(self.vocab) >= self.vocab_size:
                        break
                    merged_token = ''.join(best_pair)
                    if merged_token not in self.vocab:
                        vocab = self._merge_pair(best_pair, vocab)
                        self.vocab[merged_token] = len(self.vocab)
                        self.merges[best_pair] = len(self.vocab) - 1
                        pbar.update(1)
    

    
    def save(self, path: str):
        """Save tokenizer to file"""
        data = {
            'vocab': self.vocab,
            'merges': self.merges,
            'special_tokens': self.special_tokens,
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load tokenizer from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vocab = data['vocab']
        self.merges = data['merges']
        self.special_tokens = data['special_tokens']
        self.vocab_size = data.get('vocab_size', 5000)  # Default to 5000 if not found
        self.min_freq = data.get('min_freq', 2)
    
    def encode(self, text: str) -> List[int]:
        tokens = []
        text = self._normalize_text(text)
        words = text.split()
        
        for word in words:
            # First try the complete word
            if word in self.vocab:
                tokens.append(self.vocab[word])
                continue
            
            # Tokenize the word into subwords
            word_tokens = self._tokenize_word(word)
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.vocab['[UNK]'])
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text"""
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        result = []
        for id in ids:
            token = id_to_token.get(id, '[UNK]')
            
            if token in ["।", "?", "!", ","] or token in "ािीुूृेैोौंँःृ्":
                result.append(token)
            elif token in self.special_tokens:
                result.append(token)
            else:
                if result and not token.startswith('्') and not token in "ािीुूृेैोौंँःृ्":
                    result.append(' ')
                result.append(token)
        
        decoded = ''.join(result).strip()
        decoded = re.sub(r'\s+', ' ', decoded)
        decoded = re.sub(r'\s+([।?!,])', r'\1', decoded)
        
        return decoded
    
    def print_vocab_stats(self):
        """Print vocabulary statistics for debugging"""
        print("\nVocabulary Statistics:")
        print(f"Total vocab size: {len(self.vocab)}")
        print("\nSpecial tokens:", self.special_tokens)
        print("\nSample common words in vocab:")
        for word in list(self.common_words)[:10]:
            print(f"{word}: {self.vocab.get(word, 'Not found')}")
        print("\nSample base characters in vocab:")
        for char in list(self.base_chars)[:10]:
            print(f"{char}: {self.vocab.get(char, 'Not found')}") 