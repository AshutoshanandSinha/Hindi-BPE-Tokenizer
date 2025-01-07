import torch
from collections import defaultdict, Counter
import re
import json
from typing import Dict, List, Tuple, Set
from tqdm.auto import tqdm
import unicodedata
import os

class HindiBPE:
    def __init__(self, vocab_size: int = 8000, min_freq: int = 2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.vocab = {}
        self.merges = {}
        self.special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
        
        # Common complete words that should be single tokens
        self.common_words = {
            "में", "का", "की", "के", "है", "से", "को", "और", "ने", "पर",
            "कर", "था", "थी", "थे", "हैं", "गया", "गयी", "गये", "रहा", "रही",
            "एक", "यह", "वह", "कि", "जो", "तो", "भी", "हो", "कुछ", "अब"
        }
        
        # Initialize with complete words first
        for word in self.common_words:
            self.vocab[word] = len(self.vocab)
        
        # Then add special tokens
        for token in self.special_tokens:
            self.vocab[token] = len(self.vocab)
        
        # Finally add individual characters
        self.base_chars = set("अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह" + 
                             "ा ि ी ु ू ृ े ै ो ौ ं ः ँ")
        for char in self.base_chars:
            if char.strip() and char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        
        # Common Hindi word endings and prefixes
        self.common_suffixes = ["ों", "ाएं", "ाओं", "ाता", "ाती", "ाते", "ाना", "ाने", "ेगा", "ेगी", "कर", "िया", "ियों"]
        self.common_prefixes = ["अन", "अध", "उप", "प्र", "सम", "अभि", "परि", "विश", "सर्व"]
        
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
        
    def _tokenize_word(self, word: str) -> List[str]:
        """Split word into meaningful Hindi subunits"""
        word = self._normalize_text(word)
        
        # First check if it's a common word
        if word in self.common_words:
            return [word]
        
        tokens = []
        i = 0
        while i < len(word):
            # Try to match longest possible meaningful unit
            found = False
            for length in range(min(6, len(word) - i), 0, -1):
                subword = word[i:i+length]
                if subword in self.common_words:
                    tokens.append(subword)
                    i += length
                    found = True
                    break
            
            if not found:
                # Handle consonant clusters with vowel marks
                if i < len(word) - 1 and word[i+1] == '्':
                    j = i + 2
                    while j < len(word) and word[j] not in "ािीुूृेैोौं":
                        j += 1
                    if j > i + 2:
                        tokens.append(word[i:j])
                        i = j
                        continue
                
                # Handle single character with modifiers
                char = word[i]
                i += 1
                while i < len(word) and word[i] in "ािीुूृेैोौंँःृ्":
                    char += word[i]
                    i += 1
                tokens.append(char)
        
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
        
        # 4. Normalize spaces and punctuation - modified
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = text.strip()  # Remove leading/trailing spaces
        text = re.sub(r'[?!।॥]+', '।', text)
        text = re.sub(r'[-_]+', '-', text)
        text = re.sub(r'[""'']', '"', text)
        text = re.sub(r'\.{2,}', '...', text)
        
        return text
    
    def _get_stats(self, words: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Optimized frequency counting with linguistic awareness"""
        pairs_dict = {}
        
        for word, freq in words.items():
            symbols = word.split()
            if len(symbols) < 2:
                continue
            
            for j in range(len(symbols) - 1):
                pair = (symbols[j], symbols[j + 1])
                combined = ''.join(pair)
                
                # Boost based on linguistic patterns
                boost = 1
                
                # Higher boost for complete word formations
                if combined in ["में", "का", "की", "के", "से", "है", "को", "और", "ने", "पर"]:
                    boost = 5
                # Boost for common suffixes
                elif any(combined.endswith(suffix) for suffix in self.common_suffixes):
                    boost = 3
                # Boost for common prefixes
                elif any(combined.startswith(prefix) for prefix in self.common_prefixes):
                    boost = 2
                
                pairs_dict[pair] = pairs_dict.get(pair, 0) + freq * boost
        
        return pairs_dict
    
    def _merge_pair(self, pair: Tuple[str, str], words: Dict[str, int]) -> Dict[str, int]:
        """Highly optimized merge operation"""
        new_words = {}
        replacement = ''.join(pair)
        
        # Process all words at once
        for word, freq in words.items():
            if freq < self.min_freq:
                continue
            
            parts = word.split()
            if len(parts) < 2:
                new_words[word] = freq
                continue
            
            # Fast merge using list operations
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == pair[0] and parts[i + 1] == pair[1]:
                    new_parts.append(replacement)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            
            new_words[' '.join(new_parts)] = freq
        
        return new_words
    
    def train(self, input_file: str):
        """Optimized training process"""
        print("\nPhase 1: Data Loading and Preprocessing")
        print("-" * 40)
        
        # Load data with progress tracking
        file_size = os.path.getsize(input_file) / (1024 * 1024)
        print(f"Loading file ({file_size:.2f} MB)...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Process initial statistics with progress tracking
        print("\nTokenizing words...")
        words = text.split()
        total_words = len(words)
        print(f"Total words: {total_words:,}")
        
        print("\nCounting word frequencies...")
        word_freqs = Counter(words)
        unique_words = len(word_freqs)
        print(f"Unique words: {unique_words:,}")
        
        print("\nInitializing vocabulary...")
        # Initialize vocabulary with progress tracking
        vocab = {}
        with tqdm(total=len(word_freqs), desc="Processing words") as pbar:
            for word, freq in word_freqs.items():
                if freq >= self.min_freq:
                    chars = ' '.join(self._tokenize_word(word))
                    vocab[chars] = freq
                pbar.update(1)
        
        # Initialize base vocabulary
        print("\nAdding base characters and special tokens...")
        initial_vocab_size = len(self.vocab)
        for char in self.base_chars:
            self.vocab[char] = len(self.vocab)
        for token in self.special_tokens:
            self.vocab[token] = len(self.vocab)
        
        print(f"Base vocabulary size: {len(self.vocab) - initial_vocab_size:,}")
        print(f"Words above minimum frequency: {len(vocab):,}")
        
        print("\nPhase 2: BPE Training")
        print("-" * 40)
        num_merges = self.vocab_size - len(self.vocab)
        
        with tqdm(total=num_merges, desc="Training BPE") as pbar:
            for i in range(num_merges):
                pairs = self._get_stats(vocab)
                if not pairs:
                    break
                
                best_pair = max(pairs.items(), key=lambda x: x[1])[0]
                vocab = self._merge_pair(best_pair, vocab)
                self.merges[best_pair] = len(self.vocab)
                self.vocab[''.join(best_pair)] = len(self.vocab)
                
                pbar.update(1)
        
        print("\nPhase 3: Final Statistics")
        print("-" * 40)
        print(f"Final vocabulary size: {len(self.vocab):,}")
        print(f"Number of merges performed: {len(self.merges):,}")
    
    def save(self, path: str):
        """Save tokenizer to file"""
        data = {
            'vocab': self.vocab,
            'merges': {f"{p[0]} {p[1]}": idx for p, idx in self.merges.items()},
            'special_tokens': self.special_tokens
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """Load tokenizer from file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab = data['vocab']
        self.merges = {tuple(k.split()): v for k, v in data['merges'].items()}
        self.special_tokens = data['special_tokens']
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        tokens = []
        words = text.split()
        
        for word in words:
            chars = ' '.join(self._tokenize_word(word))
            while len(chars) > 0:
                # Find longest matching token
                i = len(chars)
                while i > 0:
                    substr = chars[:i].replace(' ', '')
                    if substr in self.vocab:
                        tokens.append(self.vocab[substr])
                        chars = chars[i:].strip()
                        break
                    i -= 1
                if i == 0:
                    # Unknown character
                    tokens.append(self.vocab['[UNK]'])
                    chars = chars[1:].strip()
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text"""
        # Create reverse vocabulary
        id_to_token = {v: k for k, v in self.vocab.items()}
        return ''.join(id_to_token.get(id, '[UNK]') for id in ids) 