from collections import defaultdict, Counter
import re
import json
from typing import Dict, List, Tuple, Set
from tqdm.auto import tqdm

class HindiBPE:
    def __init__(self, vocab_size: int = 10000, min_freq: int = 2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.vocab = {}
        self.merges = {}
        self.special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
        
        # Initialize with Hindi characters
        self.base_chars = set("अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह़क्षत्रज्ञ")
        
    def _tokenize_word(self, word: str) -> List[str]:
        """Split word into individual characters"""
        return list(word)
    
    def _get_stats(self, words: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent pairs"""
        pairs = defaultdict(int)
        for word, freq in words.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs
    
    def _merge_pair(self, pair: Tuple[str, str], words: Dict[str, int]) -> Dict[str, int]:
        """Merge all occurrences of the pair in the vocabulary"""
        new_words = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in words.items():
            parts = word.split()
            i = 0
            new_parts = []
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == pair[0] and parts[i + 1] == pair[1]:
                    new_parts.append(replacement)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            new_word = ' '.join(new_parts)
            new_words[new_word] = freq
        
        return new_words
    
    def train(self, input_file: str):
        """Train BPE on input text"""
        print("\nPhase 1: Data Loading and Preprocessing")
        print("-" * 40)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Add statistics about input data
        total_chars = len(text)
        unique_chars = len(set(text))
        print(f"Total characters: {total_chars:,}")
        print(f"Unique characters: {unique_chars}")
        
        # Normalize and split text
        text = re.sub(r'\s+', ' ', text)
        words = text.split()
        total_words = len(words)
        unique_words = len(set(words))
        print(f"Total words: {total_words:,}")
        print(f"Unique words: {unique_words:,}")
        
        print("\nPhase 2: Initial Vocabulary Building")
        print("-" * 40)
        word_freqs = Counter(words)
        initial_vocab_size = sum(1 for freq in word_freqs.values() if freq >= self.min_freq)
        print(f"Words with frequency >= {self.min_freq}: {initial_vocab_size:,}")
        
        # Initialize vocabulary with characters
        vocab = {}
        for word, freq in word_freqs.items():
            if freq < self.min_freq:
                continue
            chars = ' '.join(self._tokenize_word(word))
            vocab[chars] = freq
        
        # Add special tokens
        for token in self.special_tokens:
            self.vocab[token] = len(self.vocab)
        
        print("\nPhase 3: BPE Training")
        print("-" * 40)
        num_merges = self.vocab_size - len(self.vocab)
        
        # Main BPE training loop with progress bar
        with tqdm(total=num_merges, desc="Training BPE") as pbar:
            for i in range(num_merges):
                pairs = self._get_stats(vocab)
                if not pairs:
                    break
                    
                # Find most frequent pair
                best_pair = max(pairs.items(), key=lambda x: x[1])[0]
                
                # Merge pair in all words
                vocab = self._merge_pair(best_pair, vocab)
                
                # Add to merges and vocabulary
                self.merges[best_pair] = len(self.vocab)
                self.vocab[''.join(best_pair)] = len(self.vocab)
                
                pbar.update(1)
        
        print("\nPhase 4: Final Statistics")
        print("-" * 40)
        print(f"Final vocabulary size: {len(self.vocab):,}")
        print(f"Number of merges performed: {len(self.merges):,}")
        
        # Sample most frequent tokens
        token_freqs = Counter()
        for word, freq in word_freqs.items():
            tokens = self.encode(word)
            token_freqs.update({self.decode([t]): freq for t in tokens})
        
        print("\nTop 10 most frequent tokens:")
        for token, freq in token_freqs.most_common(10):
            print(f"  {token}: {freq:,}")
    
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