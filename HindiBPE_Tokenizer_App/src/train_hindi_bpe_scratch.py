from src.hindi_bpe_scratch import HindiBPE
import os
from collections import Counter
import matplotlib.pyplot as plt
import json
import torch
from tqdm import tqdm
import re
from itertools import islice
import pickle
from typing import List

def analyze_and_save_results(tokenizer, input_file, batch_size=1000):
    """Analyze tokenizer results with expanded statistics"""
    stats = {}
    token_freqs = Counter()
    word_lengths = []
    tokens_per_line = []
    total_chars = 0
    
    # Get vocabulary statistics
    stats['vocab_size'] = len(tokenizer.vocab)
    
    # Process text and get token frequencies
    with open(input_file, 'r', encoding='utf-8') as f:
        # Process line by line to avoid memory issues
        for line in tqdm(f, desc="Analyzing tokens"):
            if line.strip():
                total_chars += len(line)
                tokens = tokenizer.encode(line)
                token_freqs.update(tokens)
                tokens_per_line.append(len(tokens))
                word_lengths.extend(len(word) for word in line.split())
    
    # Calculate statistics with safety checks
    total_tokens = sum(token_freqs.values())
    stats['total_tokens'] = total_tokens
    stats['unique_tokens'] = len(token_freqs)
    stats['compression_ratio'] = total_chars / total_tokens if total_tokens > 0 else 0
    stats['avg_tokens_per_line'] = sum(tokens_per_line) / len(tokens_per_line) if tokens_per_line else 0
    stats['avg_word_length'] = sum(word_lengths) / len(word_lengths) if word_lengths else 0
    stats['top_tokens'] = {str(k): v for k, v in token_freqs.most_common(100)}
    
    # Save detailed results
    os.makedirs('results', exist_ok=True)
    with open('results/tokenizer_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # Print detailed statistics
    print("\nDetailed Statistics:")
    print("-" * 40)
    print(f"Average tokens per line: {stats['avg_tokens_per_line']:.2f}")
    print(f"Average word length: {stats['avg_word_length']:.2f}")
    print("\nTop 20 most frequent tokens:")
    for token, freq in list(stats['top_tokens'].items())[:20]:
        print(f"  {token}: {freq:,}")
    
    return stats

def test_tokenizer(tokenizer, test_text):
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    
    # Normalize both texts before comparison
    test_text = re.sub(r'\s+', ' ', test_text.strip())
    decoded = re.sub(r'\s+', ' ', decoded.strip())
    
    # Calculate character-level accuracy
    total_chars = len(test_text)
    matches = sum(1 for a, b in zip(test_text, decoded) if a == b)
    accuracy = (matches / total_chars) * 100 if total_chars > 0 else 0
    
    return encoded, decoded, accuracy

def test_common_phrases(tokenizer):
    print("\nTesting common Hindi phrases...")
    print("-" * 50)
    
    test_phrases = [
        "मैं आपसे मिलकर खुश हूं",  # Nice to meet you
        "आपका दिन शुभ हो",         # Have a good day
        "कृपया मेरी मदद करें",      # Please help me
        "धन्यवाद आपका",            # Thank you
        "मुझे समझ में नहीं आया",    # I don't understand
        "यह बहुत अच्छा है",         # This is very good
        "आप कैसे हैं",              # How are you
        "मैं ठीक हूं",              # I am fine
        "फिर मिलेंगे"               # See you again
    ]
    
    results = []
    for phrase in test_phrases:
        encoded = tokenizer.encode(phrase)
        decoded = tokenizer.decode(encoded)
        
        # Calculate accuracy
        total_chars = len(phrase)
        diff_chars = sum(1 for a, b in zip(phrase, decoded) if a != b)
        accuracy = (total_chars - diff_chars) / total_chars * 100
        
        results.append({
            'phrase': phrase,
            'encoded': encoded,
            'decoded': decoded,
            'accuracy': accuracy,
            'num_tokens': len(encoded)
        })
    
    # Print results
    for result in results:
        print("\nOriginal:", result['phrase'])
        print("Decoded: ", result['decoded'])
        print(f"Tokens ({result['num_tokens']}): {result['encoded']}")
        print(f"Accuracy: {result['accuracy']:.2f}%")
    
    # Calculate average metrics
    avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
    avg_tokens = sum(r['num_tokens'] for r in results) / len(results)
    print("\nAverage Results:")
    print(f"Accuracy: {avg_accuracy:.2f}%")
    print(f"Tokens per phrase: {avg_tokens:.2f}")

def main():
    # Initialize tokenizer with smaller vocab and higher min_freq
    tokenizer = HindiBPE(vocab_size=5000, min_freq=10)
    
    # Train on larger dataset
    input_file = "data/hin_mixed_2019_1M/hin_mixed_2019_1M-sentences.txt"
    print(f"Training on {input_file}")
    tokenizer.train(input_file)
    
    # Save the trained tokenizer
    tokenizer.save("results/hindi_bpe_scratch.pkl")
    
    # Analyze results
    print("\nAnalyzing results...")
    stats = analyze_and_save_results(tokenizer, input_file)
    
    print("\nFinal Results:")
    print("-" * 40)
    print(f"Vocabulary size: {stats['vocab_size']:,}")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}")
    print(f"Total tokens processed: {stats['total_tokens']:,}")
    print(f"Unique tokens: {stats['unique_tokens']:,}")
    
    
    # Test common phrases
    test_common_phrases(tokenizer)

if __name__ == "__main__":
    main() 