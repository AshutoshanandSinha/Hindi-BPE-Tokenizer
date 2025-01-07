from hindi_bpe_scratch import HindiBPE
import os
from collections import Counter
import matplotlib.pyplot as plt
import json
import torch
from tqdm import tqdm
import re

def analyze_and_save_results(tokenizer, input_file):
    """Analyze tokenizer results with expanded statistics"""
    stats = {}
    
    # Get vocabulary statistics
    stats['vocab_size'] = len(tokenizer.vocab)
    
    # Process text and get token frequencies
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Get token frequencies
    token_freqs = Counter()
    for line in tqdm(text.split('\n'), desc="Analyzing token frequencies"):
        if line.strip():
            tokens = tokenizer.encode(line)
            token_freqs.update(tokens)
    
    # Convert token IDs to actual tokens and filter
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    token_freqs_mapped = {
        id_to_token[token_id]: freq 
        for token_id, freq in token_freqs.items()
        if (
            id_to_token[token_id].strip() and  # Not empty
            id_to_token[token_id] not in [' ', ''] and  # Not spaces
            not id_to_token[token_id].isdigit() and  # Not pure numbers
            not bool(re.match(r'^\d+$', id_to_token[token_id]))  # Not numeric strings
        )
    }
    
    stats['total_tokens'] = sum(token_freqs.values())
    stats['unique_tokens'] = len(token_freqs)
    stats['compression_ratio'] = len(text) / len(token_freqs)
    stats['top_tokens'] = dict(sorted(token_freqs_mapped.items(), 
                                    key=lambda x: x[1], reverse=True)[:100])
    
    # Save detailed results
    os.makedirs('results', exist_ok=True)
    with open('results/tokenizer_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return stats

def main():
    # Initialize tokenizer with larger vocab
    tokenizer = HindiBPE(vocab_size=8000, min_freq=2)
    
    # Train
    input_file = "data/hin_mixed_2019_30K/hin_mixed_2019_30K-sentences.txt"
    print(f"Training on {input_file}")
    tokenizer.train(input_file)
    
    # Save the trained tokenizer
    tokenizer.save("hindi_bpe_scratch.json")
    
    # Analyze results
    print("\nAnalyzing results...")
    stats = analyze_and_save_results(tokenizer, input_file)
    
    # Print summary
    print("\nFinal Results:")
    print("-" * 40)
    print(f"Vocabulary size: {stats['vocab_size']:,}")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}")
    print(f"Total tokens processed: {stats['total_tokens']:,}")
    print(f"Unique tokens: {stats['unique_tokens']:,}")
    print("\nTop 100 most frequent tokens:")
    for token, freq in list(stats['top_tokens'].items())[:100]:
        print(f"  {token}: {freq:,}")
    print(f"\nDetailed results saved in 'results' directory")

if __name__ == "__main__":
    main() 