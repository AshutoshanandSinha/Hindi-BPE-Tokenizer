from hindi_bpe_scratch import HindiBPE
import os
from collections import Counter
import matplotlib.pyplot as plt
import json

def analyze_and_save_results(tokenizer, input_file, output_dir="results"):
    """Analyze tokenizer performance and save results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Generate statistics
    tokens = tokenizer.encode(text)
    token_freqs = Counter(tokens)
    
    # Plot token frequency distribution
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(token_freqs)), sorted(token_freqs.values(), reverse=True))
    plt.title('Hindi BPE Token Frequency Distribution')
    plt.xlabel('Token Rank')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.savefig(f'{output_dir}/token_distribution.png')
    plt.close()
    
    # Save detailed statistics
    stats = {
        'vocab_size': len(tokenizer.vocab),
        'total_tokens': len(tokens),
        'unique_tokens': len(token_freqs),
        'compression_ratio': len(text) / len(tokens),
        'top_tokens': {tokenizer.decode([token]): freq 
                      for token, freq in token_freqs.most_common(20)}
    }
    
    with open(f'{output_dir}/tokenizer_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return stats

def main():
    # Initialize tokenizer
    tokenizer = HindiBPE(vocab_size=10000, min_freq=2)
    
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
    print("\nTop 5 most frequent tokens:")
    for token, freq in list(stats['top_tokens'].items())[:5]:
        print(f"  {token}: {freq:,}")
    print(f"\nDetailed results saved in 'results' directory")

if __name__ == "__main__":
    main() 