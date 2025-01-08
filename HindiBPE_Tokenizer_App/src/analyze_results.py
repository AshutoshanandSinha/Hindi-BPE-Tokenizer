from collections import Counter
import matplotlib.pyplot as plt
import json
from hindi_bpe_scratch import HindiBPE

def analyze_tokenizer(model_path: str, test_file: str):
    """Analyze tokenizer performance and generate statistics"""
    tokenizer = HindiBPE()
    tokenizer.load(model_path)
    
    # Load test data
    with open(test_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Generate statistics
    tokens = tokenizer.encode(text)
    token_freqs = Counter(tokens)
    
    # Plot token frequency distribution
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(token_freqs)), sorted(token_freqs.values(), reverse=True))
    plt.title('Token Frequency Distribution')
    plt.xlabel('Token Rank')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.savefig('token_distribution.png')
    
    # Save detailed statistics
    stats = {
        'vocab_size': len(tokenizer.vocab),
        'total_tokens': len(tokens),
        'unique_tokens': len(token_freqs),
        'compression_ratio': len(text) / len(tokens),
        'top_tokens': dict(token_freqs.most_common(20))
    }
    
    with open('tokenizer_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)