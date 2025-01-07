from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors, normalizers
from tokenizers.processors import TemplateProcessing
import os
from pathlib import Path
import re
from tqdm import tqdm

def preprocess_hindi_text(text):
    # Only normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    return text

def create_hindi_bpe(
    input_files,
    vocab_size=10000,
    min_frequency=2,
    save_path="hindi_bpe.json"
):
    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(models.BPE())
    
    # Add NFKC normalization
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Replace(r" {2,}", " ")
    ])
    
    # Use SentencePiece-style pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
        replacement="▁",
        add_prefix_space=True
    )
    
    # Initialize trainer with logging
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        initial_alphabet=list("अआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह़क्षत्रज्ञ"),
        show_progress=True  # This enables built-in progress bars
    )
    
    # Print status before training
    print("\nStarting BPE training:")
    print(f"Target vocab size: {vocab_size}")
    print(f"Minimum frequency: {min_frequency}")
    print("Training progress (this may take a while)...")
    
    # Train with periodic status updates
    tokenizer.train(input_files, trainer)
    
    # Print final stats
    vocab = tokenizer.get_vocab()
    print(f"\nTraining completed:")
    print(f"Final vocab size: {len(vocab)}")
    print(f"Sample tokens from vocabulary:")
    for token, id in list(vocab.items())[:10]:
        print(f"  {token}: {id}")
    
    # Add post-processor
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )
    
    # Add decoder
    tokenizer.decoder = decoders.Metaspace(
        replacement="▁",
        add_prefix_space=True
    )
    
    tokenizer.save(save_path)
    return tokenizer

def evaluate_compression(tokenizer, test_file, chunk_size=1000):
    """Evaluate compression ratio in chunks to avoid memory issues"""
    total_original_length = 0
    total_encoded_length = 0
    
    print("\nEvaluating compression ratio...")
    from tqdm import tqdm
    
    # Count total lines first
    with open(test_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    # Process file line by line with progress bar
    with open(test_file, 'r', encoding='utf-8') as f:
        pbar = tqdm(total=total_lines, desc="Processing lines")
        
        for line in f:
            processed_text = preprocess_hindi_text(line)
            total_original_length += len(line)
            encoded = tokenizer.encode(processed_text)
            total_encoded_length += len(encoded.ids)
            pbar.update(1)
        
        pbar.close()
    
    print("\nCompression evaluation complete!")
    return total_original_length / total_encoded_length if total_encoded_length > 0 else 0 