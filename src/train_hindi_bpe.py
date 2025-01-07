from hindi_bpe_trainer import create_hindi_bpe, evaluate_compression
import requests
import os

def main():
    # Create directory for data if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Use the Hindi sentences file for training
    input_files = ["data/hin_mixed_2019_30K/hin_mixed_2019_30K-sentences.txt"]
    
    print("Starting tokenizer training...")
    print(f"Using input file: {input_files[0]}")
    
    print("\nPhase 1: Initialization")
    print("----------------------")
    print(f"Input file: {input_files[0]}")
    file_size = os.path.getsize(input_files[0]) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")
    
    print("\nPhase 2: Training")
    print("----------------")
    tokenizer = create_hindi_bpe(
        input_files,
        vocab_size=6000,
        min_frequency=2,
        save_path="hindi_bpe.json"
    )
    
    print("\nPhase 3: Evaluation")
    print("-----------------")
    compression_ratio = evaluate_compression(tokenizer, input_files[0])
    print(f"Final vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Compression ratio: {compression_ratio:.2f}")

if __name__ == "__main__":
    main() 