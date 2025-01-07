# Hindi BPE Tokenizer

A specialized BPE (Byte Pair Encoding) tokenizer trained on Hindi text, built using the Hugging Face Tokenizers library. This tokenizer is optimized for Hindi language processing with support for the complete Devanagari character set.

## Features

- Custom BPE tokenizer trained specifically for Hindi text
- NFKC normalization and whitespace handling
- SentencePiece-style pre-tokenization
- Special token support ([UNK], [PAD], [BOS], [EOS])
- Built-in Devanagari alphabet initialization
- Progress tracking during training
- Compression ratio evaluation

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── src/
│   ├── hindi_bpe_trainer.py    # Core tokenizer training functionality
│   ├── train_hindi_bpe.py      # Training script
│   └── use_tokenizer.py        # Example usage of trained tokenizer
├── hindi_bpe.json              # Trained tokenizer model
└── requirements.txt            # Project dependencies
```

## Usage

### Training a New Tokenizer

```python
from src.hindi_bpe_trainer import create_hindi_bpe

tokenizer = create_hindi_bpe(
    input_files=["path/to/your/hindi/text.txt"],
    vocab_size=10000,
    min_frequency=2,
    save_path="hindi_bpe.json"
)
```

### Using the Trained Tokenizer

```python
from tokenizers import Tokenizer

# Load the trained tokenizer
tokenizer = Tokenizer.from_file("hindi_bpe.json")

# Tokenize text
text = "आप कैसे हैं?"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded.ids)

print(f"Original: {text}")
print(f"Encoded: {encoded.ids}")
print(f"Decoded: {decoded}")
```

## Configuration

The tokenizer can be customized with the following parameters:

- `vocab_size`: Size of the vocabulary (default: 10000)
- `min_frequency`: Minimum frequency for a token to be included (default: 2)
- Special tokens: [UNK], [PAD], [BOS], [EOS]
- Pre-initialized with complete Devanagari alphabet

## Dependencies

- tokenizers==0.15.0
- requests==2.31.0

## License

[Specify your license here]


