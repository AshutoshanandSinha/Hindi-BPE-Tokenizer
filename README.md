# Hindi BPE Tokenizer

A specialized BPE (Byte Pair Encoding) tokenizer for Hindi text, implemented in two versions:
1. Using Hugging Face Tokenizers library
2. Custom implementation from scratch

The tokenizers are trained on the Hindi Mixed Corpus (30K sentences) from Leipzig Corpora Collection.

## Data Source

The training data is from the [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download/Hindi), specifically using:
- Dataset: Hindi Mixed Corpus 2019
- Size: 30K sentences
- Type: Mixed sources (news material, web text, etc.)

## Features

- Two different BPE implementations:
  - HuggingFace-based: Using the `tokenizers` library
  - Scratch implementation: Pure Python implementation
- NFKC normalization and whitespace handling
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
│   ├── hindi_bpe_trainer.py     # HuggingFace-based implementation
│   ├── hindi_bpe_scratch.py     # From-scratch implementation
│   ├── train_hindi_bpe.py       # Training script (HuggingFace)
│   ├── train_hindi_bpe_scratch.py # Training script (Scratch)
│   └── use_tokenizer.py         # Example usage
├── hindi_bpe.json               # Trained tokenizer model
└── requirements.txt             # Project dependencies
```

## Usage

### Using HuggingFace Implementation

```python
from src.hindi_bpe_trainer import create_hindi_bpe

tokenizer = create_hindi_bpe(
    input_files=["path/to/your/hindi/text.txt"],
    vocab_size=10000,
    min_frequency=2,
    save_path="hindi_bpe.json"
)
```

### Using Scratch Implementation

```python
from src.hindi_bpe_scratch import HindiBPE

# Initialize and train
bpe = HindiBPE(vocab_size=10000, min_freq=2)
bpe.train("path/to/your/hindi/text.txt")

# Save the model
bpe.save("hindi_bpe.json")

# Load and use
bpe.load("hindi_bpe.json")
encoded = bpe.encode("आप कैसे हैं?")
decoded = bpe.decode(encoded)
```

## Training Results

### Scratch Implementation Results
- Final vocabulary size: 10,000
- Number of merges performed: 9,996
- Total characters processed: 2,864,788
- Unique characters: 260
- Total words: 561,773
- Unique words: 85,321
- Words with frequency ≥ 2: 20,942
- Total tokens processed: 739,090
- Unique tokens: 9,576
- Compression ratio: 3.88

#### Token Frequency Analysis
Top 10 most frequent tokens:
```
[UNK]: 71,084
के: 21,412
में: 15,782
की: 13,014
को: 9,882
से: 9,554
का: 7,753
और: 7,678
ने: 7,103
है।: 6,791
```

#### Visualizations
The token distribution visualization can be found in `results/token_distribution.png`, showing the frequency distribution of tokens in the trained vocabulary.

## Configuration

The tokenizer can be customized with the following parameters:

- `vocab_size`: Size of the vocabulary (default: 10000)
- `min_frequency`: Minimum frequency for a token to be included (default: 2)
- Special tokens: [UNK], [PAD], [BOS], [EOS]
- Pre-initialized with complete Devanagari alphabet

## Dependencies

- tokenizers==0.15.0
- requests==2.31.0
- tqdm (for progress bars)

## License

MIT License

## Acknowledgments

Training data provided by [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download/Hindi).


