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
- Detailed token frequency analysis
- Results visualization and statistics

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
│   ├── analyze_results.py       # Analysis and visualization
│   └── use_tokenizer.py         # Example usage
├── results/
│   └── tokenizer_stats.json     # Detailed tokenizer statistics
├── hindi_bpe.json               # Trained tokenizer model (HuggingFace)
├── hindi_bpe_scratch.json       # Trained tokenizer model (Scratch)
└── requirements.txt             # Project dependencies
```

## Usage

### Using HuggingFace Implementation

```python
from src.hindi_bpe_trainer import create_hindi_bpe

tokenizer = create_hindi_bpe(
    input_files=["path/to/your/hindi/text.txt"],
    vocab_size=8000,
    min_frequency=2,
    save_path="hindi_bpe.json"
)
```

### Using Scratch Implementation

```python
from src.hindi_bpe_scratch import HindiBPE

# Initialize and train
bpe = HindiBPE(vocab_size=8000, min_freq=2)
bpe.train("path/to/your/hindi/text.txt")

# Save the model
bpe.save("hindi_bpe_scratch.json")

# Load and use
bpe.load("hindi_bpe_scratch.json")
encoded = bpe.encode("आप कैसे हैं?")
decoded = bpe.decode(encoded)
```

## Training Results

### Latest Implementation Results
- Vocabulary size: 8,000
- Total tokens processed: 853,384
- Unique tokens: 7,425
- Compression ratio: 385.83

#### Token Frequency Analysis
Top 20 most frequent tokens:
```
किया: 151,287
के: 21,559
में: 15,833
की: 13,425
को: 9,933
से: 9,650
का: 8,274
और: 7,678
ने: 7,295
है।: 7,062
पर: 6,243
कि: 5,927
है: 5,024
भी: 4,640
कर: 3,647
एक: 3,613
इस: 3,421
नहीं: 3,420
लिए: 3,183
तो: 2,747
```

## Features of the Scratch Implementation

1. **Linguistic-Aware Tokenization**:
   - Pre-initialized with common Hindi words
   - Special handling of Devanagari characters
   - Support for common prefixes and suffixes

2. **Advanced Text Normalization**:
   - NFKC Unicode normalization
   - Special character handling
   - Whitespace normalization
   - Punctuation standardization

3. **Performance Optimizations**:
   - Efficient merge operations
   - Progress tracking with tqdm
   - Memory-efficient processing

4. **Analysis Tools**:
   - Token frequency analysis
   - Compression ratio calculation
   - Detailed statistics generation

## Dependencies

- tokenizers==0.15.0
- tqdm>=4.65.0
- matplotlib>=3.7.1
- requests>=2.31.0
- torch>=2.0.0

## License

MIT License

## Acknowledgments

Training data provided by [Leipzig Corpora Collection](https://wortschatz.uni-leipzig.de/en/download/Hindi).


