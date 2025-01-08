---
title: Hindi BPE Tokenizer
emoji:🇳
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# Hindi BPE Tokenizer

A specialized BPE (Byte Pair Encoding) tokenizer for Hindi text, implemented from scratch with a focus on efficiency and accuracy. The tokenizer includes a user-friendly Streamlit web interface for easy interaction.

## Features

- **Custom BPE Implementation**:
  - 20,245 token vocabulary size (trained)
  - Minimum frequency threshold (2) for better token quality
  - Special token support ([UNK], [PAD], [BOS], [EOS])
  - Efficient batch processing with parallel optimization (32,000+ tokens/s)
  - Advanced Devanagari character handling with normalization
  - Intelligent boosting for common words, syllables, and verb forms
  - Support for Hindi punctuation and special characters
  - Achieved 4.03x compression ratio on 1M sentences

- **Streamlit Web Interface**:
  - Real-time encoding/decoding
  - Copy-to-clipboard functionality
  - Modern, responsive UI with custom styling
  - Helpful tooltips and instructions
  - Informative tabs with usage guidelines
  - Error handling and validation

- **Analysis Tools**:
  - Token frequency analysis
  - Compression ratio calculation
  - Detailed tokenization statistics
  - Common phrase testing with accuracy metrics
  - Performance metrics
  - Token distribution visualization

## Performance Metrics

Based on training with 1M Hindi sentences:

- **Tokenization Performance**:
  - Average tokens per line: 23.99
  - Average word length: 4.17
  - Total tokens processed: 23,994,528
  - Unique tokens in dataset: 17,917
  - Processing speed: ~32,000 tokens/second

- **Accuracy Metrics**:
  - 100% accuracy on common Hindi phrases
  - Average 3.56 tokens per phrase
  - Perfect reconstruction of test phrases
  - Robust handling of various sentence structures

Example tokenization results:
```python
# Input: "मैं आपसे मिलकर खुश हूं"
# Tokens: [2895, 5361, 3783, 3981, 3380]
# Accuracy: 100%

# Input: "धन्यवाद आपका"
# Tokens: [5377, 3482]
# Accuracy: 100%
```

## Installation

1. Clone this repository
2. Install Python dependencies:
```bash
pip install -r requirements.txt
```
3. Install system dependencies (for clipboard functionality):
```bash
# On Ubuntu/Debian
sudo apt-get install xclip xsel
```

## Project Structure

```
HindiBPE_Tokenizer_App/
├── src/
│   ├── hindi_bpe_scratch.py     # Core tokenizer implementation
│   ├── train_hindi_bpe_scratch.py # Training script
│   └── analyze_results.py       # Analysis utilities
├── results/
│   ├── tokenizer_stats.json     # Tokenizer statistics
│   └── hindi_bpe_scratch.pkl    # Trained tokenizer model
├── app.py                       # Streamlit web interface
├── requirements.txt             # Python dependencies
└── packages.txt                 # System dependencies
```

## Usage

### Training the Tokenizer

```python
from src.hindi_bpe_scratch import HindiBPE

# Initialize and train
tokenizer = HindiBPE(vocab_size=5000, min_freq=2)
tokenizer.train("data/hin_mixed_2019_1M/hin_mixed_2019_1M-sentences.txt")

# Save the model
tokenizer.save("results/hindi_bpe_scratch.pkl")
```

The training process includes:
- Vocabulary building with intelligent token selection
- Enhanced merging strategy for meaningful subword units
- Boosting for common words, syllables, and verb forms
- Optimized batch processing for large datasets
- Generation of detailed statistics

### Using the Web Interface

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the displayed URL
3. Enter Hindi text in the left panel to encode
4. Enter token IDs in the right panel to decode
5. Use the copy buttons to transfer results
6. Check the tabs below for usage tips and features

## Dependencies

Python requirements:
- streamlit>=1.28.0
- torch>=2.0.0
- tqdm>=4.65.0
- matplotlib>=3.7.1
- pyperclip>=1.8.2

System requirements (for clipboard functionality):
- xclip
- xsel

## Testing

The tokenizer includes comprehensive testing capabilities:
- Common Hindi phrase evaluation
- Character-level accuracy metrics
- Token frequency analysis
- Compression ratio assessment
- Detailed tokenization statistics

## License

MIT License

## Acknowledgments

Special thanks to all contributors and users who have helped improve this tokenizer.


