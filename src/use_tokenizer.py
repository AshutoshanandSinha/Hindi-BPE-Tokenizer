from tokenizers import Tokenizer

# Load the trained tokenizer
tokenizer = Tokenizer.from_file("hindi_bpe.json")

# Example usage
text = "आप कैसे हैं?"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded.ids)

print(f"Original: {text}")
print(f"Encoded: {encoded.ids}")
print(f"Decoded: {decoded}") 