#!/usr/bin/env python3
"""Test script to understand tokenization of hashhop markers."""

from transformers import AutoTokenizer

# Load the tokenizer used in training
tokenizer = AutoTokenizer.from_pretrained("fla-hub/transformer-1.3B-100B")

# Test strings
test_strings = [
    "COMPLETION:",
    "=",
    "'",
    "' ",
    " '",
    "= '",
    "COMPLETION:\nHops=1\n\nCoT=False",
    "AbCd = 'XyZw'",
]

print("Tokenization test results:")
print("-" * 60)

for text in test_strings:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    decoded_tokens = [tokenizer.decode([t]) for t in tokens]

    print(f"Text: {repr(text)}")
    print(f"Token IDs: {tokens}")
    print(f"Decoded tokens: {decoded_tokens}")
    print("-" * 60)

# Also test a full example
full_example = """AbCd = 'XyZw'
KmNp = 'QrSt'

COMPLETION:
Hops=1

CoT=False

KmNp = 'QrSt'
AbCd = 'XyZw'"""

print("\nFull example tokenization:")
print("=" * 60)
tokens = tokenizer.encode(full_example, add_special_tokens=False)
print(f"Total tokens: {len(tokens)}")

# Find COMPLETION: position
completion_text = "COMPLETION:"
completion_tokens = tokenizer.encode(completion_text, add_special_tokens=False)
print(f"\nCOMPLETION: tokens: {completion_tokens}")

# Search for it in the full sequence
for i in range(len(tokens) - len(completion_tokens) + 1):
    if tokens[i : i + len(completion_tokens)] == completion_tokens:
        print(f"Found COMPLETION: at position {i}")
        break
