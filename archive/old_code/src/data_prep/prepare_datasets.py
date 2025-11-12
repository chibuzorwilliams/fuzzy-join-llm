"""
Data Preparation: Create Ciphered and Letter-Scrambled Datasets

This script creates two modified versions of each dataset:
1. CIPHER: All letters are substituted using a fixed random mapping (A→X, B→Q, etc.)
2. LETTER-SCRAMBLED: Letters within each word are shuffled (but words stay in order)

Mathematical property: Ciphered data should have IDENTICAL string distances!
"""

import pandas as pd
import random
import string
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

# Create cipher mapping (A→random letter, B→different random letter, etc.)
letters = list(string.ascii_lowercase)
shuffled = letters.copy()
random.shuffle(shuffled)
CIPHER_MAP = dict(zip(letters, shuffled))

# Also map uppercase
CIPHER_MAP.update(dict(zip(string.ascii_uppercase, [c.upper() for c in shuffled])))

print("Cipher mapping created:")
print("Original: abcdefghijklmnopqrstuvwxyz")
print(f"Ciphered: {''.join([CIPHER_MAP[c] for c in 'abcdefghijklmnopqrstuvwxyz'])}")
print()


def apply_cipher(text):
    """Apply cipher to text - substitute each letter."""
    if pd.isna(text):
        return text
    return ''.join(CIPHER_MAP.get(c, c) for c in str(text))


def scramble_letters_in_words(text):
    """Scramble letters WITHIN each word (words stay in order)."""
    if pd.isna(text):
        return text
    
    words = str(text).split()
    scrambled_words = []
    
    for word in words:
        # Separate letters from non-letters (keep punctuation in place)
        letters = [c for c in word if c.isalpha()]
        non_letters_positions = [(i, c) for i, c in enumerate(word) if not c.isalpha()]
        
        # Shuffle letters
        if len(letters) > 1:
            random.shuffle(letters)
        
        # Reconstruct word
        result = list(word)
        letter_idx = 0
        for i, c in enumerate(word):
            if c.isalpha():
                result[i] = letters[letter_idx]
                letter_idx += 1
        
        scrambled_words.append(''.join(result))
    
    return ' '.join(scrambled_words)


def process_dataset(dataset_name, file1, file2, text_column):
    """Process one dataset - create cipher and scrambled versions."""
    print(f"\n{'='*80}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*80}")
    
    base_path = Path(f"data/{dataset_name}")
    output_path = Path("data/processed")
    output_path.mkdir(exist_ok=True)
    
    # Load data (try multiple encodings)
    print(f"Loading {file1}...")
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            df1 = pd.read_csv(base_path / file1, encoding=encoding)
            print(f"  ✓ Loaded with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode {file1} with any common encoding")
    
    print(f"Loading {file2}...")
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            df2 = pd.read_csv(base_path / file2, encoding=encoding)
            print(f"  ✓ Loaded with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode {file2} with any common encoding")
    
    # Handle column name differences
    if text_column not in df1.columns:
        # Amazon-Google uses 'title' for Amazon, 'name' for Google
        if 'title' in df1.columns:
            text_col1 = 'title'
        else:
            raise ValueError(f"Column {text_column} not found in {file1}")
    else:
        text_col1 = text_column
    
    if text_column not in df2.columns:
        if 'name' in df2.columns:
            text_col2 = 'name'
        elif 'title' in df2.columns:
            text_col2 = 'title'
        else:
            raise ValueError(f"Column {text_column} not found in {file2}")
    else:
        text_col2 = text_column
    
    print(f"  File 1: {len(df1)} records, text column: '{text_col1}'")
    print(f"  File 2: {len(df2)} records, text column: '{text_col2}'")
    
    # Create ciphered versions
    print("\nApplying cipher...")
    df1_cipher = df1.copy()
    df1_cipher[text_col1] = df1[text_col1].apply(apply_cipher)
    
    df2_cipher = df2.copy()
    df2_cipher[text_col2] = df2[text_col2].apply(apply_cipher)
    
    # Save ciphered versions
    cipher_file1 = f"{file1.replace('.csv', '')}_ciphered.csv"
    cipher_file2 = f"{file2.replace('.csv', '')}_ciphered.csv"
    
    df1_cipher.to_csv(output_path / cipher_file1, index=False)
    df2_cipher.to_csv(output_path / cipher_file2, index=False)
    
    print(f"  ✓ Saved: {cipher_file1}")
    print(f"  ✓ Saved: {cipher_file2}")
    
    # Create letter-scrambled versions
    print("\nScrambling letters within words...")
    df1_scrambled = df1.copy()
    df1_scrambled[text_col1] = df1[text_col1].apply(scramble_letters_in_words)
    
    df2_scrambled = df2.copy()
    df2_scrambled[text_col2] = df2[text_col2].apply(scramble_letters_in_words)
    
    # Save scrambled versions
    scrambled_file1 = f"{file1.replace('.csv', '')}_scrambled.csv"
    scrambled_file2 = f"{file2.replace('.csv', '')}_scrambled.csv"
    
    df1_scrambled.to_csv(output_path / scrambled_file1, index=False)
    df2_scrambled.to_csv(output_path / scrambled_file2, index=False)
    
    print(f"  ✓ Saved: {scrambled_file1}")
    print(f"  ✓ Saved: {scrambled_file2}")
    
    # Show examples
    print("\nExamples:")
    idx = 0
    original = str(df1[text_col1].iloc[idx])
    ciphered = str(df1_cipher[text_col1].iloc[idx])
    scrambled = str(df1_scrambled[text_col1].iloc[idx])
    
    print(f"  Original:  {original}")
    print(f"  Ciphered:  {ciphered}")
    print(f"  Scrambled: {scrambled}")
    
    return True


# Process all datasets
if __name__ == "__main__":
    print("="*80)
    print("ENTITY MATCHING - DATA PREPARATION")
    print("="*80)
    print("\nThis will create ciphered and letter-scrambled versions of all datasets.")
    print("Files will be saved to: data/processed/")
    print()
    
    datasets = [
        ("abt-buy", "Abt.csv", "Buy.csv", "name"),
        ("amazon-google", "Amazon.csv", "GoogleProducts.csv", "title"),
        ("dblp-acm", "DBLP2.csv", "ACM.csv", "title"),
        ("dblp-scholar", "DBLP1.csv", "Scholar.csv", "title"),
    ]
    
    for dataset_name, file1, file2, text_col in datasets:
        try:
            process_dataset(dataset_name, file1, file2, text_col)
        except Exception as e:
            print(f"\n❌ Error processing {dataset_name}: {e}")
            continue
    
    print("\n" + "="*80)
    print("✅ DATA PREPARATION COMPLETE!")
    print("="*80)
    print("\nGenerated files in data/processed/:")
    print("  - *_ciphered.csv (letter substitution)")
    print("  - *_scrambled.csv (letters shuffled within words)")
    print("\nNext step: Open these CSVs side-by-side to verify transformations!")
