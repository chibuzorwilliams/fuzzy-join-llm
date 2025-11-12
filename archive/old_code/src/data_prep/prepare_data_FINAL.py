"""
Data Preparation Script for Entity Matching Experiment - FINAL VERSION
This script creates ciphered and letter-scrambled versions of ALL datasets.

FIXED:
- Handles encoding issues (UTF-8, Latin-1, Windows-1252)
- Uses correct paths for your directory structure (data/dataset-name/file.csv)
- Transforms ALL text columns (not just one!)

Transformations:
1. CIPHER: Consistent letter substitution (a→y, b→z, etc.)
   - Should NOT affect string distance metrics (Jaro-Winkler, etc.)
2. LETTER SCRAMBLING: Randomly scramble letters within each word
   - SHOULD affect string distance metrics
   - Maintains word boundaries (spaces preserved)
"""

import pandas as pd
import random
import string
from pathlib import Path
import os

def create_cipher_mapping():
    """
    Create a consistent cipher mapping for letters.
    Each letter maps to exactly one other letter.
    This is a substitution cipher.
    """
    letters = list(string.ascii_lowercase)
    shuffled = letters.copy()
    random.seed(42)  # Fixed seed for reproducibility
    random.shuffle(shuffled)
    
    # Create bidirectional mapping (lowercase and uppercase)
    cipher = {}
    for orig, new in zip(letters, shuffled):
        cipher[orig] = new
        cipher[orig.upper()] = new.upper()
    
    return cipher

def apply_cipher(text, cipher_map):
    """
    Apply cipher to text while preserving non-letter characters.
    """
    if pd.isna(text):
        return text
    
    result = []
    for char in str(text):
        if char in cipher_map:
            result.append(cipher_map[char])
        else:
            result.append(char)  # Keep numbers, spaces, punctuation unchanged
    
    return ''.join(result)

def scramble_word(word):
    """
    Scramble letters within a single word.
    Preserves first and last character for readability (optional).
    """
    if len(word) <= 3:
        # Don't scramble very short words
        return word
    
    # Get the letters to scramble (excluding first and last)
    middle = list(word[1:-1])
    random.shuffle(middle)
    
    return word[0] + ''.join(middle) + word[-1]

def scramble_letters(text):
    """
    Scramble letters within each word while maintaining word boundaries.
    Preserves spaces, punctuation, and numbers.
    """
    if pd.isna(text):
        return text
    
    text = str(text)
    result = []
    current_word = []
    
    for char in text:
        if char.isalpha():
            current_word.append(char)
        else:
            # End of word - scramble it
            if current_word:
                word = ''.join(current_word)
                scrambled = scramble_word(word)
                result.append(scrambled)
                current_word = []
            result.append(char)  # Add the non-letter character
    
    # Don't forget the last word if text ended with a letter
    if current_word:
        word = ''.join(current_word)
        scrambled = scramble_word(word)
        result.append(scrambled)
    
    return ''.join(result)

def detect_columns(df):
    """
    Automatically detect which columns are text vs numeric.
    Returns (text_columns, numeric_columns)
    """
    text_columns = []
    numeric_columns = []
    
    for col in df.columns:
        # Check if column is numeric
        if df[col].dtype in ['int64', 'float64']:
            numeric_columns.append(col)
        # Check if column name suggests it's an ID or numeric field
        elif col.lower() in ['id', 'price', 'year', 'date']:
            numeric_columns.append(col)
        else:
            # It's a text column
            text_columns.append(col)
    
    return text_columns, numeric_columns

def transform_dataset(input_file, output_ciphered, output_scrambled, 
                     text_columns=None, numeric_columns=None):
    """
    Transform a dataset by applying cipher and letter scrambling.
    
    Args:
        input_file: Path to original CSV
        output_ciphered: Path to save ciphered version
        output_scrambled: Path to save scrambled version
        text_columns: List of column names to transform (if None, auto-detect)
        numeric_columns: List of column names to keep unchanged (if None, auto-detect)
    """
    print(f"\nProcessing {input_file}...")
    
    # Read the original data - try different encodings
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    df = None
    used_encoding = None
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(input_file, encoding=encoding)
            used_encoding = encoding
            if encoding != 'utf-8':
                print(f"  ℹ️  Using {encoding} encoding")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            # Other errors (like file not found) should propagate
            if 'codec' not in str(e).lower():
                raise
    
    if df is None:
        raise ValueError(f"Could not read file with any supported encoding: {encodings_to_try}")
    
    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows: {len(df)}")
    
    # Auto-detect columns if not specified
    if text_columns is None or numeric_columns is None:
        detected_text, detected_numeric = detect_columns(df)
        text_columns = text_columns or detected_text
        numeric_columns = numeric_columns or detected_numeric
        print(f"  Auto-detected text columns: {text_columns}")
        print(f"  Auto-detected numeric columns: {numeric_columns}")
    
    # Create cipher mapping
    cipher_map = create_cipher_mapping()
    
    # Create ciphered version
    df_ciphered = df.copy()
    for col in text_columns:
        if col in df.columns:
            print(f"  Applying cipher to column: {col}")
            df_ciphered[col] = df[col].apply(lambda x: apply_cipher(x, cipher_map))
    
    # Create scrambled version
    random.seed(42)  # Reset seed for scrambling
    df_scrambled = df.copy()
    for col in text_columns:
        if col in df.columns:
            print(f"  Applying letter scrambling to column: {col}")
            df_scrambled[col] = df[col].apply(scramble_letters)
    
    # Save the transformed datasets using the same encoding
    df_ciphered.to_csv(output_ciphered, index=False, encoding=used_encoding or 'utf-8')
    df_scrambled.to_csv(output_scrambled, index=False, encoding=used_encoding or 'utf-8')
    
    print(f"  ✓ Saved ciphered version to {output_ciphered}")
    print(f"  ✓ Saved scrambled version to {output_scrambled}")
    
    return df, df_ciphered, df_scrambled

def main():
    """
    Main function to process all datasets.
    """
    print("=" * 80)
    print("DATA PREPARATION FOR ENTITY MATCHING EXPERIMENT")
    print("Multi-Dataset Version - FINAL WITH ENCODING FIX")
    print("=" * 80)
    
    # Define all datasets to process with YOUR directory structure
    datasets = [
        {
            'name': 'Abt-Buy',
            'files': [
                ('data/abt-buy/Abt.csv', 'data/abt-buy/Abt_ciphered.csv', 'data/abt-buy/Abt_scrambled.csv'),
                ('data/abt-buy/Buy.csv', 'data/abt-buy/Buy_ciphered.csv', 'data/abt-buy/Buy_scrambled.csv')
            ],
            'text_columns': ['name', 'description'],
            'numeric_columns': ['id', 'price']
        },
        {
            'name': 'Amazon-Google',
            'files': [
                ('data/amazon-google/Amazon.csv', 'data/amazon-google/Amazon_ciphered.csv', 'data/amazon-google/Amazon_scrambled.csv'),
                ('data/amazon-google/GoogleProducts.csv', 'data/amazon-google/GoogleProducts_ciphered.csv', 'data/amazon-google/GoogleProducts_scrambled.csv')
            ],
            'text_columns': None,  # Will auto-detect
            'numeric_columns': None  # Will auto-detect
        },
        {
            'name': 'DBLP-ACM',
            'files': [
                ('data/dblp-acm/ACM.csv', 'data/dblp-acm/ACM_ciphered.csv', 'data/dblp-acm/ACM_scrambled.csv')
            ],
            'text_columns': None,  # Will auto-detect
            'numeric_columns': None  # Will auto-detect
        },
        {
            'name': 'DBLP-Scholar',
            'files': [
                ('data/dblp-scholar/Scholar.csv', 'data/dblp-scholar/Scholar_ciphered.csv', 'data/dblp-scholar/Scholar_scrambled.csv')
            ],
            'text_columns': None,  # Will auto-detect
            'numeric_columns': None  # Will auto-detect
        }
    ]
    
    # Track which datasets were processed
    processed = []
    skipped = []
    
    # Process each dataset
    for dataset in datasets:
        print(f"\n{'=' * 80}")
        print(f"Processing dataset: {dataset['name']}")
        print(f"{'=' * 80}")
        
        dataset_processed = False
        
        for input_file, output_ciphered, output_scrambled in dataset['files']:
            try:
                orig, ciphered, scrambled = transform_dataset(
                    input_file, 
                    output_ciphered, 
                    output_scrambled,
                    dataset['text_columns'],
                    dataset['numeric_columns']
                )
                
                # Show sample of transformations for verification
                print(f"\n  Sample transformations for {input_file}:")
                print(f"  {'─' * 76}")
                
                # Get text columns (either specified or detected)
                if dataset['text_columns'] is None:
                    text_cols, _ = detect_columns(orig)
                else:
                    text_cols = dataset['text_columns']
                
                for col in text_cols[:2]:  # Show first 2 columns only
                    if col in orig.columns:
                        sample_idx = 2 if len(orig) > 2 else 0
                        if len(orig) > sample_idx:
                            orig_text = str(orig[col].iloc[sample_idx])[:80]
                            ciph_text = str(ciphered[col].iloc[sample_idx])[:80]
                            scram_text = str(scrambled[col].iloc[sample_idx])[:80]
                            
                            print(f"\n  Column: {col}")
                            print(f"  Original:  {orig_text}...")
                            print(f"  Ciphered:  {ciph_text}...")
                            print(f"  Scrambled: {scram_text}...")
                
                dataset_processed = True
                
            except FileNotFoundError:
                print(f"  ⚠️  File not found: {input_file} - Skipping")
            except Exception as e:
                print(f"  ✗ ERROR processing {input_file}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if dataset_processed:
            processed.append(dataset['name'])
        else:
            skipped.append(dataset['name'])
    
    # Summary
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    
    if processed:
        print(f"\n✓ Successfully processed datasets:")
        for name in processed:
            print(f"  - {name}")
    
    if skipped:
        print(f"\n⚠️  Skipped datasets (files not found):")
        for name in skipped:
            print(f"  - {name}")
        print(f"\nNote: Some files like DBLP.csv and DBLP2.csv may not exist in your data.")
        print(f"If you only have ACM.csv and Scholar.csv, that's normal - they're already processed!")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Check the output above - all should show '✓ Successfully processed'")
    print("2. Open a ciphered file and verify ALL text columns are transformed")
    print("3. Run: python src/data_prep/verify_data_FINAL.py")
    print("4. If verification passes → commit to GitHub")

if __name__ == "__main__":
    main()
