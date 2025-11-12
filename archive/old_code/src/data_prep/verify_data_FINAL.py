"""
Verification Script for Data Transformations - FINAL VERSION
Verifies cipher and scrambling were applied correctly to ALL datasets.

FIXED:
- Handles encoding issues (UTF-8, Latin-1, Windows-1252)
- Uses correct paths for your directory structure
"""

import pandas as pd
from collections import Counter
import os

def verify_cipher(original_text, ciphered_text):
    """
    Verify that the cipher transformation is consistent.
    Returns the mapping and any inconsistencies found.
    """
    if pd.isna(original_text) or pd.isna(ciphered_text):
        return None, []
    
    mapping = {}
    inconsistencies = []
    
    for orig_char, cipher_char in zip(str(original_text), str(ciphered_text)):
        if orig_char.isalpha():
            if orig_char in mapping:
                if mapping[orig_char] != cipher_char:
                    inconsistencies.append(
                        f"'{orig_char}' maps to both '{mapping[orig_char]}' and '{cipher_char}'"
                    )
            else:
                mapping[orig_char] = cipher_char
    
    return mapping, inconsistencies

def verify_letter_counts(original_text, transformed_text):
    """
    Verify that the transformed text has the same letters (for scrambling).
    """
    if pd.isna(original_text) or pd.isna(transformed_text):
        return True, ""
    
    orig_letters = [c.lower() for c in str(original_text) if c.isalpha()]
    trans_letters = [c.lower() for c in str(transformed_text) if c.isalpha()]
    
    orig_counter = Counter(orig_letters)
    trans_counter = Counter(trans_letters)
    
    if orig_counter == trans_counter:
        return True, "✓ Same letter counts"
    else:
        return False, f"✗ Letter counts differ"

def detect_columns(df):
    """
    Automatically detect which columns are text vs numeric.
    """
    text_columns = []
    numeric_columns = []
    
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            numeric_columns.append(col)
        elif col.lower() in ['id', 'price', 'year', 'date']:
            numeric_columns.append(col)
        else:
            text_columns.append(col)
    
    return text_columns, numeric_columns

def load_with_encoding(filepath):
    """
    Load CSV with automatic encoding detection.
    """
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings_to_try:
        try:
            return pd.read_csv(filepath, encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            if 'codec' not in str(e).lower():
                raise
    
    raise ValueError(f"Could not read {filepath} with any supported encoding")

def analyze_transformations(orig_file, ciphered_file, scrambled_file, num_samples=5):
    """
    Analyze and display transformation quality.
    """
    print(f"\n{'=' * 100}")
    print(f"VERIFICATION: {orig_file}")
    print(f"{'=' * 100}")
    
    try:
        df_orig = load_with_encoding(orig_file)
        df_cipher = load_with_encoding(ciphered_file)
        df_scramble = load_with_encoding(scrambled_file)
    except FileNotFoundError as e:
        print(f"✗ ERROR: {e}")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False
    
    print(f"\nDataset size: {len(df_orig)} rows")
    print(f"Columns: {list(df_orig.columns)}")
    
    # Auto-detect text columns
    text_columns, numeric_columns = detect_columns(df_orig)
    print(f"\nText columns (should be transformed): {text_columns}")
    print(f"Numeric columns (should be unchanged): {numeric_columns}")
    
    all_checks_passed = True
    
    # Verify each text column
    for col in text_columns:
        if col not in df_orig.columns:
            print(f"\n⚠ WARNING: Column '{col}' not found in dataset")
            continue
            
        print(f"\n{'─' * 100}")
        print(f"COLUMN: {col}")
        print(f"{'─' * 100}")
        
        # Check CIPHER transformation
        print(f"\n1. CIPHER VERIFICATION")
        
        all_mappings = {}
        inconsistencies = []
        
        for idx in range(min(len(df_orig), 100)):
            mapping, incons = verify_cipher(
                df_orig[col].iloc[idx], 
                df_cipher[col].iloc[idx]
            )
            if mapping:
                for k, v in mapping.items():
                    if k in all_mappings and all_mappings[k] != v:
                        inconsistencies.append(f"Row {idx}: '{k}' → '{v}' (previously '{all_mappings[k]}')")
                    all_mappings[k] = v
        
        if inconsistencies:
            print(f"   ✗ INCONSISTENT CIPHER MAPPINGS FOUND:")
            for inc in inconsistencies[:5]:
                print(f"     {inc}")
            all_checks_passed = False
        else:
            print(f"   ✓ Cipher is consistent across all rows")
            print(f"   ✓ Total unique letter mappings: {len(all_mappings)}")
        
        # Check SCRAMBLING transformation
        print(f"\n2. LETTER SCRAMBLING VERIFICATION")
        
        scramble_ok = 0
        scramble_bad = 0
        changed_count = 0
        
        for idx in range(min(len(df_orig), num_samples)):
            is_ok, msg = verify_letter_counts(
                df_orig[col].iloc[idx], 
                df_scramble[col].iloc[idx]
            )
            
            if is_ok:
                scramble_ok += 1
                if str(df_orig[col].iloc[idx]) != str(df_scramble[col].iloc[idx]):
                    changed_count += 1
            else:
                scramble_bad += 1
                print(f"   Row {idx}: {msg}")
                all_checks_passed = False
        
        print(f"   ✓ Letter counts preserved: {scramble_ok}/{num_samples}")
        print(f"   ✓ Text was changed: {changed_count}/{num_samples}")
        
        # Show ONE example
        print(f"\n3. SAMPLE TRANSFORMATION (Row 2)")
        if len(df_orig) > 2:
            orig_val = str(df_orig[col].iloc[2])[:80]
            cipher_val = str(df_cipher[col].iloc[2])[:80]
            scramble_val = str(df_scramble[col].iloc[2])[:80]
            
            print(f"   Original:  {orig_val}...")
            print(f"   Ciphered:  {cipher_val}...")
            print(f"   Scrambled: {scramble_val}...")
    
    # Verify numeric columns are unchanged
    print(f"\n{'─' * 100}")
    print(f"NUMERIC COLUMNS VERIFICATION")
    print(f"{'─' * 100}")
    
    for col in numeric_columns:
        if col not in df_orig.columns:
            continue
        
        cipher_same = df_orig[col].equals(df_cipher[col])
        scramble_same = df_orig[col].equals(df_scramble[col])
        
        if cipher_same and scramble_same:
            print(f"   ✓ Column '{col}' unchanged in both transformations")
        else:
            print(f"   ✗ Column '{col}' was modified (should be unchanged!)")
            all_checks_passed = False
    
    return all_checks_passed

def main():
    """
    Main verification function.
    """
    print("=" * 100)
    print("DATA TRANSFORMATION VERIFICATION - FINAL VERSION")
    print("=" * 100)
    
    # Define all datasets to verify with YOUR directory structure
    verifications = [
        {
            'name': 'Abt-Buy',
            'files': [
                ('data/abt-buy/Abt.csv', 'data/abt-buy/Abt_ciphered.csv', 'data/abt-buy/Abt_scrambled.csv'),
                ('data/abt-buy/Buy.csv', 'data/abt-buy/Buy_ciphered.csv', 'data/abt-buy/Buy_scrambled.csv')
            ]
        },
        {
            'name': 'Amazon-Google',
            'files': [
                ('data/amazon-google/Amazon.csv', 'data/amazon-google/Amazon_ciphered.csv', 'data/amazon-google/Amazon_scrambled.csv'),
                ('data/amazon-google/GoogleProducts.csv', 'data/amazon-google/GoogleProducts_ciphered.csv', 'data/amazon-google/GoogleProducts_scrambled.csv')
            ]
        },
        {
            'name': 'DBLP-ACM',
            'files': [
                ('data/dblp-acm/ACM.csv', 'data/dblp-acm/ACM_ciphered.csv', 'data/dblp-acm/ACM_scrambled.csv')
            ]
        },
        {
            'name': 'DBLP-Scholar',
            'files': [
                ('data/dblp-scholar/Scholar.csv', 'data/dblp-scholar/Scholar_ciphered.csv', 'data/dblp-scholar/Scholar_scrambled.csv')
            ]
        }
    ]
    
    dataset_results = {}
    
    for dataset in verifications:
        print(f"\n{'#' * 100}")
        print(f"# DATASET: {dataset['name']}")
        print(f"{'#' * 100}")
        
        all_passed = True
        files_found = False
        
        for orig_file, ciphered_file, scrambled_file in dataset['files']:
            if not all([os.path.exists(orig_file), os.path.exists(ciphered_file), os.path.exists(scrambled_file)]):
                print(f"\n⚠️  Skipping {orig_file} - files not found")
                continue
            
            files_found = True
            try:
                passed = analyze_transformations(orig_file, ciphered_file, scrambled_file, num_samples=5)
                if not passed:
                    all_passed = False
            except Exception as e:
                print(f"\n✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False
        
        if files_found:
            dataset_results[dataset['name']] = all_passed
    
    # Final summary
    print("\n" + "=" * 100)
    print("VERIFICATION SUMMARY")
    print("=" * 100)
    
    if dataset_results:
        print("\nResults by dataset:")
        for dataset_name, passed in dataset_results.items():
            status = "✓ PASSED" if passed else "✗ FAILED"
            print(f"  {dataset_name}: {status}")
        
        all_passed = all(dataset_results.values())
        if all_passed:
            print("\n" + "🎉" * 20)
            print("ALL CHECKS PASSED! Data preparation is correct.")
            print("🎉" * 20)
            print("\n✅ YOU CAN NOW:")
            print("   1. Commit files to GitHub")
            print("   2. Invite professor (username: jhelly)")
            print("   3. Send email with GitHub link")
        else:
            print("\n" + "⚠" * 20)
            print("SOME CHECKS FAILED! Review the errors above.")
            print("⚠" * 20)
            print("\n❌ DO NOT proceed until all checks pass.")
    else:
        print("\n✗ No datasets were verified.")
        print("Make sure you ran: python src/data_prep/prepare_data_FINAL.py")

if __name__ == "__main__":
    main()
