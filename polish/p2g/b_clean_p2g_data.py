#!/usr/bin/env python3
"""
Clean Phoneme-to-Grapheme TSV Datasets
- Reads train, validation, and test TSV files.
- Normalizes the 'sentence' column by removing leading/trailing punctuation (quotes, dashes) and whitespace.
- Saves the cleaned data to new files with a '_cleaned' suffix.

Usage:
    python clean_p2g_data.py --tsv_dir /path/to/phonemized
"""

import pandas as pd
import os
import argparse
from tqdm import tqdm

def normalize_sentence(sentence):
    """
    Cleans a sentence by removing leading/trailing punctuation and whitespace.
    
    Args:
        sentence (str): The input sentence string.
        
    Returns:
        str: The normalized sentence.
    """
    if not isinstance(sentence, str):
        return sentence
    
    # Define a set of characters to strip from the beginning and end of the string.
    # This includes various types of quotes, dashes, spaces, and tabs.
    chars_to_strip = '"”„«»‘’`—–- \t'
    
    # 1. Strip leading/trailing whitespace.
    # 2. Strip the specified punctuation characters from the ends.
    # 3. Strip whitespace again in case the punctuation was surrounded by it.
    normalized_sentence = sentence.strip().strip(chars_to_strip).strip()
    
    return normalized_sentence

def clean_tsv_file(input_path, output_path):
    """
    Reads a TSV file, cleans the 'sentence' column, and saves it to a new file.
    
    Args:
        input_path (str): Path to the original TSV file.
        output_path (str): Path to save the cleaned TSV file.
    """
    if not os.path.exists(input_path):
        print(f"  - ⚠️  Warning: File not found, skipping: {input_path}")
        return 0, 0
        
    print(f"  - Processing: {os.path.basename(input_path)}")
    
    # Load the TSV file
    df = pd.read_csv(input_path, sep='\t')
    
    # Store original sentence for comparison
    df['original_sentence'] = df['sentence']
    
    # Apply the normalization function
    # Using tqdm to show progress for large files
    tqdm.pandas(desc="    Normalizing sentences")
    df['sentence'] = df['sentence'].progress_apply(normalize_sentence)
    
    # Find how many sentences were actually changed
    changes = (df['sentence'] != df['original_sentence']).sum()
    
    # Drop the temporary column
    df = df.drop(columns=['original_sentence'])
    
    # Save the cleaned dataframe
    df.to_csv(output_path, sep='\t', index=False)
    
    print(f"    ✓ Cleaned file saved to: {os.path.basename(output_path)}")
    print(f"    - Sentences changed: {changes:,} / {len(df):,}")
    
    return len(df), changes

def main():
    parser = argparse.ArgumentParser(description="Clean P2G TSV data files.")
    parser.add_argument(
        "--tsv_dir",
        type=str,
        default="/home/abhinav.pm/ABHI/SAL/v4/phonemized",
        help="Directory containing the phonemized TSV files."
    )
    args = parser.parse_args()
    
    print("="*70)
    print(" " * 20 + "P2G Data Cleaning Script")
    print("="*70)
    print(f"🔍 Looking for TSV files in: {args.tsv_dir}\n")
    
    # Define the original and new filenames
    files_to_clean = {
        "train": "common_voice_pl_train_phoneme.tsv",
        "validation": "common_voice_pl_validation_phoneme.tsv",
        "test": "common_voice_pl_test_phoneme.tsv",
        "train_20h": "common_voice_pl_train_20h_phoneme.tsv" # Also clean the 20h subset
    }
    
    total_samples = 0
    total_changes = 0
    
    # Process each file
    for split, filename in files_to_clean.items():
        input_file = os.path.join(args.tsv_dir, filename)
        
        # Create the new filename with a '_cleaned' suffix
        base, ext = os.path.splitext(filename)
        output_file = os.path.join(args.tsv_dir, f"{base}_cleaned{ext}")
        
        samples, changes = clean_tsv_file(input_file, output_file)
        total_samples += samples
        total_changes += changes
        print("-" * 50)

    print("\n" + "="*70)
    print("🎉 Cleaning Complete!")
    print("="*70)
    print(f"  - Total samples processed: {total_samples:,}")
    print(f"  - Total sentences modified: {total_changes:,}")
    print("\nNext Steps:")
    print("1. Update your training script to use the new '_cleaned.tsv' files.")
    print("   Example in `train_p2g.py`:")
    print("   train_tsv = os.path.join(TSV_DIR, 'common_voice_pl_train_20h_phoneme_cleaned.tsv')")
    print("   valid_tsv = os.path.join(TSV_DIR, 'common_voice_pl_validation_phoneme_cleaned.tsv')")
    print("\n2. Update your evaluation script to use the cleaned test file.")
    print("   test_tsv_path = '.../common_voice_pl_test_phoneme_cleaned.tsv'")
    print("="*70)

if __name__ == "__main__":
    main()