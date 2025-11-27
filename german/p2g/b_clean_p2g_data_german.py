#!/usr/bin/env python3
"""
Clean German Phoneme-to-Grapheme TSV Datasets
- Reads German train, validation, and test TSV files.
- Normalizes the 'sentence' column by removing leading/trailing punctuation and whitespace.
- Saves the cleaned data to new files with a '_cleaned' suffix.

Usage:
    python clean_p2g_data_german.py --tsv_dir /path/to/german_phonemized
"""

import pandas as pd
import os
import argparse
from tqdm import tqdm

def normalize_sentence(sentence):
    """
    Cleans a sentence by removing leading/trailing punctuation and whitespace.
    """
    if not isinstance(sentence, str):
        return sentence
    chars_to_strip = '"”„«»‘’`—–- \t'
    normalized_sentence = sentence.strip().strip(chars_to_strip).strip()
    return normalized_sentence

def clean_tsv_file(input_path, output_path):
    """
    Reads a TSV file, cleans the 'sentence' column, and saves it to a new file.
    """
    if not os.path.exists(input_path):
        print(f"  -⚠️  Warning: File not found, skipping: {input_path}")
        return 0, 0
        
    print(f"  - Processing: {os.path.basename(input_path)}")
    df = pd.read_csv(input_path, sep='\t')
    df['original_sentence'] = df['sentence']
    
    tqdm.pandas(desc="    Normalizing sentences")
    df['sentence'] = df['sentence'].progress_apply(normalize_sentence)
    
    changes = (df['sentence'] != df['original_sentence']).sum()
    df = df.drop(columns=['original_sentence'])
    df.to_csv(output_path, sep='\t', index=False)
    
    print(f"    ✓ Cleaned file saved to: {os.path.basename(output_path)}")
    print(f"    - Sentences changed: {changes:,} / {len(df):,}")
    
    return len(df), changes

def main():
    parser = argparse.ArgumentParser(description="Clean German P2G TSV data files.")
    parser.add_argument(
        "--tsv_dir",
        type=str,
        default="/home/abhinav.pm/ABHI/SAL/v2/german_phoneme_smallset", #<-- UPDATE THIS PATH
        help="Directory containing the German phonemized TSV files."
    )
    args = parser.parse_args()
    
    print("="*70)
    print(" " * 18 + "German P2G Data Cleaning Script")
    print("="*70)
    print(f"🔍 Looking for TSV files in: {args.tsv_dir}\n")
    
    # Define the original and new filenames for GERMAN
    files_to_clean = {
        "train": "common_voice_de_train_phoneme.tsv",
        "validation": "common_voice_de_validation_phoneme.tsv",
        "test": "common_voice_de_test_phoneme.tsv",
        "train_20h": "common_voice_de_train_20h_phoneme.tsv"
    }
    
    total_samples, total_changes = 0, 0
    for split, filename in files_to_clean.items():
        input_file = os.path.join(args.tsv_dir, filename)
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
    print("1. Update your training script (`train_p2g_german.py`) to use the new '_cleaned.tsv' files.")
    print("="*70)

if __name__ == "__main__":
    main()