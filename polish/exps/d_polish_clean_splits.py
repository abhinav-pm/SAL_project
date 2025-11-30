#!/usr/bin/env python3
"""
Clean Polish Dataset Splits
- Reads the raw split TSV files
- Normalizes sentences (removes extra punctuation, whitespace)
- Saves cleaned versions

Usage:
    python polish_03_clean_splits.py
"""

import pandas as pd
import os
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
TSV_DIR = "/home2/dasari.priyanka/ABHI/SAL/polish/exps/polish_final_splits"

# ============================================================================

def normalize_sentence(sentence):
    """
    Clean a Polish sentence by removing leading/trailing punctuation and whitespace.
    """
    if not isinstance(sentence, str):
        return sentence
    
    # Characters to strip (Polish-specific quotation marks included)
    chars_to_strip = '""„«»''`—–- \t.,;:!?'
    
    # Strip
    normalized = sentence.strip().strip(chars_to_strip).strip()
    
    return normalized

def clean_file(input_path, output_path):
    """Clean a single TSV file"""
    
    if not os.path.exists(input_path):
        print(f"  ⚠️  File not found, skipping: {input_path}")
        return 0, 0
    
    print(f"\n  Processing: {os.path.basename(input_path)}")
    
    # Read
    df = pd.read_csv(input_path, sep='\t')
    print(f"    Loaded: {len(df):,} samples")
    
    # Track changes
    df['original_sentence'] = df['sentence']
    
    # Clean
    tqdm.pandas(desc="    Cleaning")
    df['sentence'] = df['sentence'].progress_apply(normalize_sentence)
    
    # Count changes
    changes = (df['sentence'] != df['original_sentence']).sum()
    
    # Remove temp column
    df = df.drop(columns=['original_sentence'])
    
    # Remove empty sentences
    df = df[df['sentence'].str.len() > 0]
    
    # Save
    df.to_csv(output_path, sep='\t', index=False)
    
    print(f"    ✓ Saved: {os.path.basename(output_path)}")
    print(f"    ✓ Changes: {changes:,}")
    print(f"    ✓ Final samples: {len(df):,}")
    
    return len(df), changes

def main():
    print("="*70)
    print(" "*18 + "POLISH DATASET CLEANING")
    print("="*70)
    print(f"📂 Directory: {TSV_DIR}\n")
    
    # Files to clean
    files_to_clean = {
        "train": "train_130h.tsv",
        "validation": "dev_5h.tsv",
        "test": "test_5h.tsv"
    }
    
    total_samples = 0
    total_changes = 0
    
    for split, filename in files_to_clean.items():
        input_file = os.path.join(TSV_DIR, filename)
        
        # Output filename
        base, ext = os.path.splitext(filename)
        output_file = os.path.join(TSV_DIR, f"{base}_cleaned{ext}")
        
        samples, changes = clean_file(input_file, output_file)
        total_samples += samples
        total_changes += changes
        
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("🎉 CLEANING COMPLETE!")
    print("="*70)
    print(f"  Total samples: {total_samples:,}")
    print(f"  Total changes: {total_changes:,}")
    print(f"\nNext step: Run polish_04_create_20h_subset.py")
    print("="*70)

if __name__ == "__main__":
    main()