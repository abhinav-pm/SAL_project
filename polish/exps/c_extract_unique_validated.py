#!/usr/bin/env python3
"""
Extract Unique Samples from validated.tsv
- Removes samples that are already in train/dev/test
- Creates a clean "unique_validated.tsv" file
- Can be used to supplement existing train.tsv

Usage:
    python extract_unique_validated.py
"""

import pandas as pd
import os
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
POLISH_DATA_DIR = "/scratch/priyanka/common_voice_polish23/cv-corpus-23.0-2025-09-05/pl"
OUTPUT_DIR = "/home2/dasari.priyanka/ABHI/SAL/polish/exps"

VALIDATED_TSV = os.path.join(POLISH_DATA_DIR, "validated.tsv")
TRAIN_TSV = os.path.join(POLISH_DATA_DIR, "train.tsv")
DEV_TSV = os.path.join(POLISH_DATA_DIR, "dev.tsv")
TEST_TSV = os.path.join(POLISH_DATA_DIR, "test.tsv")

# ============================================================================

def extract_unique():
    print("="*70)
    print(" "*10 + "EXTRACT UNIQUE SAMPLES FROM VALIDATED.TSV")
    print("="*70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load validated
    print(f"\n[1/4] Loading validated.tsv...")
    df_validated = pd.read_csv(VALIDATED_TSV, sep='\t', low_memory=False)
    print(f"  ✓ Loaded: {len(df_validated):,} rows")
    
    # Load splits
    print(f"\n[2/4] Loading existing splits...")
    
    df_train = pd.read_csv(TRAIN_TSV, sep='\t')
    print(f"  ✓ train.tsv: {len(df_train):,} rows")
    
    df_dev = pd.read_csv(DEV_TSV, sep='\t')
    print(f"  ✓ dev.tsv:   {len(df_dev):,} rows")
    
    df_test = pd.read_csv(TEST_TSV, sep='\t')
    print(f"  ✓ test.tsv:  {len(df_test):,} rows")
    
    # Get filenames from splits
    print(f"\n[3/4] Identifying unique samples...")
    
    split_files = set()
    split_files.update(df_train['path'].unique())
    split_files.update(df_dev['path'].unique())
    split_files.update(df_test['path'].unique())
    
    print(f"  Total files in train/dev/test: {len(split_files):,}")
    
    # Filter validated to keep only unique samples
    df_unique = df_validated[~df_validated['path'].isin(split_files)].copy()
    
    print(f"  Unique samples in validated:   {len(df_unique):,}")
    print(f"  Overlap with splits:           {len(df_validated) - len(df_unique):,}")
    
    # Save
    print(f"\n[4/4] Saving unique samples...")
    
    output_file = os.path.join(OUTPUT_DIR, "unique_validated.tsv")
    df_unique.to_csv(output_file, sep='\t', index=False)
    
    print(f"  ✓ Saved to: {output_file}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    print(f"  Original validated samples:  {len(df_validated):,}")
    print(f"  Samples in train/dev/test:   {len(df_validated) - len(df_unique):,}")
    print(f"  Unique samples extracted:    {len(df_unique):,}")
    print(f"\n  Output: {output_file}")
    
    # Usage recommendation
    print(f"\n{'='*70}")
    print("USAGE")
    print("="*70)
    print(f"\n  You now have:")
    print(f"  1. unique_validated.tsv     ({len(df_unique):,} samples)")
    print(f"  2. train.tsv (original)     ({len(df_train):,} samples)")
    print(f"  3. dev.tsv (original)       ({len(df_dev):,} samples)")
    print(f"  4. test.tsv (original)      ({len(df_test):,} samples)")
    
    print(f"\n  Option A: Use unique_validated.tsv for NEW 130h splits")
    print(f"     → If {len(df_unique):,} samples is enough for 130h")
    print(f"     → Run: python b_polish_extract_splits.py")
    print(f"     → (Modify script to use unique_validated.tsv)")
    
    print(f"\n  Option B: Combine unique + train for larger training set")
    print(f"     → Merge unique_validated.tsv + train.tsv")
    print(f"     → Total: {len(df_unique) + len(df_train):,} samples")
    print(f"     → Keep dev/test as-is")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    extract_unique()