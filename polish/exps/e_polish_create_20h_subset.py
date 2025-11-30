#!/usr/bin/env python3
"""
Create 20h Polish Subset
- Loads the cleaned 130h Polish training TSV
- Randomly selects samples until 20 hours
- Saves as 'train_20h_cleaned.tsv'

Usage:
    python polish_04_create_20h_subset.py
"""

import pandas as pd
import os
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
TSV_DIR = "/home2/dasari.priyanka/ABHI/SAL/polish/exps/polish_final_splits"
TARGET_HOURS = 20.0
TARGET_SECONDS = TARGET_HOURS * 3600

# ============================================================================

def create_subset():
    print("="*70)
    print(f" "*20 + f"CREATE {TARGET_HOURS}H POLISH SUBSET")
    print("="*70)
    
    input_file = os.path.join(TSV_DIR, "train_130h_cleaned.tsv")
    output_file = os.path.join(TSV_DIR, "train_20h_cleaned.tsv")
    
    print(f"\nInput:  {input_file}")
    print(f"Output: {output_file}")
    
    # Check input
    if not os.path.exists(input_file):
        print(f"\n❌ ERROR: Input file not found!")
        print(f"   Expected: {input_file}")
        print(f"\n   Did you run polish_03_clean_splits.py first?")
        return
    
    # Load
    print(f"\n[1/4] Loading data...")
    df = pd.read_csv(input_file, sep='\t')
    print(f"✓ Loaded {len(df):,} samples")
    
    # Check duration column
    if 'duration_s' not in df.columns:
        print(f"\n❌ ERROR: 'duration_s' column missing!")
        print(f"   Columns found: {list(df.columns)}")
        print(f"\n   Did you run polish_02_extract_splits.py correctly?")
        return
    
    total_hours = df['duration_s'].sum() / 3600
    print(f"✓ Total duration: {total_hours:.2f} hours")
    
    if total_hours < TARGET_HOURS:
        print(f"\n❌ ERROR: Not enough data!")
        print(f"   Available: {total_hours:.2f}h")
        print(f"   Required: {TARGET_HOURS}h")
        return
    
    # Shuffle
    print(f"\n[2/4] Shuffling data...")
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Select samples
    print(f"\n[3/4] Selecting samples to reach {TARGET_HOURS}h...")
    
    df_shuffled['cumsum_duration'] = df_shuffled['duration_s'].cumsum()
    df_subset = df_shuffled[df_shuffled['cumsum_duration'] <= TARGET_SECONDS].copy()
    df_subset = df_subset.drop(columns=['cumsum_duration'])
    
    # Stats
    final_duration_h = df_subset['duration_s'].sum() / 3600
    
    print(f"✓ Subset created:")
    print(f"  Samples: {len(df_subset):,}")
    print(f"  Duration: {final_duration_h:.4f} hours")
    
    # Save
    print(f"\n[4/4] Saving...")
    df_subset.to_csv(output_file, sep='\t', index=False)
    print(f"✓ Saved to: {os.path.basename(output_file)}")
    
    print(f"\n{'='*70}")
    print("🎉 SUBSET CREATION COMPLETE!")
    print("="*70)
    print(f"Output: {output_file}")
    print(f"\nNext step: Phonemize the data")
    print("  - train_130_cleaned.tsv")
    print("  - train_20h_cleaned.tsv")
    print("  - validation_5_cleaned.tsv")
    print("  - test_5_cleaned.tsv")
    print("="*70)

if __name__ == "__main__":
    create_subset()