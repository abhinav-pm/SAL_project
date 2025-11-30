#!/usr/bin/env python3
"""
Create 20-hour Subset
- Loads the cleaned 130h training TSV.
- Randomly selects samples until the total duration reaches 20 hours.
- Saves the result as 'train_20h_cleaned.tsv'.

Usage:
    python create_20h_subset.py --tsv_dir /home/abhinav.pm/ABHI/SAL/v2/german_dataset_tsv_splits
"""

import pandas as pd
import os
import argparse
import numpy as np

# Configuration
TARGET_HOURS = 20.0
TARGET_SECONDS = TARGET_HOURS * 3600

def create_subset(input_path, output_path):
    print(f"--- Processing: {input_path} ---")
    
    if not os.path.exists(input_path):
        print(f"❌ ERROR: Input file not found: {input_path}")
        return

    # 1. Load Data
    df = pd.read_csv(input_path, sep='\t')
    print(f"✓ Loaded {len(df):,} samples.")

    # 2. Check for duration column
    if 'duration_s' not in df.columns:
        print("❌ ERROR: 'duration_s' column missing in TSV.")
        print("   Did you run the splitting script I provided earlier?")
        return

    total_available_hours = df['duration_s'].sum() / 3600
    print(f"   Total available duration: {total_available_hours:.2f} hours")

    if total_available_hours < TARGET_HOURS:
        print(f"❌ ERROR: Source file only has {total_available_hours:.2f}h. Cannot create {TARGET_HOURS}h subset.")
        return

    # 3. Shuffle Data
    # We shuffle to ensure the 20h subset is a random representation of the full dataset
    # (mix of speakers, sentence lengths, etc.)
    print("   Shuffling data...")
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 4. Select samples until 20h is reached
    print(f"   Selecting random samples to reach {TARGET_HOURS} hours...")
    
    # Calculate cumulative sum of duration
    df_shuffled['cumsum_duration'] = df_shuffled['duration_s'].cumsum()
    
    # Filter rows where cumulative sum is less than or equal to target
    # We assume the first row that exceeds the target is the cutoff
    df_subset = df_shuffled[df_shuffled['cumsum_duration'] <= TARGET_SECONDS].copy()
    
    # Drop the temporary calculation column
    df_subset = df_subset.drop(columns=['cumsum_duration'])

    # 5. Calculate final stats
    final_duration_sec = df_subset['duration_s'].sum()
    final_duration_hours = final_duration_sec / 3600

    print(f"   ✓ Subset created.")
    print(f"   Samples: {len(df_subset):,}")
    print(f"   Duration: {final_duration_hours:.4f} hours")

    # 6. Save
    df_subset.to_csv(output_path, sep='\t', index=False)
    print(f"   Saved to: {os.path.basename(output_path)}")

def main():
    parser = argparse.ArgumentParser(description="Create 20h subset from training data.")
    parser.add_argument(
        "--tsv_dir",
        type=str,
        default="/home/abhinav.pm/ABHI/SAL/v2/130_hours_exps/german_dataset_130_tsv_splits",
        help="Directory containing the cleaned train TSV."
    )
    args = parser.parse_args()

    print("="*60)
    print(f"Creating {TARGET_HOURS}h Training Subset")
    print("="*60)

    input_file = os.path.join(args.tsv_dir, "train_130_cleaned.tsv")
    output_file = os.path.join(args.tsv_dir, "train_20h_cleaned.tsv")

    create_subset(input_file, output_file)

    print("="*60)
    print("Done! You can now use 'train_20h_cleaned.tsv' for your experiments.")
    print("="*60)

if __name__ == "__main__":
    main()