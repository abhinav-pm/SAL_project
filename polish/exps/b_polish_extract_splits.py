#!/usr/bin/env python3
"""
Polish Dataset Merger & Splitter
1. Loads original Train/Dev/Test.
2. Augments Train with samples from unique_validated.tsv to reach 130 hours.
3. Trims/Augments Dev and Test to reach 5 hours each.
"""

import os
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
POLISH_DATA_DIR = "/scratch/priyanka/common_voice_polish23/cv-corpus-23.0-2025-09-05/pl"
CLIPS_DIR = os.path.join(POLISH_DATA_DIR, "clips")

# Your calculated unique file from the previous step
UNIQUE_VALIDATED_TSV = "/home2/dasari.priyanka/ABHI/SAL/polish/exps/unique_validated.tsv"

# Output location
OUTPUT_DIR = "/home2/dasari.priyanka/ABHI/SAL/polish/exps/polish_final_splits"

# Targets
TARGET_TRAIN_HOURS = 130.0
TARGET_VALID_HOURS = 5.0
TARGET_TEST_HOURS = 5.0

# ============================================================================

def get_duration(filename):
    """Get duration of an audio file in seconds."""
    try:
        audio_path = os.path.join(CLIPS_DIR, filename)
        if not os.path.exists(audio_path):
            return 0.0
        info = sf.info(audio_path)
        return info.duration
    except:
        return 0.0

def process_dataframe(df, desc="Processing"):
    """Adds duration column to dataframe."""
    durations = []
    valid_indices = []
    
    # Check if duration already exists (e.g. from previous runs)
    if 'duration_s' in df.columns:
        return df[df['duration_s'] > 0].copy()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc, ncols=80):
        d = get_duration(row['path'])
        if d > 0:
            durations.append(d)
            valid_indices.append(idx)
        else:
            # If duration is 0, file is missing or corrupt
            pass
            
    df_clean = df.loc[valid_indices].copy()
    df_clean['duration_s'] = durations
    return df_clean

def main():
    print("="*70)
    print(" "*10 + "POLISH DATASET MERGE & SPLIT")
    print("="*70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Original Splits
    print(f"\n[1/5] Loading Original Common Voice Splits...")
    df_orig_train = pd.read_csv(os.path.join(POLISH_DATA_DIR, "train.tsv"), sep='\t')
    df_orig_dev = pd.read_csv(os.path.join(POLISH_DATA_DIR, "dev.tsv"), sep='\t')
    df_orig_test = pd.read_csv(os.path.join(POLISH_DATA_DIR, "test.tsv"), sep='\t')

    # Calculate durations for originals
    df_orig_train = process_dataframe(df_orig_train, desc="  Calc Train Durations")
    df_orig_dev = process_dataframe(df_orig_dev, desc="  Calc Dev Durations")
    df_orig_test = process_dataframe(df_orig_test, desc="  Calc Test Durations")

    train_dur = df_orig_train['duration_s'].sum() / 3600
    dev_dur = df_orig_dev['duration_s'].sum() / 3600
    test_dur = df_orig_test['duration_s'].sum() / 3600

    print(f"  ✓ Original Train: {train_dur:.2f} hrs")
    print(f"  ✓ Original Dev:   {dev_dur:.2f} hrs")
    print(f"  ✓ Original Test:  {test_dur:.2f} hrs")

    # 2. Load Unique Validated (The fillers)
    print(f"\n[2/5] Loading Unique Validated (Fillers)...")
    if not os.path.exists(UNIQUE_VALIDATED_TSV):
        print("❌ Error: unique_validated.tsv not found.")
        return

    df_unique = pd.read_csv(UNIQUE_VALIDATED_TSV, sep='\t')
    # Shuffle unique to ensure random selection
    df_unique = df_unique.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # We calculate durations on the fly to save time, only processing what we need is hard 
    # so we process all to be safe and sortable
    df_unique = process_dataframe(df_unique, desc="  Calc Unique Durations")
    unique_total_dur = df_unique['duration_s'].sum() / 3600
    print(f"  ✓ Available Unique: {unique_total_dur:.2f} hrs")

    # 3. Construct Validation and Test (5h each)
    # Strategy: Use Original Dev/Test. If too big, trim. If too small, add from unique.
    print(f"\n[3/5] Constructing Dev & Test (Target: {TARGET_VALID_HOURS}h each)...")

    def adjust_split(df_orig, df_pool, target_hours, split_name):
        current_hrs = df_orig['duration_s'].sum() / 3600
        target_sec = target_hours * 3600
        
        final_df = pd.DataFrame()
        remaining_pool = df_pool.copy()

        if current_hrs >= target_hours:
            # Trim existing
            print(f"  -> Trimming {split_name} from {current_hrs:.2f}h to {target_hours}h")
            cumsum = df_orig['duration_s'].cumsum()
            # Find index where duration crosses target
            cutoff_idx = cumsum.searchsorted(target_sec)
            final_df = df_orig.iloc[:cutoff_idx+1].copy()
        else:
            # Add from pool
            needed = target_hours - current_hrs
            print(f"  -> Augmenting {split_name} ({current_hrs:.2f}h) with {needed:.2f}h from unique")
            final_df = df_orig.copy()
            
            # Take from pool
            cumsum_pool = df_pool['duration_s'].cumsum()
            needed_sec = needed * 3600
            if cumsum_pool.max() < needed_sec:
                print(f"  ⚠️ Warning: Not enough data to fill {split_name}")
                cutoff_idx = len(df_pool)
            else:
                cutoff_idx = cumsum_pool.searchsorted(needed_sec)
            
            # Add segments
            to_add = df_pool.iloc[:cutoff_idx+1]
            final_df = pd.concat([final_df, to_add])
            
            # Update pool (remove used)
            remaining_pool = df_pool.iloc[cutoff_idx+1:].copy().reset_index(drop=True)
            
        return final_df, remaining_pool

    # Adjust Dev
    df_final_dev, df_unique = adjust_split(df_orig_dev, df_unique, TARGET_VALID_HOURS, "Validation")
    
    # Adjust Test
    df_final_test, df_unique = adjust_split(df_orig_test, df_unique, TARGET_TEST_HOURS, "Test")

    # 4. Construct Train (130h)
    print(f"\n[4/5] Constructing Train (Target: {TARGET_TRAIN_HOURS}h)...")
    
    df_final_train, df_unique = adjust_split(df_orig_train, df_unique, TARGET_TRAIN_HOURS, "Train")

    # 5. Save
    print(f"\n[5/5] Saving Final Splits...")
    
    # Standardize columns
    cols = ['client_id', 'path', 'sentence', 'duration_s']
    
    def save_tsv(df, filename):
        # Ensure columns exist (client_id might be missing in some rows)
        if 'client_id' not in df.columns:
            df['client_id'] = ""
        
        # Keep only necessary columns + duration for verification
        out_df = df[cols] if set(cols).issubset(df.columns) else df
        
        out_path = os.path.join(OUTPUT_DIR, filename)
        out_df.to_csv(out_path, sep='\t', index=False)
        dur = out_df['duration_s'].sum() / 3600
        print(f"  ✓ Saved {filename} ({len(out_df):,} rows, {dur:.2f} hrs)")

    save_tsv(df_final_train, "train_130h.tsv")
    save_tsv(df_final_dev, "dev_5h.tsv")
    save_tsv(df_final_test, "test_5h.tsv")

    print(f"\n{'='*70}")
    print("DONE!")
    print(f"Data saved to: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()