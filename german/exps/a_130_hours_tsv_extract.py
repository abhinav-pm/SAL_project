#!/usr/bin/env python3
"""
Splits a large, unsplit Hugging Face audio dataset (saved to disk) into
train, validation, and test sets based on target durations.
"""

import os
import io
import soundfile as sf
import pandas as pd
import numpy as np
from datasets import load_from_disk, Audio as AudioFeature
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message="PySoundFile failed to read metadata")

# ============================================================================
# CONFIGURATION
# ============================================================================
# Path to the large, unsplit dataset directory
DATASET_PATH = "/scratch/ABHI/common_voice_de_190h"

# Directory where the new TSV files for each split will be saved
OUTPUT_DIR = "/home/abhinav.pm/ABHI/SAL/v2/130_hours_exps/german_dataset_tsv_splits"

# --- SPLIT TARGETS (in hours) ---
# UPDATED TO TOTAL 130 HOURS
TARGET_TRAIN_HOURS = 130.0
TARGET_VALID_HOURS = 5.0
TARGET_TEST_HOURS = 5.0

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def create_duration_based_splits(dataset_path, output_dir, train_h, valid_h, test_h):
    # 1. SETUP & LOAD DATASET
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.isdir(dataset_path):
        print(f"❌ ERROR: Dataset directory not found at: {dataset_path}")
        return

    print(f"--- Loading dataset from: {dataset_path} ---")
    try:
        dataset = load_from_disk(dataset_path)
        dataset = dataset.cast_column("audio", AudioFeature(decode=False))
        print(f"✓ Successfully loaded {len(dataset):,} samples.")
    except Exception as e:
        print(f"❌ FAILED to load dataset: {e}")
        return

    # 2. CALCULATE DURATION
    print("\n--- [Step 1/4] Calculating duration for all samples... ---")
    all_samples_metadata = []
    
    for i in tqdm(range(len(dataset)), desc="Calculating durations"):
        example = dataset[i]
        try:
            audio_data = example.get('audio')
            audio_bytes = audio_data.get('bytes') if audio_data else None

            if not audio_bytes:
                continue

            info = sf.info(io.BytesIO(audio_bytes))
            
            # Save essential metadata
            all_samples_metadata.append({
                'path': example.get('path', ''),
                'sentence': example.get('sentence', ''),
                'client_id': example.get('client_id', ''), # Capture client_id if available
                'duration_s': info.duration
            })
        except Exception:
            continue
    
    total_available_hours = sum(item['duration_s'] for item in all_samples_metadata) / 3600
    print(f"✓ Calculated durations for {len(all_samples_metadata):,} valid samples.")
    print(f"  Total available audio: {total_available_hours:.2f} hours")
    
    required_hours = train_h + valid_h + test_h
    if total_available_hours < required_hours:
        print(f"⚠️ WARNING: You requested {required_hours} hours, but dataset only has {total_available_hours:.2f} hours.")

    # 3. SHUFFLE AND ASSIGN
    print("\n--- [Step 2/4] Shuffling and assigning samples to splits... ---")
    
    target_train_s = train_h * 3600
    target_valid_s = valid_h * 3600
    target_test_s = test_h * 3600

    df = pd.DataFrame(all_samples_metadata)
    
    # Random shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Assign splits
    current_train_s, current_valid_s, current_test_s = 0, 0, 0
    split_assignments = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Assigning splits"):
        duration = row['duration_s']
        if current_train_s < target_train_s:
            split_assignments.append('train')
            current_train_s += duration
        elif current_valid_s < target_valid_s:
            split_assignments.append('validation')
            current_valid_s += duration
        elif current_test_s < target_test_s:
            split_assignments.append('test')
            current_test_s += duration
        else:
            split_assignments.append('other')

    df['split'] = split_assignments

    # 4. REPORT AND SAVE
    print("\n--- [Step 3/4] Generating report... ---")
    summary = []
    for split_name in ['train', 'validation', 'test', 'other']:
        split_df = df[df['split'] == split_name]
        total_duration = split_df['duration_s'].sum()
        summary.append({
            'Split': split_name,
            'Num Samples': len(split_df),
            'Duration (h)': f"{total_duration/3600:,.2f}"
        })

    print(pd.DataFrame(summary).to_string(index=False))

    print(f"\n--- [Step 4/4] Saving split TSV files to: {output_dir} ---")
    for split_name in ['train', 'validation', 'test', 'other']:
        split_df = df[df['split'] == split_name]
        
        # FIX: Keep 'duration_s' in the output so we don't have to calculate it again later
        columns_to_save = ['path', 'sentence', 'duration_s']
        if 'client_id' in split_df.columns:
            columns_to_save.append('client_id')
            
        output_df = split_df[columns_to_save]
        
        output_path = os.path.join(output_dir, f"{split_name}.tsv")
        output_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
        print(f"  ✓ Saved {split_name}.tsv")

    print("\n" + "="*60)
    print("🎉 SPLITTING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    create_duration_based_splits(
        DATASET_PATH,
        OUTPUT_DIR,
        TARGET_TRAIN_HOURS,
        TARGET_VALID_HOURS,
        TARGET_TEST_HOURS
    )