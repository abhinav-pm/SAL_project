#!/usr/bin/env python3
"""
Calculate audio durations for train/validation/test splits
based on TSV files and a Hugging Face dataset saved to disk.
"""

import os
import io
import pandas as pd
import soundfile as sf
from datasets import load_from_disk, Audio as AudioFeature
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message="PySoundFile failed to read metadata")

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_PATH = "/scratch/ABHI/common_voice_de_190h"

# Update these paths to your actual TSV files
TRAIN_TSV_PATH = "/home/abhinav.pm/ABHI/SAL/v2/german_phoneme_smallset/common_voice_de_train_phoneme.tsv"
VALID_TSV_PATH = "/home/abhinav.pm/ABHI/SAL/v2/german_phoneme_smallset/common_voice_de_validation_phoneme.tsv"
TEST_TSV_PATH = "/home/abhinav.pm/ABHI/SAL/v2/german_phoneme_smallset/common_voice_de_test_phoneme.tsv"  # Add if you have test split

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_tsv_filenames(tsv_path):
    """Load TSV and extract filenames"""
    if not os.path.exists(tsv_path):
        print(f"   ⚠ TSV not found: {tsv_path}")
        return set()
    
    df = pd.read_csv(tsv_path, sep='\t')
    filenames = set(df['path'].apply(os.path.basename))
    return filenames

def calculate_split_duration(dataset, split_filenames, split_name):
    """Calculate duration for a specific split"""
    if not split_filenames:
        print(f"\n❌ No files found for {split_name} split")
        return 0, 0, 0
    
    print(f"\n--- Calculating duration for {split_name.upper()} split ---")
    print(f"   Target samples: {len(split_filenames):,}")
    
    total_seconds = 0
    successful_files = 0
    failed_files = 0
    matched_files = 0
    first_error = None
    
    for example in tqdm(dataset, desc=f"   Processing {split_name}", unit="sample"):
        try:
            filename = os.path.basename(example['path'])
            
            # Check if this file belongs to current split
            if filename not in split_filenames:
                continue
            
            matched_files += 1
            
            audio_data = example.get('audio')
            if not audio_data:
                failed_files += 1
                if not first_error:
                    first_error = "Sample missing 'audio' key"
                continue
            
            audio_bytes = audio_data.get('bytes')
            if not audio_bytes:
                failed_files += 1
                if not first_error:
                    first_error = "Audio data missing 'bytes' key"
                continue
            
            bytes_io = io.BytesIO(audio_bytes)
            info = sf.info(bytes_io)
            total_seconds += info.duration
            successful_files += 1
        
        except Exception as e:
            failed_files += 1
            if not first_error:
                first_error = f"Error: {str(e)[:100]}"
    
    # Print summary for this split
    total_minutes = total_seconds / 60
    total_hours = total_seconds / 3600
    
    print(f"\n   {split_name.upper()} Split Summary:")
    print(f"   {'─'*50}")
    print(f"   Samples in TSV:        {len(split_filenames):,}")
    print(f"   Samples matched:       {matched_files:,}")
    print(f"   Successfully read:     {successful_files:,}")
    print(f"   Failed/Skipped:        {failed_files:,}")
    if first_error:
        print(f"   First error:           {first_error}")
    print(f"   {'─'*50}")
    print(f"   Duration (seconds):    {total_seconds:,.2f}")
    print(f"   Duration (minutes):    {total_minutes:,.2f}")
    print(f"   Duration (hours):      {total_hours:,.2f}")
    
    if matched_files != len(split_filenames):
        print(f"   ⚠ WARNING: Expected {len(split_filenames)} samples, found {matched_files}")
    
    return total_seconds, successful_files, failed_files

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    print("="*70)
    print(" "*15 + "SPLIT DURATION CALCULATOR")
    print("="*70)
    
    # Load dataset
    print(f"\n[1/4] Loading dataset from: {DATASET_PATH}")
    if not os.path.isdir(DATASET_PATH):
        print(f"❌ ERROR: Dataset directory not found!")
        return
    
    try:
        dataset = load_from_disk(DATASET_PATH)
        print(f"   ✓ Loaded {len(dataset):,} total samples")
        
        # Disable audio decoding to avoid FFmpeg errors
        dataset = dataset.cast_column("audio", AudioFeature(decode=False))
        print(f"   ✓ Audio decoding disabled (using bytes directly)")
    except Exception as e:
        print(f"❌ FAILED to load dataset: {e}")
        return
    
    # Load TSV files
    print(f"\n[2/4] Loading TSV files...")
    train_files = load_tsv_filenames(TRAIN_TSV_PATH)
    valid_files = load_tsv_filenames(VALID_TSV_PATH)
    test_files = load_tsv_filenames(TEST_TSV_PATH)
    
    print(f"   ✓ Train TSV: {len(train_files):,} files")
    print(f"   ✓ Valid TSV: {len(valid_files):,} files")
    print(f"   ✓ Test TSV:  {len(test_files):,} files")
    
    # Calculate durations
    print(f"\n[3/4] Calculating durations for each split...")
    
    train_duration, train_success, train_failed = calculate_split_duration(
        dataset, train_files, "train")
    
    valid_duration, valid_success, valid_failed = calculate_split_duration(
        dataset, valid_files, "validation")
    
    test_duration, test_success, test_failed = calculate_split_duration(
        dataset, test_files, "test")
    
    # Final summary
    print("\n" + "="*70)
    print(" "*20 + "FINAL SUMMARY")
    print("="*70)
    
    total_duration = train_duration + valid_duration + test_duration
    total_samples = train_success + valid_success + test_success
    total_failed = train_failed + valid_failed + test_failed
    
    print(f"\n{'Split':<15} {'Samples':<12} {'Duration (h)':<15} {'Duration (m)':<15}")
    print("─"*70)
    print(f"{'Train':<15} {train_success:<12,} {train_duration/3600:<15.2f} {train_duration/60:<15.2f}")
    print(f"{'Validation':<15} {valid_success:<12,} {valid_duration/3600:<15.2f} {valid_duration/60:<15.2f}")
    print(f"{'Test':<15} {test_success:<12,} {test_duration/3600:<15.2f} {test_duration/60:<15.2f}")
    print("─"*70)
    print(f"{'TOTAL':<15} {total_samples:<12,} {total_duration/3600:<15.2f} {total_duration/60:<15.2f}")
    
    if total_failed > 0:
        print(f"\n⚠ Total failed/skipped samples: {total_failed:,}")
    
    print("\n" + "="*70)
    print("✓ Duration calculation complete!")
    print("="*70)

if __name__ == "__main__":
    main()