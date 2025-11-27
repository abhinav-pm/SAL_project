#!/usr/bin/env python3
"""
Prepare P2G Training Data
- Analyzes phonemized TSV files with duration statistics
- Creates 20-hour subset for low-resource experiments (matching paper)
- Shows train/validation/test split statistics

Usage:
    python prepare_p2g_data.py --tsv_dir /path/to/phonemized --create_20h_subset
"""

import pandas as pd
import os
import argparse
from tqdm import tqdm
import numpy as np

def calculate_duration_from_audio_path(audio_path):
    """
    Calculate duration from audio file
    Returns duration in seconds
    """
    try:
        import librosa
        duration = librosa.get_duration(path=audio_path)
        return duration
    except:
        return None

def analyze_tsv_with_duration(tsv_path, cache_path=None):
    """
    Analyze TSV file and calculate total duration
    
    Args:
        tsv_path (str): Path to TSV file
        cache_path (str): Path to Common Voice audio cache
    
    Returns:
        tuple: (df, total_duration_hours)
    """
    print(f"\nAnalyzing: {os.path.basename(tsv_path)}")
    
    # Load TSV
    df = pd.read_csv(tsv_path, sep='\t')
    df = df.dropna(subset=['phonemes', 'sentence'])
    df = df[(df['phonemes'] != "") & (df['sentence'] != "")]
    
    print(f"  Samples: {len(df):,}")
    
    # Try to calculate duration
    total_duration = 0
    duration_available = False
    
    # Check if 'duration' column exists
    if 'duration' in df.columns:
        total_duration = df['duration'].sum()
        duration_available = True
    
    # If audio paths are available and cache path is provided
    elif cache_path and os.path.exists(cache_path):
        print(f"  Calculating duration from audio files...")
        durations = []
        
        for idx in tqdm(range(min(100, len(df))), desc="  Sampling durations", ncols=70):
            row = df.iloc[idx]
            audio_path = row.get('audio_path', row.get('path', ''))
            
            if audio_path:
                # Try to find audio file
                full_path = os.path.join(cache_path, os.path.basename(audio_path))
                if os.path.exists(full_path):
                    dur = calculate_duration_from_audio_path(full_path)
                    if dur:
                        durations.append(dur)
        
        if durations:
            # Estimate total duration from sample
            avg_duration = np.mean(durations)
            total_duration = avg_duration * len(df)
            duration_available = True
            print(f"  Estimated from {len(durations)} samples")
    
    total_hours = total_duration / 3600 if duration_available else 0
    total_minutes = total_duration / 60 if duration_available else 0
    
    return df, total_hours, total_minutes, duration_available


def create_20h_subset(df_train, output_path, cache_path=None):
    """
    Create a 20-hour subset from training data
    
    Args:
        df_train (DataFrame): Training data
        output_path (str): Output TSV path
        cache_path (str): Path to audio cache
    
    Returns:
        DataFrame: 20-hour subset
    """
    print(f"\n{'='*70}")
    print("Creating 20-hour training subset (matching paper's low-resource exp)")
    print("="*70)
    
    target_seconds = 20 * 3600  # 20 hours
    
    # If duration column exists
    if 'duration' in df_train.columns:
        print(f"  Using 'duration' column from TSV")
        
        # Sort by duration (prefer longer samples for efficiency)
        df_sorted = df_train.sort_values('duration', ascending=False).reset_index(drop=True)
        
        cumsum_duration = df_sorted['duration'].cumsum()
        n_samples = (cumsum_duration <= target_seconds).sum() + 1
        
        df_20h = df_sorted.head(n_samples)
        actual_duration = df_20h['duration'].sum()
        
    # Otherwise, estimate based on sample durations
    elif cache_path and os.path.exists(cache_path):
        print(f"  Calculating durations from audio files...")
        
        # Sample to estimate average duration
        sample_size = min(1000, len(df_train))
        durations = []
        
        for idx in tqdm(range(sample_size), desc="  Sampling", ncols=70):
            row = df_train.iloc[idx]
            audio_path = row.get('audio_path', row.get('path', ''))
            
            if audio_path:
                full_path = os.path.join(cache_path, os.path.basename(audio_path))
                if os.path.exists(full_path):
                    dur = calculate_duration_from_audio_path(full_path)
                    if dur:
                        durations.append(dur)
        
        if not durations:
            print(f"  ⚠ Warning: Could not calculate durations")
            print(f"  Using approximate: {int(target_seconds / 5)} samples (~5s each)")
            df_20h = df_train.head(int(target_seconds / 5))
            actual_duration = len(df_20h) * 5
        else:
            avg_duration = np.mean(durations)
            n_samples = int(target_seconds / avg_duration)
            
            df_20h = df_train.head(n_samples)
            actual_duration = len(df_20h) * avg_duration
    
    else:
        # Fallback: assume average 5 seconds per sample
        print(f"  ⚠ Using fallback: assuming ~5s per sample")
        n_samples = int(target_seconds / 5)
        df_20h = df_train.head(n_samples)
        actual_duration = target_seconds
    
    # Save subset
    df_20h.to_csv(output_path, sep='\t', index=False)
    
    actual_hours = actual_duration / 3600
    actual_minutes = actual_duration / 60
    
    print(f"\n  ✓ Created 20h subset:")
    print(f"     Samples: {len(df_20h):,}")
    print(f"     Duration: {actual_hours:.2f} hours ({actual_minutes:.2f} minutes)")
    print(f"     Saved to: {output_path}")
    
    return df_20h


def main():
    parser = argparse.ArgumentParser(description="Analyze and prepare P2G training data")
    parser.add_argument(
        "--tsv_dir",
        type=str,
        default="/home/abhinav.pm/ABHI/SAL/v4/phonemized",
        help="Directory containing phonemized TSV files"
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="/scratch/ABHI/huggingface_cache2",
        help="Path to Common Voice audio cache (for duration calculation)"
    )
    parser.add_argument(
        "--create_20h_subset",
        action="store_true",
        help="Create 20-hour training subset"
    )
    
    args = parser.parse_args()
    
    # File paths
    train_tsv = os.path.join(args.tsv_dir, "common_voice_pl_train_phoneme.tsv")
    valid_tsv = os.path.join(args.tsv_dir, "common_voice_pl_validation_phoneme.tsv")
    test_tsv = os.path.join(args.tsv_dir, "common_voice_pl_test_phoneme.tsv")
    
    # Check files exist
    for tsv_file in [train_tsv, valid_tsv, test_tsv]:
        if not os.path.exists(tsv_file):
            print(f"❌ ERROR: File not found: {tsv_file}")
            return
    
    print("="*70)
    print(" "*20 + "P2G Data Analysis")
    print("="*70)
    
    # Analyze each split
    results = {}
    
    for split_name, tsv_path in [
        ("Train", train_tsv),
        ("Validation", valid_tsv),
        ("Test", test_tsv)
    ]:
        df, hours, minutes, has_duration = analyze_tsv_with_duration(
            tsv_path, 
            args.cache_path if os.path.exists(args.cache_path) else None
        )
        
        results[split_name] = {
            'df': df,
            'samples': len(df),
            'hours': hours,
            'minutes': minutes,
            'has_duration': has_duration
        }
    
    # Display summary table
    print(f"\n{'='*70}")
    print("Dataset Statistics")
    print("="*70)
    print(f"{'Split':<15} {'Samples':<12} {'Duration (h)':<15} {'Duration (m)':<15}")
    print("─" * 70)
    
    total_samples = 0
    total_hours = 0
    total_minutes = 0
    
    for split_name in ["Train", "Validation", "Test"]:
        r = results[split_name]
        total_samples += r['samples']
        total_hours += r['hours']
        total_minutes += r['minutes']
        
        if r['has_duration']:
            print(f"{split_name:<15} {r['samples']:,<12} {r['hours']:<15.2f} {r['minutes']:<15.2f}")
        else:
            print(f"{split_name:<15} {r['samples']:,<12} {'N/A':<15} {'N/A':<15}")
    
    print("─" * 70)
    
    if results['Train']['has_duration']:
        print(f"{'TOTAL':<15} {total_samples:,<12} {total_hours:<15.2f} {total_minutes:<15.2f}")
    else:
        print(f"{'TOTAL':<15} {total_samples:,<12} {'N/A':<15} {'N/A':<15}")
    
    print("="*70)
    
    # Show example data
    print(f"\nExample data from training set:")
    df_train = results['Train']['df']
    for i in range(min(3, len(df_train))):
        phoneme = df_train.iloc[i]['phonemes']
        text = df_train.iloc[i]['sentence']
        phoneme_short = phoneme[:50] + "..." if len(phoneme) > 50 else phoneme
        text_short = text[:50] + "..." if len(text) > 50 else text
        print(f"\n{i+1}. Phonemes: {phoneme_short}")
        print(f"   Text:     {text_short}")
        if 'duration' in df_train.columns:
            print(f"   Duration: {df_train.iloc[i]['duration']:.2f}s")
    
    # Create 20h subset if requested
    if args.create_20h_subset:
        output_20h = os.path.join(args.tsv_dir, "common_voice_pl_train_20h_phoneme.tsv")
        df_20h = create_20h_subset(
            results['Train']['df'],
            output_20h,
            args.cache_path if os.path.exists(args.cache_path) else None
        )
        
        print(f"\n{'='*70}")
        print("✓ 20-hour subset created!")
        print("="*70)
        print(f"\nTo train P2G with 20h data, update train_p2g.py:")
        print(f"  train_tsv = os.path.join(TSV_DIR, 'common_voice_pl_train_20h_phoneme.tsv')")
        print(f"\nThis matches the paper's low-resource experiment (20h)!")
        print("="*70)
    
    # Recommendations
    print(f"\n{'='*70}")
    print("Recommendations")
    print("="*70)
    
    if total_hours < 20:
        print("⚠ Your dataset has less than 20 hours")
        print("  - Consider using full dataset for training")
    elif total_hours < 100:
        print("✓ Good dataset size for P2G training")
        print("  - Can create 20h subset for low-resource experiments")
        print("  - Use full data for best results")
    else:
        print("✓ Large dataset - great for P2G training!")
        print("  - Recommend using full data")
        print("  - Can also test 20h subset to match paper")
    
    print("\nTraining options:")
    print("  1. Full data (~130h):  python train_p2g.py")
    print("  2. 20h subset:         python train_p2g.py  # after updating train_tsv path")
    print("="*70)


if __name__ == "__main__":
    main()