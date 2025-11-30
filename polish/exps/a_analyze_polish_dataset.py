#!/usr/bin/env python3
"""
Analyze Polish Common Voice Dataset
- Checks total duration available in train.tsv
- Analyzes data structure
- Reports statistics

Usage:
    python analyze_polish_dataset.py
"""

import pandas as pd
import os
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
POLISH_DATA_DIR = "/scratch/priyanka/common_voice_polish23/cv-corpus-23.0-2025-09-05/pl"
TRAIN_TSV = os.path.join(POLISH_DATA_DIR, "train.tsv")
DEV_TSV = os.path.join(POLISH_DATA_DIR, "dev.tsv")
TEST_TSV = os.path.join(POLISH_DATA_DIR, "test.tsv")
CLIPS_DIR = os.path.join(POLISH_DATA_DIR, "clips")

# ============================================================================

def analyze_dataset():
    print("="*70)
    print(" "*15 + "POLISH DATASET ANALYSIS")
    print("="*70)
    
    print(f"\n📁 Dataset Location: {POLISH_DATA_DIR}")
    print(f"🎵 Clips Directory: {CLIPS_DIR}")
    
    # Check if directories exist
    if not os.path.exists(POLISH_DATA_DIR):
        print(f"❌ ERROR: Dataset directory not found!")
        return
    
    if not os.path.exists(CLIPS_DIR):
        print(f"❌ ERROR: Clips directory not found!")
        return
    
    print(f"\n✓ Directories exist")
    
    # Find all TSV files in the directory
    import glob
    all_tsv_files = glob.glob(os.path.join(POLISH_DATA_DIR, "*.tsv"))
    
    # Filter out the ones we want to analyze (exclude metadata files)
    skip_files = ['clip_durations.tsv', 'reported.tsv']
    
    splits = {}
    for tsv_path in sorted(all_tsv_files):
        filename = os.path.basename(tsv_path)
        if filename not in skip_files:
            split_name = filename.replace('.tsv', '')
            splits[split_name] = tsv_path
    
    print(f"Found {len(splits)} TSV files to analyze:")
    for name in splits.keys():
        print(f"  - {name}")
    
    print(f"\n{'='*70}")
    print("ANALYZING SPLITS")
    print("="*70)
    
    total_hours = 0
    summary_data = []
    
    for split_name, tsv_path in splits.items():
        print(f"\n--- {split_name.upper()} ---")
        
        if not os.path.exists(tsv_path):
            print(f"  ⚠️  File not found: {tsv_path}")
            continue
        
        # Read TSV (Common Voice format is tab-separated)
        df = pd.read_csv(tsv_path, sep='\t')
        
        print(f"  Rows: {len(df):,}")
        print(f"  Columns: {list(df.columns)}")
        
        # Show first few column names to understand structure
        print(f"\n  Sample data (first row):")
        for col in df.columns[:5]:
            print(f"    {col}: {df[col].iloc[0]}")
        
        # The Common Voice format typically has these columns:
        # client_id, path (filename), sentence, up_votes, down_votes, age, gender, accents, locale
        
        # Check if 'path' column exists (this is the audio filename)
        if 'path' not in df.columns:
            print(f"  ⚠️  WARNING: 'path' column not found!")
            # Try to find the audio filename column
            for col in df.columns:
                if 'mp3' in str(df[col].iloc[0]).lower():
                    print(f"  Found audio filenames in column: '{col}'")
                    df['path'] = df[col]
                    break
        
        # Check for sentence column
        sentence_col = None
        for col in ['sentence', 'text']:
            if col in df.columns:
                sentence_col = col
                break
        
        if sentence_col is None:
            # Find column with text data
            for col in df.columns:
                if isinstance(df[col].iloc[0], str) and len(df[col].iloc[0]) > 20:
                    sentence_col = col
                    print(f"  Found sentences in column: '{col}'")
                    break
        
        # Calculate total duration by checking actual audio files
        print(f"\n  Calculating duration from audio files...")
        
        # Sample 100 files to estimate average duration
        sample_size = min(100, len(df))
        durations = []
        
        import soundfile as sf
        
        for idx in tqdm(range(sample_size), desc=f"  Sampling {split_name}"):
            try:
                # Get filename from the row
                if 'path' in df.columns:
                    filename = df.iloc[idx]['path']
                else:
                    filename = df.iloc[idx, 1]  # Usually second column
                
                audio_path = os.path.join(CLIPS_DIR, filename)
                
                if os.path.exists(audio_path):
                    info = sf.info(audio_path)
                    durations.append(info.duration)
            except Exception as e:
                continue
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            estimated_total_duration = avg_duration * len(df)
            estimated_hours = estimated_total_duration / 3600
            
            print(f"  ✓ Average sample duration: {avg_duration:.2f} seconds")
            print(f"  ✓ Estimated total duration: {estimated_hours:.2f} hours")
            
            total_hours += estimated_hours
            
            summary_data.append({
                'Split': split_name,
                'Samples': f"{len(df):,}",
                'Avg Duration (s)': f"{avg_duration:.2f}",
                'Estimated Hours': f"{estimated_hours:.2f}"
            })
        else:
            print(f"  ⚠️  Could not calculate duration")
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY - ALL SPLITS")
    print("="*70)
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        print(f"\n📊 TOTAL ESTIMATED HOURS: {total_hours:.2f} hours")
        
        # Provide specific recommendations
        print(f"\n{'='*70}")
        print("RECOMMENDATIONS")
        print("="*70)
        
        # Find the largest splits
        splits_by_hours = sorted(summary_data, key=lambda x: float(x['Estimated Hours']), reverse=True)
        
        print(f"\n💡 Recommended approach for {total_hours:.2f}h available:")
        
        if total_hours >= 140:
            print(f"✅ OPTION 1: Use train/dev/test as-is (enough data!)")
            print(f"   - Extract 130h from train")
            print(f"   - Use dev for validation (~5h)")
            print(f"   - Use test for testing (~5h)")
        elif total_hours >= 50:
            print(f"⚠️  OPTION 2: Combine splits (not enough for 130h)")
            print(f"   Suggested approach:")
            
            # Calculate what we can actually do
            usable_for_train = max(0, total_hours - 10)  # Reserve 5h for valid + 5h for test
            
            print(f"   - Combine all available data")
            print(f"   - Create ~{usable_for_train:.0f}h train (from combined data)")
            print(f"   - Create ~5h validation")
            print(f"   - Create ~5h test")
            print(f"\n   Total usable: ~{usable_for_train + 10:.0f}h")
        else:
            print(f"⚠️  WARNING: Limited data available")
            print(f"   Consider:")
            print(f"   - Using smaller train set (e.g., 20h)")
            print(f"   - Combining train/dev/test splits")
            print(f"   - Using all {total_hours:.0f}h for training")
        
        print(f"\n{'='*70}")
        print("LARGEST SPLITS (use these):")
        print("="*70)
        for i, split_info in enumerate(splits_by_hours[:5], 1):
            print(f"{i}. {split_info['Split']:20s} - {split_info['Estimated Hours']:>8s}h ({split_info['Samples']:>10s} samples)")
        
        # Check for validated.tsv specifically
        validated_info = next((s for s in summary_data if 'validated' in s['Split'].lower()), None)
        if validated_info:
            print(f"\n💡 TIP: 'validated' split often has the most data!")
            print(f"   Consider using validated.tsv as your main source")
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    
    if total_hours >= 140:
        print("1. ✅ Run extraction script (polish_02_extract_splits.py)")
        print("2. Extract 130h train + 5h validation + 5h test")
    else:
        print("1. ⚠️  Modify extraction script to use available data")
        print(f"2. Extract ~{max(20, total_hours-10):.0f}h train + 5h validation + 5h test")
        print("3. Or create custom split based on your needs")
    
    print("3. Clean the data")
    print("4. Create 20h subset (if enough data)")
    print("5. Phonemize")
    print("="*70)

if __name__ == "__main__":
    analyze_dataset()