#!/usr/bin/env python3
"""
Check Overlap Between validated.tsv and train/dev/test Splits
- Checks if validated.tsv contains samples from train/dev/test
- Identifies unique samples in validated.tsv (not in train/dev/test)
- Shows statistics and recommendations

Usage:
    python check_validated_overlap.py
"""

import pandas as pd
import os
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
POLISH_DATA_DIR = "/scratch/priyanka/common_voice_polish23/cv-corpus-23.0-2025-09-05/pl"

VALIDATED_TSV = os.path.join(POLISH_DATA_DIR, "validated.tsv")
TRAIN_TSV = os.path.join(POLISH_DATA_DIR, "train.tsv")
DEV_TSV = os.path.join(POLISH_DATA_DIR, "dev.tsv")
TEST_TSV = os.path.join(POLISH_DATA_DIR, "test.tsv")

# ============================================================================

def check_overlap():
    print("="*70)
    print(" "*12 + "VALIDATED.TSV OVERLAP CHECK")
    print("="*70)
    
    # Check all files exist
    files = {
        'validated': VALIDATED_TSV,
        'train': TRAIN_TSV,
        'dev': DEV_TSV,
        'test': TEST_TSV
    }
    
    print(f"\n[1/5] Checking files exist...")
    for name, path in files.items():
        if os.path.exists(path):
            print(f"  ✓ {name}.tsv found")
        else:
            print(f"  ❌ {name}.tsv NOT found: {path}")
            return
    
    # Load all files
    print(f"\n[2/5] Loading TSV files...")
    
    print(f"  Loading validated.tsv...")
    df_validated = pd.read_csv(VALIDATED_TSV, sep='\t', low_memory=False)
    print(f"    ✓ Loaded: {len(df_validated):,} rows")
    
    print(f"  Loading train.tsv...")
    df_train = pd.read_csv(TRAIN_TSV, sep='\t')
    print(f"    ✓ Loaded: {len(df_train):,} rows")
    
    print(f"  Loading dev.tsv...")
    df_dev = pd.read_csv(DEV_TSV, sep='\t')
    print(f"    ✓ Loaded: {len(df_dev):,} rows")
    
    print(f"  Loading test.tsv...")
    df_test = pd.read_csv(TEST_TSV, sep='\t')
    print(f"    ✓ Loaded: {len(df_test):,} rows")
    
    # Use 'path' column as unique identifier (audio filename)
    print(f"\n[3/5] Extracting unique audio filenames...")
    
    validated_files = set(df_validated['path'].unique())
    train_files = set(df_train['path'].unique())
    dev_files = set(df_dev['path'].unique())
    test_files = set(df_test['path'].unique())
    
    print(f"  ✓ Unique files in validated: {len(validated_files):,}")
    print(f"  ✓ Unique files in train:     {len(train_files):,}")
    print(f"  ✓ Unique files in dev:       {len(dev_files):,}")
    print(f"  ✓ Unique files in test:      {len(test_files):,}")
    
    # Check overlaps
    print(f"\n[4/5] Checking for overlaps...")
    
    # Overlap between validated and each split
    overlap_train = validated_files & train_files
    overlap_dev = validated_files & dev_files
    overlap_test = validated_files & test_files
    
    # Combined overlap (files in any of train/dev/test)
    all_split_files = train_files | dev_files | test_files
    overlap_any = validated_files & all_split_files
    
    # Unique to validated (NOT in train/dev/test)
    unique_to_validated = validated_files - all_split_files
    
    print(f"\n  Overlap Analysis:")
    print(f"  {'─'*66}")
    print(f"  validated ∩ train:     {len(overlap_train):,} files")
    print(f"  validated ∩ dev:       {len(overlap_dev):,} files")
    print(f"  validated ∩ test:      {len(overlap_test):,} files")
    print(f"  {'─'*66}")
    print(f"  validated ∩ (any split): {len(overlap_any):,} files")
    print(f"  validated - (any split): {len(unique_to_validated):,} files")
    print(f"  {'─'*66}")
    
    # Calculate percentages
    pct_overlap = (len(overlap_any) / len(validated_files)) * 100
    pct_unique = (len(unique_to_validated) / len(validated_files)) * 100
    
    print(f"\n  Percentage breakdown:")
    print(f"    {pct_overlap:.1f}% of validated is in train/dev/test")
    print(f"    {pct_unique:.1f}% of validated is unique (NOT in splits)")
    
    # Check if train/dev/test are subsets of validated
    print(f"\n[5/5] Checking if splits are subsets of validated...")
    
    train_in_validated = len(train_files - validated_files)
    dev_in_validated = len(dev_files - validated_files)
    test_in_validated = len(test_files - validated_files)
    
    print(f"\n  Files in train but NOT in validated: {train_in_validated:,}")
    print(f"  Files in dev but NOT in validated:   {dev_in_validated:,}")
    print(f"  Files in test but NOT in validated:  {test_in_validated:,}")
    
    if train_in_validated == 0 and dev_in_validated == 0 and test_in_validated == 0:
        print(f"\n  ✅ ALL train/dev/test files are in validated!")
        print(f"     → validated.tsv is a SUPERSET of train/dev/test")
    else:
        print(f"\n  ⚠️  Some split files are NOT in validated")
    
    # Summary and recommendations
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("="*70)
    
    summary_data = {
        'Dataset': ['validated', 'train', 'dev', 'test', 'train+dev+test (combined)', 'unique to validated'],
        'Files': [
            f"{len(validated_files):,}",
            f"{len(train_files):,}",
            f"{len(dev_files):,}",
            f"{len(test_files):,}",
            f"{len(all_split_files):,}",
            f"{len(unique_to_validated):,}"
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    # Recommendations
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS")
    print("="*70)
    
    if pct_overlap > 90:
        print(f"\n✅ SCENARIO 1: validated.tsv ≈ train+dev+test (high overlap)")
        print(f"   Overlap: {pct_overlap:.1f}%")
        print(f"\n   Recommendation:")
        print(f"   → Use validated.tsv to create NEW custom splits")
        print(f"   → This avoids using pre-defined splits")
        print(f"   → Creates 130h train + 5h val + 5h test from validated")
        print(f"\n   Why: Pre-defined splits are too small for 130h training")
        
    elif pct_unique > 50:
        print(f"\n⚠️  SCENARIO 2: validated.tsv has unique data (low overlap)")
        print(f"   Unique: {pct_unique:.1f}%")
        print(f"\n   Recommendation:")
        print(f"   → Extract ONLY unique samples from validated")
        print(f"   → Combine with existing train.tsv for training")
        print(f"   → Keep dev.tsv and test.tsv as-is for evaluation")
        print(f"\n   Why: Preserve original dev/test splits for fair comparison")
        
    else:
        print(f"\n📊 SCENARIO 3: Mixed overlap")
        print(f"   Overlap: {pct_overlap:.1f}%, Unique: {pct_unique:.1f}%")
        print(f"\n   Recommendation:")
        print(f"   → Option A: Use validated.tsv for everything (custom splits)")
        print(f"   → Option B: Combine unique samples with existing splits")
        
    # Specific next steps
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print("="*70)
    
    if len(unique_to_validated) > 50000:  # If significant unique data
        print(f"\n💡 You have {len(unique_to_validated):,} unique samples in validated!")
        print(f"\n   OPTION 1 (Recommended): Use validated.tsv for custom splits")
        print(f"   ✓ Gives you full control over train/val/test split")
        print(f"   ✓ Can create 130h train + 5h val + 5h test")
        print(f"   ✓ All from same source (consistent)")
        print(f"\n   Command:")
        print(f"   python b_polish_extract_splits.py  # Uses validated.tsv")
        
        print(f"\n   OPTION 2: Extract only unique samples from validated")
        print(f"   ✓ Preserves original dev/test splits")
        print(f"   ✓ Adds unique data to training")
        print(f"   ✗ More complex (need to filter validated)")
        print(f"\n   Command:")
        print(f"   python extract_unique_from_validated.py  # Would need to create this")
        
    else:
        print(f"\n   → Use validated.tsv for custom 130h splits")
        print(f"   Command: python b_polish_extract_splits.py")
    
    # Save detailed overlap info
    print(f"\n{'='*70}")
    print("SAVING DETAILED ANALYSIS")
    print("="*70)
    
    output_dir = "/home2/dasari.priyanka/ABHI/SAL/polish/exps"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save overlap files list
    overlap_file = os.path.join(output_dir, "validated_overlap_analysis.txt")
    
    with open(overlap_file, 'w') as f:
        f.write("VALIDATED.TSV OVERLAP ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total files in validated:    {len(validated_files):,}\n")
        f.write(f"Total files in train:        {len(train_files):,}\n")
        f.write(f"Total files in dev:          {len(dev_files):,}\n")
        f.write(f"Total files in test:         {len(test_files):,}\n\n")
        
        f.write(f"Overlap with train:          {len(overlap_train):,}\n")
        f.write(f"Overlap with dev:            {len(overlap_dev):,}\n")
        f.write(f"Overlap with test:           {len(overlap_test):,}\n")
        f.write(f"Overlap with any split:      {len(overlap_any):,}\n\n")
        
        f.write(f"Unique to validated:         {len(unique_to_validated):,}\n\n")
        
        f.write(f"Percentage overlap:          {pct_overlap:.2f}%\n")
        f.write(f"Percentage unique:           {pct_unique:.2f}%\n\n")
        
        f.write("="*70 + "\n\n")
        
        # List some sample overlaps
        if len(overlap_train) > 0:
            f.write("Sample files in BOTH validated AND train (first 10):\n")
            for i, filename in enumerate(list(overlap_train)[:10], 1):
                f.write(f"  {i}. {filename}\n")
            f.write("\n")
        
        if len(unique_to_validated) > 0:
            f.write("Sample files UNIQUE to validated (first 10):\n")
            for i, filename in enumerate(list(unique_to_validated)[:10], 1):
                f.write(f"  {i}. {filename}\n")
            f.write("\n")
    
    print(f"✓ Saved detailed analysis to: {overlap_file}")
    
    print(f"\n{'='*70}")

if __name__ == "__main__":
    check_overlap()