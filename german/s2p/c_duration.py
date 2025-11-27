#!/usr/bin/env python3
"""
Calculates the total audio duration for a Hugging Face dataset
that was saved to disk using `dataset.save_to_disk()`.

This script reads the audio bytes directly from the dataset's Arrow files
and uses the soundfile library to efficiently extract the duration without
fully decoding each audio sample.
"""

import os
import io
import soundfile as sf
# Import the Audio feature to disable decoding
from datasets import load_from_disk, Audio as AudioFeature 
from tqdm import tqdm
import warnings

# Ignore specific soundfile warnings about metadata tags if they appear
warnings.filterwarnings("ignore", message="PySoundFile failed to read metadata")

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_PATH = "/scratch/ABHI/common_voice_de_190h" 

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def calculate_duration_from_saved_dataset(dataset_path):
    """
    Loads a dataset from disk and calculates the total duration of its audio content.
    """
    if not os.path.isdir(dataset_path):
        print(f"❌ ERROR: Dataset directory not found at: {dataset_path}")
        return

    print(f"--- Loading dataset from: {dataset_path} ---")
    try:
        dataset = load_from_disk(dataset_path)
        total_samples = len(dataset)
        print(f"✓ Successfully loaded {total_samples:,} samples.")
        
        # =================================================================
        # THE FIX: Explicitly disable audio decoding to avoid FFmpeg error
        # =================================================================
        print("--- Casting 'audio' column to prevent automatic decoding ---")
        dataset = dataset.cast_column("audio", AudioFeature(decode=False))
        print("✓ Casting complete. Proceeding with duration calculation.")
        # =================================================================

    except Exception as e:
        print(f"❌ FAILED to load or cast dataset: {e}")
        return

    print("\n--- Calculating total audio duration ---")
    print("This may take a few minutes for a large dataset...")

    total_seconds = 0
    successful_files = 0
    failed_files = 0
    first_error = None

    for example in tqdm(dataset, desc="Processing samples", unit="sample"):
        try:
            audio_data = example.get('audio')
            if not audio_data:
                failed_files += 1
                if not first_error: first_error = "Sample missing 'audio' key"
                continue

            audio_bytes = audio_data.get('bytes')
            if not audio_bytes:
                failed_files += 1
                if not first_error: first_error = "Audio data missing 'bytes' key"
                continue

            bytes_io = io.BytesIO(audio_bytes)
            info = sf.info(bytes_io)
            total_seconds += info.duration
            successful_files += 1

        except Exception as e:
            failed_files += 1
            if not first_error:
                first_error = f"Error processing a sample: {str(e)[:100]}"

    # --- Print the final summary ---
    total_minutes = total_seconds / 60
    total_hours = total_seconds / 3600

    print("\n" + "="*60)
    print(" " * 20 + "DURATION CALCULATION COMPLETE")
    print("="*60)
    print(f"  Dataset Path:      {dataset_path}")
    print(f"  Total Samples:     {total_samples:,}")
    print(f"  Successfully Read: {successful_files:,}")
    print(f"  Failed/Skipped:    {failed_files:,}")
    if first_error:
        print(f"  First Error Seen:  '{first_error}'")
    print("-" * 60)
    print(f"  Total Duration (Seconds): {total_seconds:,.2f}")
    print(f"  Total Duration (Minutes): {total_minutes:,.2f}")
    print(f"  Total Duration (Hours):   {total_hours:,.2f}")
    print("="*60)

if __name__ == "__main__":
    calculate_duration_from_saved_dataset(DATASET_PATH)