from datasets import load_dataset, Audio
import os
import wave
import contextlib

# Define the same cache directory
cache_path = "/scratch/ABHI/huggingface_cache2"

print("--- Loading all dataset splits from cache ---\n")

# Get the correct available splits
available_splits = ['train', 'test', 'validation', 'other', 'invalidated']
loaded_splits = {}

# Try to load each split
for split_name in available_splits:
    try:
        print(f"Attempting to load '{split_name}' split...")
        dataset = load_dataset(
            "fsicoli/common_voice_22_0", 
            "pl", 
            split=split_name, 
            cache_dir=cache_path
        )
        # Disable audio decoding
        dataset = dataset.cast_column("audio", Audio(decode=False))
        loaded_splits[split_name] = dataset
        print(f"✓ '{split_name}' split loaded: {len(dataset)} examples\n")
    except Exception as e:
        print(f"✗ '{split_name}' split not found or error: {str(e)[:150]}\n")

print("="*60)
print("DATASET SPLITS SUMMARY")
print("="*60)

total_duration_seconds = 0
total_duration_hours = 0

# Calculate duration for each split using file-based method
for split_name, dataset in loaded_splits.items():
    print(f"\n--- {split_name.upper()} SPLIT ---")
    print(f"Number of examples: {len(dataset)}")
    
    try:
        print("Calculating duration from audio files...")
        
        total_seconds = 0
        valid_files = 0
        invalid_files = 0
        
        for i, example in enumerate(dataset):
            try:
                # Get the audio file path
                audio_path = example['audio']['path']
                
                # Check if it's an absolute path or relative
                if not os.path.isabs(audio_path):
                    # Try to find it in the cache directory
                    # Common Voice typically stores files in a clips subdirectory
                    possible_paths = [
                        os.path.join(cache_path, audio_path),
                        os.path.join(cache_path, 'clips', audio_path),
                        audio_path
                    ]
                    
                    audio_path = None
                    for p in possible_paths:
                        if os.path.exists(p):
                            audio_path = p
                            break
                
                if audio_path and os.path.exists(audio_path):
                    # Get duration using wave library (for WAV files) or file info
                    if audio_path.endswith('.wav'):
                        with contextlib.closing(wave.open(audio_path, 'r')) as f:
                            frames = f.getnframes()
                            rate = f.getframerate()
                            duration = frames / float(rate)
                            total_seconds += duration
                            valid_files += 1
                    elif audio_path.endswith('.mp3'):
                        # For MP3, we'll need to use a different approach
                        # Let's try using librosa or pydub if available
                        try:
                            import librosa
                            duration = librosa.get_duration(path=audio_path)
                            total_seconds += duration
                            valid_files += 1
                        except ImportError:
                            # If librosa not available, estimate from file size
                            # MP3 bitrate is typically 128 kbps for Common Voice
                            file_size = os.path.getsize(audio_path)
                            # Rough estimate: 128 kbps = 16 KB/s
                            estimated_duration = file_size / (16 * 1024)
                            total_seconds += estimated_duration
                            valid_files += 1
                else:
                    invalid_files += 1
                
                # Progress indicator
                if (i + 1) % 1000 == 0:
                    print(f"  Processed {i + 1}/{len(dataset)} examples... (Valid: {valid_files}, Invalid: {invalid_files})")
                    
            except Exception as e:
                invalid_files += 1
                if invalid_files == 1:  # Print first error for debugging
                    print(f"  Error with file {i}: {str(e)[:100]}")
        
        total_hours = total_seconds / 3600
        total_minutes = total_seconds / 60
        
        print(f"\nResults:")
        print(f"  Valid files processed: {valid_files}/{len(dataset)}")
        print(f"  Invalid/missing files: {invalid_files}/{len(dataset)}")
        print(f"\nTotal duration:")
        print(f"  {total_seconds:.2f} seconds")
        print(f"  {total_minutes:.2f} minutes")
        print(f"  {total_hours:.2f} hours")
        
        total_duration_seconds += total_seconds
        total_duration_hours += total_hours
        
    except Exception as e:
        print(f"Error calculating duration for {split_name}: {str(e)}")

print("\n" + "="*60)
print("OVERALL SUMMARY")
print("="*60)
print(f"Total splits found: {len(loaded_splits)}")
print(f"Total examples across all splits: {sum(len(ds) for ds in loaded_splits.values())}")
print(f"\nTotal duration across all splits:")
print(f"  {total_duration_seconds:.2f} seconds")
print(f"  {total_duration_seconds/60:.2f} minutes")
print(f"  {total_duration_hours:.2f} hours")
print("="*60)