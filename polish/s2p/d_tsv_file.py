from datasets import load_dataset, Audio
import pandas as pd
import os

# Define paths
cache_path = "/scratch/ABHI/huggingface_cache2"
destination_folder = "/home/abhinav.pm/ABHI/SAL/v4/common_voice_pl_tsv_files"

# Create destination folder
os.makedirs(destination_folder, exist_ok=True)

print("--- Loading dataset splits and exporting to TSV ---\n")

# Available splits
available_splits = ['train', 'test', 'validation', 'other', 'invalidated']

for split_name in available_splits:
    try:
        print(f"Processing '{split_name}' split...")
        
        # Load the dataset split
        dataset = load_dataset(
            "fsicoli/common_voice_22_0", 
            "pl", 
            split=split_name, 
            cache_dir=cache_path
        )
        
        # Disable audio decoding
        dataset = dataset.cast_column("audio", Audio(decode=False))
        
        print(f"  Loaded {len(dataset)} examples")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset[:])
        
        # Remove the audio column (it contains binary data)
        if 'audio' in df.columns:
            # Extract just the path from audio column if it exists
            if isinstance(df['audio'].iloc[0], dict) and 'path' in df['audio'].iloc[0]:
                df['audio_path'] = df['audio'].apply(lambda x: x['path'] if isinstance(x, dict) else x)
            df = df.drop(columns=['audio'])
        
        # Save to TSV
        output_file = os.path.join(destination_folder, f"{split_name}.tsv")
        df.to_csv(output_file, sep='\t', index=False)
        
        file_size = os.path.getsize(output_file)
        print(f"  ✓ Saved to: {split_name}.tsv ({file_size / (1024*1024):.2f} MB)")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}\n")
        
    except Exception as e:
        print(f"  ✗ Error with '{split_name}' split: {str(e)}\n")

print("="*60)
print("EXPORT COMPLETE")
print("="*60)
print(f"TSV files saved to: {destination_folder}\n")

# Show what was created
print("Created files:")
for file in sorted(os.listdir(destination_folder)):
    if file.endswith('.tsv'):
        file_path = os.path.join(destination_folder, file)
        file_size = os.path.getsize(file_path)
        
        # Count lines
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f) - 1  # Subtract header
        
        print(f"  {file}")
        print(f"    Size: {file_size / (1024*1024):.2f} MB")
        print(f"    Rows: {line_count:,}")
        print()

print("="*60)