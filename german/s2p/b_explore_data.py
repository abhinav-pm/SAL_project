#!/usr/bin/env python3
"""
Script to explore the HuggingFace dataset structure (without decoding audio)
"""
from datasets import load_from_disk
import os

dataset_path = '/scratch/ABHI/common_voice_de_190h'

print("="*70)
print("Exploring Dataset Structure")
print("="*70)

# Load the dataset
print(f"\nLoading dataset from: {dataset_path}")
dataset = load_from_disk(dataset_path)

print(f"\nDataset type: {type(dataset)}")
print(f"Dataset structure: {dataset}")

# Check if it's a DatasetDict or Dataset
if hasattr(dataset, 'keys'):
    print(f"\nSplits available: {list(dataset.keys())}")
    for split_name in dataset.keys():
        split = dataset[split_name]
        print(f"\n{split_name}: {len(split)} samples")
        print(f"  Features: {split.features}")
else:
    print(f"\nTotal samples: {len(dataset)}")
    print(f"Features: {dataset.features}")

# Show sample data WITHOUT decoding audio
print("\n" + "="*70)
print("Sample Data (first 3 examples - audio paths only):")
print("="*70)

if hasattr(dataset, 'keys'):
    # If it's a DatasetDict, use the first split
    first_split = list(dataset.keys())[0]
    sample_data = dataset[first_split]
else:
    sample_data = dataset

# Access raw data without decoding audio
for i in range(min(3, len(sample_data))):
    print(f"\nExample {i+1}:")
    # Get the raw row data without decoding
    row = sample_data._data.slice(i, i+1).to_pydict()
    
    for key in row.keys():
        value = row[key][0] if row[key] else None
        
        if key == 'audio':
            # For audio, just show it has data
            print(f"  {key}: [Audio data present]")
        elif key == 'path':
            print(f"  {key}: {value}")
        elif key == 'sentence':
            value_str = str(value)
            if value and len(value_str) > 100:
                value_str = value_str[:100] + "..."
            print(f"  {key}: {value_str}")
        else:
            print(f"  {key}: {value}")

print("\n" + "="*70)
print("Dataset Summary:")
print("="*70)
print(f"Total samples: {len(sample_data)}")
print(f"Columns: {list(sample_data.column_names)}")
print(f"Audio format: {sample_data.features['audio']}")
print("="*70)