#!/usr/bin/env python3
"""
Speech-to-Phoneme (S2P) Inference Script
Test your trained model on audio files
"""

import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import numpy as np
import os
import soundfile as sf

print("="*70)
print(" "*20 + "S2P Model Inference")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_PATH = "./s2p_model_german_phoneme_full2"  # Your trained model
AUDIO_FILE = "/home/abhinav.pm/ABHI/SAL/common_voice_de_17299916.mp3"  # Audio file to test

# You can also test with audio from your dataset
USE_DATASET_AUDIO = True  # Set to True to test with dataset audio
DATASET_PATH = "/scratch/ABHI/common_voice_de_190h"
TEST_SAMPLE_INDEX = 0  # Which sample from dataset to test

print(f"\n📋 Configuration:")
print(f"   Model path: {MODEL_PATH}")
print(f"   Use dataset audio: {USE_DATASET_AUDIO}")

# ============================================================================
# LOAD MODEL AND PROCESSOR
# ============================================================================
print(f"\n[1/3] Loading trained model and processor...")

if not os.path.exists(MODEL_PATH):
    print(f"   ❌ ERROR: Model not found at {MODEL_PATH}")
    exit(1)

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print(f"   ✓ Model loaded")
print(f"   ✓ Device: {device}")
print(f"   ✓ Vocabulary size: {len(processor.tokenizer)}")

# ============================================================================
# LOAD AUDIO
# ============================================================================
print(f"\n[2/3] Loading audio...")

if USE_DATASET_AUDIO:
    print(f"   Loading from dataset (sample {TEST_SAMPLE_INDEX})...")
    
    from datasets import load_from_disk, Audio as AudioFeature
    import io
    
    # Load dataset without decoding
    dataset = load_from_disk(DATASET_PATH)
    dataset = dataset.cast_column("audio", AudioFeature(decode=False))
    
    # Get sample
    row_dict = dataset._data.slice(TEST_SAMPLE_INDEX, TEST_SAMPLE_INDEX+1).to_pydict()
    
    # Extract audio bytes
    audio_data = row_dict['audio'][0]
    audio_bytes = audio_data['bytes']
    
    # Decode with soundfile
    audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
    
    # Get the original sentence for comparison
    original_sentence = row_dict['sentence'][0]
    filename = os.path.basename(row_dict['path'][0])
    
    print(f"   ✓ Loaded audio from dataset")
    print(f"   Filename: {filename}")
    print(f"   Original text: {original_sentence}")
    
else:
    print(f"   Loading from file: {AUDIO_FILE}...")
    
    if not os.path.exists(AUDIO_FILE):
        print(f"   ❌ ERROR: Audio file not found!")
        exit(1)
    
    # Load audio with librosa
    audio_array, sample_rate = librosa.load(AUDIO_FILE, sr=None)
    original_sentence = None
    filename = AUDIO_FILE
    
    print(f"   ✓ Loaded audio file")

# Convert to mono if stereo
if len(audio_array.shape) > 1:
    audio_array = audio_array.mean(axis=1)

# Resample to 16kHz if needed
if sample_rate != 16000:
    print(f"   Resampling from {sample_rate}Hz to 16000Hz...")
    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
    sample_rate = 16000

print(f"   ✓ Audio prepared")
print(f"   Sample rate: {sample_rate}Hz")
print(f"   Duration: {len(audio_array)/sample_rate:.2f}s")

# ============================================================================
# RUN INFERENCE
# ============================================================================
print(f"\n[3/3] Running inference...")

# Process audio
inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)

# Move to device
input_values = inputs.input_values.to(device)
attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, 'attention_mask') else None

# Run model
with torch.no_grad():
    if attention_mask is not None:
        logits = model(input_values, attention_mask=attention_mask).logits
    else:
        logits = model(input_values).logits

# Get predictions
predicted_ids = torch.argmax(logits, dim=-1)
predicted_phonemes = processor.batch_decode(predicted_ids)[0]

print(f"   ✓ Inference complete!")

# ============================================================================
# RESULTS
# ============================================================================
print(f"\n{'='*70}")
print("RESULTS")
print("="*70)
print(f"\nFile: {filename}")
if original_sentence:
    print(f"Original text: {original_sentence}")
print(f"\nPredicted phonemes:")
print(f"   {predicted_phonemes}")
print(f"\nPhoneme sequence length: {len(predicted_phonemes)} characters")
print("="*70)

# ============================================================================
# OPTIONAL: TEST WITH YOUR TSV PHONEMES
# ============================================================================
if USE_DATASET_AUDIO:
    print(f"\n📊 Comparison with ground truth phonemes:")
    
    # Load TSV to get ground truth
    import pandas as pd
    TSV_PATH = "common_voice_de_190h_phoneme2.tsv"
    
    if os.path.exists(TSV_PATH):
        df = pd.read_csv(TSV_PATH, sep='\t')
        
        # Find matching row
        matching_row = df[df['path'].str.contains(filename.replace('.mp3', ''))]
        
        if len(matching_row) > 0:
            ground_truth = matching_row.iloc[0]['phonemes']
            
            print(f"\nGround truth phonemes (from TSV):")
            print(f"   {ground_truth}")
            
            print(f"\nPredicted phonemes:")
            print(f"   {predicted_phonemes}")
            
            # Simple character-level comparison
            pred_clean = predicted_phonemes.replace(" ", "").replace("|", "")
            gt_clean = ground_truth.replace(" ", "")
            
            print(f"\nLength comparison:")
            print(f"   Ground truth: {len(gt_clean)} phonemes")
            print(f"   Predicted: {len(pred_clean)} phonemes")
            
        else:
            print(f"   ⚠ No matching ground truth found in TSV")
    else:
        print(f"   ⚠ TSV file not found for comparison")

print(f"\n{'='*70}\n")