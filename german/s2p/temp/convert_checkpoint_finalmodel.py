#!/usr/bin/env python3
"""
Convert a checkpoint to a final model
"""
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Paths
CHECKPOINT_PATH = "./s2p_model_german_phoneme_split/checkpoint-2500"  # Latest checkpoint
OUTPUT_DIR = "./s2p_model_german_phoneme_split"

print(f"Loading from checkpoint: {CHECKPOINT_PATH}")

# Load processor (already saved in main directory)
processor = Wav2Vec2Processor.from_pretrained(OUTPUT_DIR)
print("✓ Processor loaded")

# Load model from checkpoint
model = Wav2Vec2ForCTC.from_pretrained(CHECKPOINT_PATH)
print("✓ Model loaded from checkpoint")

# Save to main directory
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"✓ Final model saved to: {OUTPUT_DIR}")

print("\n" + "="*60)
print("✅ Conversion complete!")
print("="*60)
print(f"You can now use: python3 predict_de.py --model_path {OUTPUT_DIR} --random_test_sample")