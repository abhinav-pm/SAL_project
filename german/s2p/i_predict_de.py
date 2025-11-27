#!/usr/bin/env python3
"""
Speech-to-Phoneme (S2P) Inference Script for German
Use a trained model to predict phonemes from an audio file.

Dependencies:
    pip install torch transformers datasets soundfile pandas resampy

Example Usage:
1. Predict a random sample from the test set:
   python3 predict_de.py --model_path ./s2p_model_german_phoneme_split --random_test_sample

2. Predict a specific audio file:
   python3 predict_de.py --model_path ./s2p_model_german_phoneme_split --audio_path /path/to/your/audio.mp3
"""

import torch
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_from_disk, Audio as AudioFeature
import argparse
import os
import random
import soundfile as sf
import io
import resampy
import numpy as np

# ============================================================================
# CONFIGURATION (For random test sample option)
# ============================================================================
# Path to the phonemized TSV file for the test set
TEST_TSV_PATH = "/home/abhinav.pm/ABHI/SAL/v2/german_phoneme_smallset/common_voice_de_test_phoneme.tsv"
# Path to the dataset on disk
DATASET_PATH = "/scratch/ABHI/common_voice_de_190h"


class PhonemePredictor:
    """A class to handle loading the model and running predictions."""

    def __init__(self, model_path):
        """
        Loads the trained model and processor.
        
        Args:
            model_path (str): Path to the directory containing the trained model.
        """
        print(f"[*] Loading model from: {model_path}")
        if not os.path.isdir(model_path):
            raise NotADirectoryError(f"Model path not found: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Using device: {self.device}")

        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.model.to(self.device)
        print("[*] Model and processor loaded successfully.")

    def predict(self, audio_array, sample_rate=16000):
        """
        Predicts the phoneme sequence for a given audio array.

        Args:
            audio_array (np.ndarray): Audio data as numpy array.
            sample_rate (int): Sample rate of the audio.
        
        Returns:
            str: The predicted phoneme sequence.
        """
        # Convert to mono if stereo
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        # Resample if needed
        if sample_rate != 16000:
            audio_array = resampy.resample(audio_array, sample_rate, 16000)
        
        # Ensure float32
        audio_array = audio_array.astype(np.float32)
        
        # Preprocess the audio array
        input_values = self.processor(
            audio_array, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        ).input_values

        # Move tensor to the correct device
        input_values = input_values.to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(input_values).logits

        # Decode the model output
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription


def predict_random_test_sample(predictor):
    """
    Loads the test set, picks a random sample, and runs prediction.
    Compares the prediction with the ground truth phonemes.
    """
    print("\n[+] Running prediction on a random test sample...")

    # 1. Load the phonemized test TSV file
    if not os.path.exists(TEST_TSV_PATH):
        print(f"  ❌ ERROR: Test TSV file not found at {TEST_TSV_PATH}")
        return
    
    df_test = pd.read_csv(TEST_TSV_PATH, sep='\t')
    df_test = df_test.dropna(subset=['phonemes'])
    df_test = df_test[df_test['phonemes'] != ""]
    print(f"  - Loaded {len(df_test)} samples from test TSV.")

    # 2. Pick a random sample
    random_sample = df_test.sample(1).iloc[0]
    filename = os.path.basename(random_sample['path'])
    true_phonemes = random_sample['phonemes']
    
    # 3. Load the dataset from disk
    print(f"  - Randomly selected file: {filename}")
    print(f"  - Loading dataset from: {DATASET_PATH}")
    
    dataset = load_from_disk(DATASET_PATH)
    dataset = dataset.cast_column("audio", AudioFeature(decode=False))
    
    # Find the sample in the dataset
    audio_bytes = None
    sample_rate = None
    
    for item in dataset:
        if os.path.basename(item['path']) == filename:
            audio_bytes = item['audio']['bytes']
            sample_rate = item['audio'].get('sampling_rate', 48000)  # Default if not specified
            break
    
    if audio_bytes is None:
        print(f"  ❌ ERROR: Could not find the audio file for {filename} in the dataset.")
        return

    # 4. Load audio from bytes
    audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
    
    # 5. Predict
    predicted_phonemes = predictor.predict(audio_array, sample_rate)

    # 6. Display results
    print("\n" + "="*70)
    print(" " * 25 + "INFERENCE RESULT")
    print("="*70)
    print(f"  Audio File:   {filename}")
    print(f"  Ground Truth: {true_phonemes}")
    print(f"  Predicted:    {predicted_phonemes}")
    print("="*70)


def predict_from_path(predictor, audio_path):
    """
    Runs prediction on a user-specified audio file path.
    """
    print(f"\n[+] Running prediction on file: {audio_path}")

    if not os.path.exists(audio_path):
        print(f"  ❌ ERROR: Audio file not found at {audio_path}")
        return

    # Load audio file
    audio_array, sample_rate = sf.read(audio_path)
    
    # Predict
    predicted_phonemes = predictor.predict(audio_array, sample_rate)

    # Display results
    print("\n" + "="*70)
    print(" " * 25 + "INFERENCE RESULT")
    print("="*70)
    print(f"  Audio File: {os.path.basename(audio_path)}")
    print(f"  Predicted:  {predicted_phonemes}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Run Speech-to-Phoneme inference for German.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the trained model directory."
    )
    
    # Use a mutually exclusive group to ensure only one mode is chosen
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--random_test_sample", 
        action="store_true", 
        help="Predict on a random sample from the test set."
    )
    group.add_argument(
        "--audio_path", 
        type=str, 
        help="Path to a local audio file for prediction."
    )
    
    args = parser.parse_args()

    # Initialize the predictor
    try:
        predictor = PhonemePredictor(args.model_path)
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        return

    # Run the chosen prediction mode
    if args.random_test_sample:
        predict_random_test_sample(predictor)
    elif args.audio_path:
        predict_from_path(predictor, args.audio_path)


if __name__ == "__main__":
    main()