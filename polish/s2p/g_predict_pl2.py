#!/usr/bin/env python3
"""
Speech-to-Phoneme (S2P) Inference Script - FIXED v2
Use a trained model to predict phonemes from an audio file.

FIX: Downloads audio if needed, then loads from dataset properly.

Dependencies:
    pip install torch transformers datasets librosa soundfile pandas

Example Usage:
1. Predict a random sample from the Common Voice test set:
   python g_predict_pl.py --model_path ./s2p_model_polish_phoneme_v1 --random_test_sample

2. Predict a specific audio file:
   python g_predict_pl.py --model_path ./s2p_model_polish_phoneme_v1 --audio_path /path/to/your/audio.mp3
"""

import torch
import librosa
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
import argparse
import os
import random
import io
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
TEST_TSV_PATH = "/home/abhinav.pm/ABHI/SAL/v4/phonemized/common_voice_pl_test_phoneme.tsv"
CACHE_PATH = "/scratch/ABHI/huggingface_cache2"


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

    def predict_from_path(self, audio_path):
        """
        Predicts the phoneme sequence for a given audio file path.

        Args:
            audio_path (str): Path to the audio file (e.g., .mp3, .wav).
        
        Returns:
            str: The predicted phoneme sequence.
        """
        # Load and resample audio file
        speech_array, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)
        return self._predict_from_array(speech_array)

    def predict_from_array(self, speech_array, sampling_rate=None):
        """
        Predicts the phoneme sequence from audio array.

        Args:
            speech_array (np.ndarray): Audio waveform array.
            sampling_rate (int): Original sampling rate (will resample if not 16000).
        
        Returns:
            str: The predicted phoneme sequence.
        """
        # Resample if needed
        if sampling_rate is not None and sampling_rate != 16000:
            speech_array = librosa.resample(
                speech_array.astype(np.float32), 
                orig_sr=sampling_rate, 
                target_sr=16000
            )
        
        return self._predict_from_array(speech_array)

    def _predict_from_array(self, speech_array):
        """
        Internal method to predict from numpy array at 16kHz.
        
        Args:
            speech_array (np.ndarray): Audio waveform array at 16kHz.
        
        Returns:
            str: The predicted phoneme sequence.
        """
        # Ensure float32
        speech_array = speech_array.astype(np.float32)
        
        # Preprocess the audio array
        input_values = self.processor(
            speech_array, 
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
    df_test = df_test[df_test['phonemes'].str.strip() != ""]
    print(f"  - Loaded {len(df_test)} samples from test TSV.")

    # 2. Pick a random sample
    random_sample = df_test.sample(1).iloc[0]
    filename = os.path.basename(random_sample.get('audio_path', random_sample.get('path')))
    true_phonemes = random_sample['phonemes']
    
    print(f"  - Randomly selected file: {filename}")
    
    # 3. Load the dataset WITH audio (this will download if needed)
    print("  - Loading dataset (may download audio if not cached)...")
    dataset_test = load_dataset(
        "fsicoli/common_voice_22_0", 
        "pl", 
        split="test", 
        cache_dir=CACHE_PATH,
        trust_remote_code=True
    )
    print(f"  - Dataset loaded: {len(dataset_test)} samples")

    # 4. Find the matching sample
    print("  - Searching for audio sample...")
    found = False
    for item in dataset_test:
        if os.path.basename(item['path']) == filename:
            print("  - Sample found! Loading audio...")
            
            # Get audio data - HuggingFace datasets automatically loads it
            audio_data = item['audio']
            
            if audio_data is None:
                print("  ❌ ERROR: Audio data is None")
                return
            
            # Extract the audio array and sampling rate
            speech_array = audio_data['array']
            sampling_rate = audio_data['sampling_rate']
            
            print(f"  - Audio loaded: {len(speech_array)} samples at {sampling_rate}Hz")
            
            # 5. Predict
            print("  - Running prediction...")
            predicted_phonemes = predictor.predict_from_array(speech_array, sampling_rate)

            # 6. Display results
            print("\n" + "="*70)
            print(" " * 25 + "INFERENCE RESULT")
            print("="*70)
            print(f"  Audio File:   {filename}")
            print(f"  Ground Truth: {true_phonemes}")
            print(f"  Predicted:    {predicted_phonemes}")
            print("="*70)
            
            found = True
            break
    
    if not found:
        print(f"  ❌ ERROR: Could not find audio file {filename} in the dataset")


def predict_from_path(predictor, audio_path):
    """
    Runs prediction on a user-specified audio file path.
    """
    print(f"\n[+] Running prediction on file: {audio_path}")

    if not os.path.exists(audio_path):
        print(f"  ❌ ERROR: Audio file not found at {audio_path}")
        return

    # Predict
    print("  - Loading audio...")
    predicted_phonemes = predictor.predict_from_path(audio_path)

    # Display results
    print("\n" + "="*70)
    print(" " * 25 + "INFERENCE RESULT")
    print("="*70)
    print(f"  Audio File: {os.path.basename(audio_path)}")
    print(f"  Predicted:  {predicted_phonemes}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Run Speech-to-Phoneme inference.")
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
        help="Predict on a random sample from the Common Voice test set."
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