#!/usr/bin/env python3
"""
Speech-to-Phoneme (S2P) Evaluation Script for German
Evaluates a trained model on the entire test set, calculates the
Phoneme Error Rate (PER), and saves the predictions to a file.

Dependencies:
    pip install torch transformers datasets soundfile pandas tqdm jiwer resampy

Example Usage:
# Evaluate and save results to the default 'evaluation_results.tsv'
python evaluate_de.py --model_path ./s2p_model_german_phoneme_split

# Evaluate and specify a custom output file name
python evaluate_de.py --model_path ./s2p_model_german_phoneme_split --output_file german_predictions.tsv

# Quick test with limited samples
python evaluate_de.py --model_path ./s2p_model_german_phoneme_split --limit 100
"""

import torch
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_from_disk, Audio as AudioFeature
import argparse
import os
from tqdm import tqdm
import jiwer
import soundfile as sf
import io
import resampy
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
TEST_TSV_PATH = "/home/abhinav.pm/ABHI/SAL/v2/german_phoneme_smallset/common_voice_de_test_phoneme.tsv"
DATASET_PATH = "/scratch/ABHI/common_voice_de_190h"


class PhonemePredictor:
    """A class to handle loading the model and running predictions."""
    
    def __init__(self, model_path):
        if not os.path.isdir(model_path):
            raise NotADirectoryError(f"Model path not found: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Using device: {self.device}")

        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
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
        try:
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Resample if needed
            if sample_rate != 16000:
                audio_array = resampy.resample(audio_array, sample_rate, 16000)
            
            # Ensure float32
            audio_array = audio_array.astype(np.float32)
            
            # Preprocess
            input_values = self.processor(
                audio_array, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            ).input_values.to(self.device)

            # Inference
            with torch.no_grad():
                logits = self.model(input_values).logits
            
            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            return transcription
            
        except Exception as e:
            print(f"\n[!] Warning: Could not process audio. Error: {str(e)[:100]}")
            return ""


def evaluate(model_path, output_file, limit=None):
    """
    Main evaluation function. Loads data, runs predictions, saves results, and calculates PER.
    """
    predictor = PhonemePredictor(model_path)

    # 1. Load ground truth data
    print(f"\n[1/4] Loading ground truth from: {TEST_TSV_PATH}")
    if not os.path.exists(TEST_TSV_PATH):
        print(f"  ❌ ERROR: Test TSV file not found at {TEST_TSV_PATH}")
        return
    
    df_test = pd.read_csv(TEST_TSV_PATH, sep='\t')
    df_test = df_test.dropna(subset=['phonemes'])
    df_test = df_test[df_test['phonemes'].str.strip() != ""]
    
    if limit:
        df_test = df_test.head(limit)
    print(f"  - Found {len(df_test)} samples to evaluate.")

    # 2. Load the dataset from disk
    print(f"\n[2/4] Loading dataset from: {DATASET_PATH}")
    dataset = load_from_disk(DATASET_PATH)
    dataset = dataset.cast_column("audio", AudioFeature(decode=False))
    print(f"  - Loaded {len(dataset)} samples from disk")

    # 3. Create audio path map
    print("\n[3/4] Building filename to dataset index map...")
    filename_to_idx = {}
    for idx in tqdm(range(len(dataset)), desc="  - Mapping", ncols=70):
        filename = os.path.basename(dataset[idx]['path'])
        filename_to_idx[filename] = idx
    print(f"  - Created map with {len(filename_to_idx)} entries")

    # 4. Run predictions
    print(f"\n[4/4] Running predictions on {len(df_test)} samples...")
    ground_truths = []
    predictions = []
    filenames = []
    
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="  - Predicting", ncols=70):
        filename = os.path.basename(row['path'])
        
        # Check if filename exists in dataset
        if filename not in filename_to_idx:
            print(f"\n  ⚠ Warning: {filename} not found in dataset, skipping...")
            continue
        
        dataset_idx = filename_to_idx[filename]
        
        try:
            # Get audio bytes from dataset
            audio_bytes = dataset[dataset_idx]['audio']['bytes']
            
            # Load audio
            audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
            
            # Get ground truth
            true_phonemes = row['phonemes']
            
            # Predict
            predicted_phonemes = predictor.predict(audio_array, sample_rate)
            
            if predicted_phonemes:  # Only add if prediction succeeded
                ground_truths.append(true_phonemes)
                predictions.append(predicted_phonemes)
                filenames.append(filename)
                
        except Exception as e:
            print(f"\n  ⚠ Error processing {filename}: {str(e)[:100]}")
            continue

    # 5. Save results to a file
    if not ground_truths:
        print("\n❌ No predictions were successfully generated. Cannot calculate PER or save results.")
        return

    print(f"\n[5/5] Saving prediction results to {output_file}...")
    results_df = pd.DataFrame({
        "filename": filenames,
        "ground_truth": ground_truths,
        "prediction": predictions
    })
    results_df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
    print(f"  - Successfully saved {len(results_df)} predictions.")

    # 6. Calculate Phoneme Error Rate (PER)
    print("\n[6/6] Calculating Phoneme Error Rate (PER)...")
    per = jiwer.wer(ground_truths, predictions)
    
    # 7. Display results
    print("\n" + "="*70)
    print(" " * 25 + "EVALUATION RESULTS")
    print("="*70)
    print(f"  Samples Evaluated:        {len(ground_truths)}")
    print(f"  Phoneme Error Rate (PER): {per:.2%}")
    print(f"  Results saved to:         {output_file}")
    print("="*70)
    
    # Show a few examples
    print("\n📊 Sample Predictions (first 5):")
    print("-" * 70)
    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        print(f"\n{i+1}. File: {row['filename']}")
        print(f"   Truth: {row['ground_truth'][:80]}")
        print(f"   Pred:  {row['prediction'][:80]}")
    print("-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Speech-to-Phoneme model for German.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the trained model directory."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="evaluation_results_german.tsv", 
        help="Path to save the prediction results TSV file."
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=None, 
        help="Limit the number of test samples for a quick evaluation."
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print(" " * 20 + "German S2P Model Evaluation")
    print("="*70)
    
    evaluate(args.model_path, args.output_file, args.limit)


if __name__ == "__main__":
    main()