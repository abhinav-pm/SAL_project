#!/usr/bin/env python3
"""
Speech-to-Phoneme (S2P) Evaluation Script
Evaluates a trained model on the entire test set, calculates the
Phoneme Error Rate (PER), and saves the predictions to a file.

Dependencies:
    pip install torch transformers datasets librosa soundfile pandas tqdm jiwer

Example Usage:
# Evaluate and save results to the default 'evaluation_results.tsv'
python evaluate_model.py --model_path ./s2p_model_polish_phoneme_v1_multi_gpu

# Evaluate and specify a custom output file name
python evaluate_model.py --model_path ./s2p_model_polish_phoneme_v1_multi_gpu --output_file my_model_predictions.tsv
"""

import torch
import librosa
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset, Audio as AudioFeature
import argparse
import os
from tqdm import tqdm
import jiwer

# ============================================================================
# CONFIGURATION
# ============================================================================
TEST_TSV_PATH = "/home/abhinav.pm/ABHI/SAL/v4/phonemized/common_voice_pl_test_phoneme.tsv"
CACHE_PATH = "/scratch/ABHI/huggingface_cache2"


class PhonemePredictor:
    # ... (This class is unchanged) ...
    def __init__(self, model_path):
        if not os.path.isdir(model_path):
            raise NotADirectoryError(f"Model path not found: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Using device: {self.device}")

        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.model.to(self.device)
        print("[*] Model and processor loaded successfully.")

    def predict(self, audio_path):
        try:
            speech_array, _ = librosa.load(audio_path, sr=16000, mono=True)
            input_values = self.processor(
                speech_array, sampling_rate=16000, return_tensors="pt", padding=True
            ).input_values.to(self.device)

            with torch.no_grad():
                logits = self.model(input_values).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            return transcription
        except Exception as e:
            print(f"\n[!] Warning: Could not process file {audio_path}. Error: {e}")
            return ""


def evaluate(model_path, output_file, limit=None): # <-- ADDED output_file argument
    """
    Main evaluation function. Loads data, runs predictions, saves results, and calculates PER.
    """
    predictor = PhonemePredictor(model_path)

    # 1. Load ground truth data
    print(f"\n[1/3] Loading ground truth from: {TEST_TSV_PATH}")
    df_test = pd.read_csv(TEST_TSV_PATH, sep='\t')
    df_test = df_test.dropna(subset=['phonemes'])
    df_test = df_test[df_test['phonemes'].str.strip() != ""]
    
    if limit:
        df_test = df_test.head(limit)
    print(f"  - Found {len(df_test)} samples to evaluate.")

    # 2. Create audio path map
    print("\n[2/3] Building audio file path map...")
    dataset_test = load_dataset("fsicoli/common_voice_22_0", "pl", split="test", cache_dir=CACHE_PATH)
    dataset_test = dataset_test.cast_column("audio", AudioFeature(decode=False))
    audio_path_map = {os.path.basename(item['path']): item['path'] for item in tqdm(dataset_test, desc="  - Mapping paths")}

    # 3. Run predictions
    print(f"\n[3/3] Running predictions on {len(df_test)} samples...")
    ground_truths = []
    predictions = []
    filenames = []  # <-- NEW: List to store filenames for saving

    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="  - Predicting"):
        filename = os.path.basename(row.get('audio_path', row.get('path')))
        audio_path = audio_path_map.get(filename)
        
        if not audio_path:
            continue
            
        true_phonemes = row['phonemes']
        predicted_phonemes = predictor.predict(audio_path)
        
        if predicted_phonemes:
            ground_truths.append(true_phonemes)
            predictions.append(predicted_phonemes)
            filenames.append(filename) # <-- NEW: Save the filename

    # 4. Save results to a file
    if not ground_truths:
        print("\n❌ No predictions were successfully generated. Cannot calculate PER or save results.")
        return

    # <-- START OF NEW CODE BLOCK -->
    print(f"\nSaving prediction results to {output_file}...")
    results_df = pd.DataFrame({
        "filename": filenames,
        "ground_truth": ground_truths,
        "prediction": predictions
    })
    results_df.to_csv(output_file, sep='\t', index=False, encoding='utf-8')
    print(f"  - Successfully saved {len(results_df)} predictions.")
    # <-- END OF NEW CODE BLOCK -->

    # 5. Calculate Phoneme Error Rate (PER)
    print("\nCalculating Phoneme Error Rate (PER)...")
    per = jiwer.wer(ground_truths, predictions)
    
    # 6. Display results
    print("\n" + "="*70)
    print(" " * 25 + "EVALUATION RESULTS")
    print("="*70)
    print(f"  Samples Evaluated:      {len(ground_truths)}")
    print(f"  Phoneme Error Rate (PER): {per:.2%}")
    print("  (Note: Breakdown is unavailable with this version of 'jiwer')")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Speech-to-Phoneme model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory.")
    # <-- NEW ARGUMENT FOR SAVING -->
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="evaluation_results.tsv", 
        help="Path to save the prediction results TSV file."
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of test samples for a quick evaluation.")
    
    args = parser.parse_args()
    
    evaluate(args.model_path, args.output_file, args.limit) # <-- Pass the new argument


if __name__ == "__main__":
    main()