#!/usr/bin/env python3
"""
Evaluate a trained Phoneme-to-Grapheme (P2G) model.
- Loads a fine-tuned model from a checkpoint.
- Runs inference on the test set.
- Calculates and reports Word Error Rate (WER) and Character Error Rate (CER).
- Saves predictions to a CSV file for analysis.

Usage:
    python evaluate_p2g.py \
        --model_path ./p2g_model_polish_v2 \
        --test_tsv_path /path/to/phonemized/common_voice_pl_test_phoneme.tsv \
        --output_file evaluation_results.csv
"""

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import argparse
import os
import jiwer

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description="Evaluate a P2G model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./p2g_model_polish_v2",
        help="Path to the fine-tuned model directory"
    )
    parser.add_argument(
        "--test_tsv_path",
        type=str,
        default="/home/abhinav.pm/ABHI/SAL/v4/phonemized/common_voice_pl_test_phoneme.tsv",
        help="Path to the phonemized test TSV file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="p2g_evaluation_results.csv",
        help="Path to save the CSV file with predictions"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for quick testing)"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams for beam search generation"
    )
    
    args = parser.parse_args()

    print("="*70)
    print(" "*20 + "P2G Model Evaluation")
    print("="*70)
    print(f"📋 Configuration:")
    print(f"   Model Path: {args.model_path}")
    print(f"   Test Data: {args.test_tsv_path}")
    print(f"   Output File: {args.output_file}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Num Beams: {args.num_beams}")

    # ============================================================================
    # STEP 1: LOAD MODEL AND TOKENIZER
    # ============================================================================
    print("\n[1/5] Loading model and tokenizer...")

    if not os.path.exists(args.model_path):
        print(f"❌ ERROR: Model path not found: {args.model_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    model.to(device)
    model.eval()  # Set model to evaluation mode

    print(f"   ✓ Model loaded on device: {device}")

    # ============================================================================
    # STEP 2: LOAD AND PREPARE TEST DATA
    # ============================================================================
    print("\n[2/5] Loading and preparing test data...")

    if not os.path.exists(args.test_tsv_path):
        print(f"❌ ERROR: Test TSV file not found: {args.test_tsv_path}")
        return

    df_test = pd.read_csv(args.test_tsv_path, sep='\t')
    df_test = df_test.dropna(subset=['phonemes', 'sentence'])
    df_test = df_test[(df_test['phonemes'] != "") & (df_test['sentence'] != "")]

    if args.max_samples:
        print(f"   Using a subset of {args.max_samples} samples for evaluation.")
        df_test = df_test.head(args.max_samples)

    print(f"   ✓ Loaded {len(df_test)} test samples.")

    phonemes = df_test['phonemes'].tolist()
    ground_truths = df_test['sentence'].tolist()
    
    # ============================================================================
    # STEP 3: RUN INFERENCE
    # ============================================================================
    print("\n[3/5] Running inference on the test set...")
    
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(phonemes), args.batch_size), desc="Evaluating"):
            batch_phonemes = phonemes[i:i+args.batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_phonemes, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=256
            ).to(device)
            
            # Generate predictions
            generated_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=256,
                num_beams=args.num_beams,
                early_stopping=True
            )
            
            # Decode and store
            batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(batch_preds)

    print("   ✓ Inference complete.")

    # ============================================================================
    # STEP 4: CALCULATE METRICS
    # ============================================================================
    print("\n[4/5] Calculating evaluation metrics...")

    wer = jiwer.wer(ground_truths, predictions)
    cer = jiwer.cer(ground_truths, predictions)

    print("\n" + "="*70)
    print("Evaluation Metrics")
    print("="*70)
    print(f"  Word Error Rate (WER): {wer:.4f} ({wer*100:.2f}%)")
    print(f"  Character Error Rate (CER): {cer:.4f} ({cer*100:.2f}%)")
    print("="*70)
    
    # ============================================================================
    # STEP 5: SAVE RESULTS AND SHOW EXAMPLES
    # ============================================================================
    print("\n[5/5] Saving results and showing examples...")

    results_df = pd.DataFrame({
        'phonemes': phonemes,
        'ground_truth': ground_truths,
        'prediction': predictions
    })
    results_df.to_csv(args.output_file, index=False)
    print(f"\n✓ Full evaluation results saved to: {args.output_file}")

    print("\n" + "="*70)
    print("Prediction Examples (Top 5)")
    print("="*70)
    for i, row in results_df.head(5).iterrows():
        print(f"\n[{i+1}] PHONEMES:  {row['phonemes']}")
        print(f"    TRUTH:     {row['ground_truth']}")
        print(f"    PREDICTED: {row['prediction']}")
        if row['ground_truth'].lower() == row['prediction'].lower():
            print("    -> ✓ Correct")
        else:
            print("    -> ❌ Incorrect")
    print("="*70)

if __name__ == "__main__":
    main()