#!/usr/bin/env python3
"""
Evaluate German Phoneme-to-Grapheme (P2G) Model
- Runs inference on German test set
- Calculates WER and CER
- Saves predictions to CSV

Usage:
    python evaluate_p2g_german.py \
        --model_path ./p2g_model_german_v1 \
        --test_tsv_path /path/to/common_voice_de_test_phoneme_cleaned.tsv \
        --output_file german_p2g_results.csv
python3 evaluate_p2g.py --model_path ./p2g_model_german_v1 --test_tsv_path /home/abhinav.pm/ABHI/SAL/v2/german_phoneme_smallset/common_voice_de_test_phoneme_cleaned.tsv --output_file german_p2g_results.csv
"""

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import argparse
import os
import jiwer
import warnings

warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description="Evaluate German P2G model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./p2g_model_german_v1",
        help="Path to the fine-tuned German model"
    )
    parser.add_argument(
        "--test_tsv_path",
        type=str,
        default="/home/abhinav.pm/ABHI/SAL/v2/german_phoneme_smallset/common_voice_de_test_phoneme_cleaned.tsv",
        help="Path to German test TSV file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="german_p2g_results.csv",
        help="Output CSV file"
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
        help="Max samples to evaluate (for testing)"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Beam size"
    )
    
    args = parser.parse_args()

    print("="*70)
    print(" "*18 + "German P2G Evaluation")
    print("="*70)
    print(f"📋 Configuration:")
    print(f"   Model: {args.model_path}")
    print(f"   Test Data: {args.test_tsv_path}")
    print(f"   Output: {args.output_file}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Num Beams: {args.num_beams}")

    # ========================================================================
    # STEP 1: LOAD MODEL
    # ========================================================================
    print("\n[1/5] Loading model and tokenizer...")

    if not os.path.exists(args.model_path):
        print(f"❌ ERROR: Model not found: {args.model_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    print(f"   ✓ Model loaded on: {device}")

    # ========================================================================
    # STEP 2: LOAD TEST DATA
    # ========================================================================
    print("\n[2/5] Loading German test data...")

    if not os.path.exists(args.test_tsv_path):
        print(f"❌ ERROR: Test file not found: {args.test_tsv_path}")
        return

    df_test = pd.read_csv(args.test_tsv_path, sep='\t')
    df_test = df_test.dropna(subset=['phonemes', 'sentence'])
    df_test = df_test[(df_test['phonemes'] != "") & (df_test['sentence'] != "")]

    if args.max_samples:
        print(f"   Using subset: {args.max_samples} samples")
        df_test = df_test.head(args.max_samples)

    print(f"   ✓ Loaded {len(df_test)} test samples")

    phonemes = df_test['phonemes'].tolist()
    ground_truths = df_test['sentence'].tolist()
    
    # ========================================================================
    # STEP 3: RUN INFERENCE
    # ========================================================================
    print("\n[3/5] Running inference...")
    
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(0, len(phonemes), args.batch_size), desc="Evaluating"):
            batch_phonemes = phonemes[i:i+args.batch_size]
            
            inputs = tokenizer(
                batch_phonemes,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)
            
            generated_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=256,
                num_beams=args.num_beams,
                early_stopping=True
            )
            
            batch_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            predictions.extend(batch_preds)

    print("   ✓ Inference complete")

    # ========================================================================
    # STEP 4: CALCULATE METRICS
    # ========================================================================
    print("\n[4/5] Calculating metrics...")

    wer = jiwer.wer(ground_truths, predictions)
    cer = jiwer.cer(ground_truths, predictions)

    print("\n" + "="*70)
    print("German P2G Evaluation Metrics")
    print("="*70)
    print(f"  Word Error Rate (WER): {wer:.4f} ({wer*100:.2f}%)")
    print(f"  Character Error Rate (CER): {cer:.4f} ({cer*100:.2f}%)")
    print("="*70)
    
    # ========================================================================
    # STEP 5: SAVE RESULTS
    # ========================================================================
    print("\n[5/5] Saving results...")

    results_df = pd.DataFrame({
        'phonemes': phonemes,
        'ground_truth': ground_truths,
        'prediction': predictions
    })
    results_df.to_csv(args.output_file, index=False)
    print(f"\n✓ Results saved to: {args.output_file}")

    print("\n" + "="*70)
    print("Prediction Examples (Top 5)")
    print("="*70)
    for i, row in results_df.head(5).iterrows():
        print(f"\n[{i+1}] PHONEMES:  {row['phonemes'][:70]}...")
        print(f"    TRUTH:     {row['ground_truth']}")
        print(f"    PREDICTED: {row['prediction']}")
        if row['ground_truth'].lower() == row['prediction'].lower():
            print("    -> ✓ Correct")
        else:
            print("    -> ❌ Incorrect")
    print("="*70)

    # Paper comparison
    print("\n" + "="*70)
    print("Paper Comparison (German, 20h)")
    print("="*70)
    print(f"  Paper (Basic P2G):      ~30.7% WER")
    print(f"  Paper (P2G + DANP):     ~29.97% WER")
    print(f"  Paper (P2G + TKM):      ~28.78% WER")
    print(f"  Your model:             {wer*100:.2f}% WER")
    
    if wer < 0.32:
        print(f"\n  ✅ Great! Your model is performing well!")
    else:
        print(f"\n  ℹ️  Consider: more epochs, DANP, or larger model")
    print("="*70)

if __name__ == "__main__":
    main()