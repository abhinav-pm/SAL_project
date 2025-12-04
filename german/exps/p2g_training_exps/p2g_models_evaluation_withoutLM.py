#!/usr/bin/env python3
import torch
import pandas as pd
import json
import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evaluate import load

"""
python evaluate_p2g.py \
  --test_file ./test_5h_phoneme.tsv \
  --model_path /scratch/kallind/p2g_model_polish_tkm_v1

"""

# Force CPU if GPU is not needed, or use CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Using device: {device}")

def load_data(tsv_path):
    print(f"[*] Loading test data from {tsv_path}...")
    try:
        df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
    except:
        df = pd.read_csv(tsv_path, sep='\t', error_bad_lines=False)
    
    # Filter valid rows
    if 'hypotheses_json' in df.columns:
        df = df.dropna(subset=['hypotheses_json', 'sentence'])
    else:
        df = df.dropna(subset=['sentence'])
        
    print(f"[*] Loaded {len(df)} samples.")
    return df

def evaluate(args):
    # 1. Load Model & Tokenizer
    print(f"[*] Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)
    model.eval()

    # 2. Load Metrics
    wer_metric = load("wer")
    cer_metric = load("cer")

    # 3. Load Data
    df = load_data(args.test_file)
    
    # 4. Inference Loop
    predictions = []
    references = []
    inputs_used = [] # Log what phonemes we actually used
    
    print("[*] Starting Inference...")
    batch_size = args.batch_size
    
    # We loop in batches for speed
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i+batch_size]
        batch_inputs = []
        batch_refs = batch_df['sentence'].tolist()
        
        # Prepare Inputs (Extract Top-1 Hypothesis)
        for _, row in batch_df.iterrows():
            if 'hypotheses_json' in row and pd.notna(row['hypotheses_json']):
                # REAL WORLD SCENARIO: Use the best guess from S2P
                try:
                    hyps = json.loads(row['hypotheses_json'])
                    # Assuming list is sorted by score descending, or we pick max score
                    # Your generation script usually puts the best beam first (index 0)
                    top_hyp = hyps[0]['phonemes']
                except:
                    top_hyp = "" # Fallback
            elif 'phonemes' in row:
                # FALLBACK: Use ground truth or provided phonemes column
                top_hyp = row['phonemes']
            elif 'ground_truth_phonemes' in row:
                # ORACLE SCENARIO: Use perfect phonemes (Upper bound check)
                top_hyp = row['ground_truth_phonemes']
            else:
                top_hyp = ""
            
            batch_inputs.append(top_hyp)
        
        # Tokenize
        inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                inputs["input_ids"],
                max_length=256,
                num_beams=4, # Use beam search for better P2G quality
                early_stopping=True
            )
        
        # Decode
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        predictions.extend(decoded_preds)
        references.extend(batch_refs)
        inputs_used.extend(batch_inputs)

    # 5. Calculate Metrics
    print("[*] Calculating Metrics...")
    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)
    
    print("\n" + "="*40)
    print(f"RESULTS for {args.model_path}")
    print("="*40)
    print(f"WER: {wer:.4f} ({wer*100:.2f}%)")
    print(f"CER: {cer:.4f} ({cer*100:.2f}%)")
    print("="*40)

    # 6. Save Predictions to CSV for analysis
    output_df = pd.DataFrame({
        'input_phonemes': inputs_used,
        'reference_text': references,
        'generated_text': predictions
    })
    
    save_path = os.path.join(args.model_path, "eval_results.csv")
    output_df.to_csv(save_path, index=False)
    print(f"[*] Detailed predictions saved to: {save_path}")

    # 7. Print some examples
    print("\n[*] Examples:")
    for i in range(5):
        print(f"\nSample {i+1}:")
        print(f"  Input (Phonemes): {inputs_used[i]}")
        print(f"  Ref   (Text):     {references[i]}")
        print(f"  Pred  (Text):     {predictions[i]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True, help="Path to test TSV file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model directory")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    evaluate(args)