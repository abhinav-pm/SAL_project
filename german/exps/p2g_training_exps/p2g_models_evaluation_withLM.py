#!/usr/bin/env python3
import torch
import pandas as pd
import json
import argparse
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from evaluate import load

'''
python3 p2g_models_evaluation_withLM.py --test_file /home2/dasari.priyanka/ABHI/SAL/polish/exps/phonimized/test_5h_phoneme.tsv --p2g_model_path /home2/dasari.priyanka/ABHI/SAL/polish/p2g_beam1_model_polish
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(tsv_path):
    # (Same loading logic as before)
    try:
        df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
    except:
        df = pd.read_csv(tsv_path, sep='\t', error_bad_lines=False)
    
    if 'hypotheses_json' in df.columns:
        df = df.dropna(subset=['hypotheses_json', 'sentence'])
    else:
        df = df.dropna(subset=['sentence'])
    return df

def calculate_lm_perplexity(text, lm_model, lm_tokenizer):
    """Calculates how 'surprised' the LM is by the text. Lower is better."""
    inputs = lm_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = lm_model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item() # This is the CrossEntropyLoss (NLL)

def evaluate_with_rescoring(args):
    # 1. Load P2G Model (Your trained model)
    print(f"[*] Loading P2G Model: {args.p2g_model_path}...")
    p2g_tokenizer = AutoTokenizer.from_pretrained(args.p2g_model_path)
    p2g_model = AutoModelForSeq2SeqLM.from_pretrained(args.p2g_model_path).to(device)
    p2g_model.eval()

    # 2. Load External LM (Polish GPT-2)
    # We use a small Polish GPT-2 model to score the grammar
    # lm_name = "sdadas/polish-gpt2-small" 
    lm_name = "dbmdz/german-gpt2" 
    print(f"[*] Loading LM Model: {lm_name}...")
    lm_tokenizer = AutoTokenizer.from_pretrained(lm_name)
    lm_model = AutoModelForCausalLM.from_pretrained(lm_name).to(device)
    lm_model.eval()

    # Metrics
    wer_metric = load("wer")

    # Data
    df = load_data(args.test_file)
    print(f"[*] Starting Evaluation on {len(df)} samples...")

    predictions_no_lm = []
    predictions_with_lm = []
    references = []
    
    # LM Weight (Alpha): How much to trust the LM vs the P2G model
    # 0.0 = Pure P2G (Without LM)
    # 0.5 = Balanced
    # 1.0 = Trust Grammar only
    LM_ALPHA = 0.3 

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # --- 1. Get Input Phonemes ---
        if 'hypotheses_json' in row and pd.notna(row['hypotheses_json']):
            try:
                hyps = json.loads(row['hypotheses_json'])
                input_phonemes = hyps[0]['phonemes'] # Top-1 S2P hypothesis
            except:
                input_phonemes = ""
        elif 'phonemes' in row:
            input_phonemes = row['phonemes']
        else:
            continue

        if not input_phonemes: continue

        # --- 2. Generate N-Best List (Beams) ---
        inputs = p2g_tokenizer(input_phonemes, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            # Return Top-5 candidates per sentence
            outputs = p2g_model.generate(
                inputs["input_ids"],
                max_length=128,
                num_beams=5,
                num_return_sequences=5, 
                return_dict_in_generate=True,
                output_scores=True
            )

        # Decode the 5 candidates
        candidates = p2g_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        
        # Get P2G sequence scores (log probabilities)
        # Note: This is an approximation of the sequence score
        p2g_scores = outputs.sequences_scores.cpu().numpy()

        # --- 3. Score with LM ---
        final_scores = []
        
        for i, candidate in enumerate(candidates):
            if not candidate.strip(): 
                final_scores.append(-9999) # Penalize empty
                continue
                
            # Calculate LM Loss (Lower is better, so we negate it to make Higher=Better)
            lm_loss = calculate_lm_perplexity(candidate, lm_model, lm_tokenizer)
            lm_score = -lm_loss 
            
            # COMBINE SCORES
            # Total = P2G_LogProb + (Alpha * LM_LogProb)
            total_score = p2g_scores[i] + (LM_ALPHA * lm_score)
            final_scores.append(total_score)

        # --- 4. Select Winners ---
        
        # Winner "Without LM" is just the first beam (highest P2G score)
        best_no_lm = candidates[0]
        
        # Winner "With LM" is the one with highest total_score
        best_with_lm_idx = np.argmax(final_scores)
        best_with_lm = candidates[best_with_lm_idx]

        predictions_no_lm.append(best_no_lm)
        predictions_with_lm.append(best_with_lm)
        references.append(row['sentence'])

    # --- 5. Final Results ---
    wer_no_lm = wer_metric.compute(predictions=predictions_no_lm, references=references)
    wer_with_lm = wer_metric.compute(predictions=predictions_with_lm, references=references)

    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)
    print(f"WITHOUT LM (Pure P2G):  WER = {wer_no_lm:.4f} ({wer_no_lm*100:.2f}%)")
    print(f"WITH LM (Rescoring):    WER = {wer_with_lm:.4f} ({wer_with_lm*100:.2f}%)")
    print("="*50)
    
    # Save comparison
    out_df = pd.DataFrame({
        'reference': references,
        'pred_no_lm': predictions_no_lm,
        'pred_with_lm': predictions_with_lm
    })
    out_df.to_csv(os.path.join(args.p2g_model_path, "results_lm_comparison.csv"), index=False)
    print(f"Detailed CSV saved to {args.p2g_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--p2g_model_path", type=str, required=True)
    args = parser.parse_args()
    
    evaluate_with_rescoring(args)