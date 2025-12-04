#!/usr/bin/env python3
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
import os
import warnings
import json

warnings.filterwarnings('ignore')
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("="*70)
print(" "*15 + "Phoneme-to-Grapheme (P2G) Training - TKM MODE")
print(" "*18 + "(Dynamic Sampling Enabled)")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
# UPDATE THESE PATHS
TRAIN_TSV_PATH = "./p2g_training_polish_20h/train_tkm_k32.tsv" 
VALID_TSV_PATH = "./dev_5h_phoneme.tsv" # <--- NEW VALIDATION PATH
OUTPUT_DIR = "/scratch/kallind/p2g_model_polish_tkm_v1"
MODEL_NAME = "google/mt5-small"

# Training parameters
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 256
WARMUP_RATIO = 0.1
SAVE_STEPS = 500
LOGGING_STEPS = 50

# ============================================================================
# STEP 1: DEFINE TKM DATASET CLASS
# ============================================================================
class TKMDataset(Dataset):
    def __init__(self, tsv_path, tokenizer, max_source_len, max_target_len):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        
        print(f"[*] Loading TKM data from {tsv_path}...")
        try:
            self.data = pd.read_csv(tsv_path, sep='\t', on_bad_lines='skip')
        except:
            self.data = pd.read_csv(tsv_path, sep='\t', error_bad_lines=False)
            
        # Check if hypotheses_json exists, otherwise strict filtering might empty the dataset
        if 'hypotheses_json' in self.data.columns:
            self.data = self.data.dropna(subset=['hypotheses_json', 'sentence'])
        else:
            print(f"[!] Warning: 'hypotheses_json' not found in {os.path.basename(tsv_path)}. Using ground truth only.")
            self.data = self.data.dropna(subset=['sentence'])
            
        print(f"[*] Loaded {len(self.data)} valid samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        target_text = row['sentence']
        
        # --- DYNAMIC SAMPLING LOGIC ---
        try:
            # Only try to parse if the column exists and isn't null
            if 'hypotheses_json' in row and pd.notna(row['hypotheses_json']):
                hypotheses = json.loads(row['hypotheses_json'])
                
                # Extract phonemes and scores
                candidates = [h['phonemes'] for h in hypotheses]
                scores = np.array([h['score'] for h in hypotheses], dtype=np.float32)
                
                # Normalize scores to probabilities (Sum to 1)
                if scores.sum() > 0:
                    probs = scores / scores.sum()
                else:
                    probs = np.ones_like(scores) / len(scores)
                
                # Weighted Random Sample
                selected_phonemes = np.random.choice(candidates, p=probs)
            else:
                raise ValueError("No hypotheses data")
            
        except Exception:
            # Fallback to ground truth if TKM data is missing or broken
            # This handles standard Validation files that might not have the TKM columns
            if 'ground_truth_phonemes' in row:
                selected_phonemes = row['ground_truth_phonemes']
            elif 'phonemes' in row:
                selected_phonemes = row['phonemes'] # Fallback for standard TSVs
            else:
                selected_phonemes = "" # Should be caught by dropna, but safe default

        # --- TOKENIZATION ---
        model_inputs = self.tokenizer(
            selected_phonemes,
            max_length=self.max_source_len,
            truncation=True,
            padding=False 
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text,
                max_length=self.max_target_len,
                truncation=True,
                padding=False
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

# ============================================================================
# STEP 2: LOAD TOKENIZER AND MODEL
# ============================================================================
print(f"\n[2/5] Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ============================================================================
# STEP 3: PREPARE DATASETS (UPDATED)
# ============================================================================
print(f"\n[3/5] Preparing datasets...")

# 1. Load Training Data
print("--- Loading Training Set ---")
train_dataset = TKMDataset(TRAIN_TSV_PATH, tokenizer, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH)

# 2. Load Validation Data
print("--- Loading Validation Set ---")
valid_dataset = TKMDataset(VALID_TSV_PATH, tokenizer, MAX_SOURCE_LENGTH, MAX_TARGET_LENGTH)

print(f" \u2713 Train: {len(train_dataset)} samples")
print(f" \u2713 Valid: {len(valid_dataset)} samples")

# ============================================================================
# STEP 4: SETUP TRAINING
# ============================================================================
print(f"\n[4/5] Setting up training...")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100
)

# Calculate steps
steps_per_epoch = len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION)
total_steps = steps_per_epoch * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    warmup_steps=warmup_steps,
    lr_scheduler_type="linear",
    save_total_limit=2,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    fp16=False,
    dataloader_num_workers=0, 
    logging_dir=f"{OUTPUT_DIR}/logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# ============================================================================
# STEP 5: START TRAINING
# ============================================================================
print(f"\nStarting TKM training...")
trainer.train()

trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to: {OUTPUT_DIR}")