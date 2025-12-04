#!/usr/bin/env python3
"""
train_p2g_danp.py
Universal Training Script for DANP Strategies.
UPDATED: Added --max_samples for debugging.

python3 train_p2g_danp.py --train_file ... --valid_file ... --output_dir ... --max_samples 50
"""

import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
import os
import warnings
import argparse

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    # New argument for debugging
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="For debugging: limit number of rows to load (e.g., 100)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Data
    print(f"Loading train: {args.train_file}")
    df_train = pd.read_csv(args.train_file, sep='\t')
    
    print(f"Loading valid: {args.valid_file}")
    df_valid = pd.read_csv(args.valid_file, sep='\t')

    # --- DEBUGGING LOGIC ---
    if args.max_samples is not None:
        print(f"\n⚠️ DEBUG MODE: Truncating data to first {args.max_samples} samples.")
        df_train = df_train.iloc[:args.max_samples]
        df_valid = df_valid.iloc[:args.max_samples] # Also truncate valid for speed
    # -----------------------

    # Cleaning
    df_train = df_train.dropna(subset=['phonemes', 'sentence'])
    df_valid = df_valid.dropna(subset=['phonemes', 'sentence'])
    
    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", legacy=False)
    
    # 3. Preprocess
    def preprocess_function(examples):
        model_inputs = tokenizer(examples['phonemes'], max_length=256, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['sentence'], max_length=256, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = Dataset.from_pandas(df_train[['phonemes', 'sentence']])
    valid_dataset = Dataset.from_pandas(df_valid[['phonemes', 'sentence']])

    train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=['phonemes', 'sentence'])
    valid_dataset = valid_dataset.map(preprocess_function, batched=True, remove_columns=['phonemes', 'sentence'])

    # 4. Model & Trainer
    model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)

    # Adjust steps if in debug mode (don't wait 500 steps if we only have 100 samples)
    eval_strat_steps = 500
    if args.max_samples is not None and args.max_samples < 1000:
        eval_strat_steps = 10 

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=eval_strat_steps,
        save_steps=eval_strat_steps,
        logging_steps=10,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        warmup_ratio=0.1,
        save_total_limit=2,
        predict_with_generate=True,
        
        # --- CHANGE THIS LINE ---
        fp16=False,  # mT5 is unstable with FP16, use FP32 (False) or BF16 if supported
        # ------------------------
        
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )

    print(f"\nStarting training with {len(train_dataset)} training samples...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()