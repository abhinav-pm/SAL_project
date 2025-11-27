#!/usr/bin/env python3
"""
German Phoneme-to-Grapheme (P2G) Training Script - FIXED
- Fine-tunes mT5 on German phoneme-to-text data
- FIXES: Proper warmup, disabled fp16 for debugging, no multiprocessing
"""

import torch
import pandas as pd
from datasets import Dataset
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
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Fix fork warning

print("="*70)
print(" "*12 + "German Phoneme-to-Grapheme (P2G) Training (FIXED)")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
TSV_DIR = "/home/abhinav.pm/ABHI/SAL/v2/german_phoneme_smallset"
OUTPUT_DIR = "./p2g_model_german_v1"
MODEL_NAME = "google/mt5-small"
USE_20H_SUBSET = True
USE_CLEANED = True  # Use cleaned TSV files

# Training parameters
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4
NUM_EPOCHS = 10
LEARNING_RATE = 3e-4
MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 256
WARMUP_RATIO = 0.1  # 10% of total steps
SAVE_STEPS = 500
LOGGING_STEPS = 50

print(f"\n📋 Configuration:")
print(f"   TSV directory: {TSV_DIR}")
print(f"   Output dir: {OUTPUT_DIR}")
print(f"   Model: {MODEL_NAME}")
print(f"   Use 20h subset: {USE_20H_SUBSET}")
print(f"   Use cleaned data: {USE_CLEANED}")

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print(f"\n[1/5] Loading German phonemized TSV files...")

# Select file based on configuration
if USE_20H_SUBSET:
    if USE_CLEANED:
        train_tsv = os.path.join(TSV_DIR, "common_voice_de_train_20h_phoneme_cleaned.tsv")
    else:
        train_tsv = os.path.join(TSV_DIR, "common_voice_de_train_20h_phoneme.tsv")
    print(f"   Using 20-hour German subset")
else:
    if USE_CLEANED:
        train_tsv = os.path.join(TSV_DIR, "common_voice_de_train_phoneme_cleaned.tsv")
    else:
        train_tsv = os.path.join(TSV_DIR, "common_voice_de_train_phoneme.tsv")
    print(f"   Using full German training data")

if USE_CLEANED:
    valid_tsv = os.path.join(TSV_DIR, "common_voice_de_validation_phoneme_cleaned.tsv")
else:
    valid_tsv = os.path.join(TSV_DIR, "common_voice_de_validation_phoneme.tsv")

# Check if files exist
if not os.path.exists(train_tsv):
    print(f"   ❌ ERROR: Training file not found: {train_tsv}")
    if USE_CLEANED:
        print(f"   → Run: python clean_p2g_data_german.py --tsv_dir {TSV_DIR}")
    exit(1)

if not os.path.exists(valid_tsv):
    print(f"   ❌ ERROR: Validation file not found: {valid_tsv}")
    exit(1)

df_train = pd.read_csv(train_tsv, sep='\t')
df_valid = pd.read_csv(valid_tsv, sep='\t')

# Filter
df_train = df_train.dropna(subset=['phonemes', 'sentence'])
df_train = df_train[(df_train['phonemes'] != "") & (df_train['sentence'] != "")]
df_valid = df_valid.dropna(subset=['phonemes', 'sentence'])
df_valid = df_valid[(df_valid['phonemes'] != "") & (df_valid['sentence'] != "")]

print(f"   ✓ Train: {len(df_train)} samples")
print(f"   ✓ Valid: {len(df_valid)} samples")

# Show examples
print(f"\n   Example data:")
for i in range(min(2, len(df_train))):
    print(f"   {i+1}. Phoneme: {df_train.iloc[i]['phonemes'][:60]}...")
    print(f"      Text:    {df_train.iloc[i]['sentence'][:60]}...")

# ============================================================================
# STEP 2: LOAD TOKENIZER AND MODEL
# ============================================================================
print(f"\n[2/5] Loading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print(f"   ✓ Model: {MODEL_NAME}")
print(f"   ✓ Parameters: {model.num_parameters() / 1e6:.1f}M")
print(f"   ✓ Tokenizer vocab size: {len(tokenizer)}")

# ============================================================================
# STEP 3: PREPARE DATASETS
# ============================================================================
print(f"\n[3/5] Preparing datasets...")

def preprocess_function(examples):
    """Preprocess with proper label handling"""
    # Tokenize inputs (phonemes)
    model_inputs = tokenizer(
        examples['phonemes'],
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
        padding=False
    )
    
    # Tokenize targets (text)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples['sentence'],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
            padding=False
        )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Create datasets
train_dataset = Dataset.from_pandas(df_train[['phonemes', 'sentence']])
valid_dataset = Dataset.from_pandas(df_valid[['phonemes', 'sentence']])

print(f"   Tokenizing...")
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    desc="Train",
    remove_columns=train_dataset.column_names
)

valid_dataset = valid_dataset.map(
    preprocess_function,
    batched=True,
    desc="Valid",
    remove_columns=valid_dataset.column_names
)

print(f"   ✓ Train: {len(train_dataset)} samples")
print(f"   ✓ Valid: {len(valid_dataset)} samples")

# Validate data
print(f"\n   Validating first sample...")
sample = train_dataset[0]
print(f"      Input IDs: {len(sample['input_ids'])} tokens")
print(f"      Label IDs: {len(sample['labels'])} tokens")
print(f"      Input: {sample['input_ids'][:10]}")
print(f"      Labels: {sample['labels'][:10]}")

if all(x == -100 for x in sample['labels']):
    print(f"      ❌ ERROR: All labels are -100!")
    exit(1)
else:
    print(f"      ✓ Data looks valid")

# ============================================================================
# STEP 4: SETUP TRAINING
# ============================================================================
print(f"\n[4/5] Setting up training...")

# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100
)

# Calculate total steps
steps_per_epoch = len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION)
total_steps = steps_per_epoch * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

print(f"   Steps per epoch: {steps_per_epoch}")
print(f"   Total steps: {total_steps}")
print(f"   Warmup steps: {warmup_steps} ({WARMUP_RATIO*100:.0f}%)")

# Training arguments
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
    fp16=False,  # Disabled for debugging
    dataloader_num_workers=0,  # Disable multiprocessing
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to=[],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_first_step=True,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print(f"\n   Training configuration:")
print(f"      - Train samples: {len(train_dataset):,}")
print(f"      - Valid samples: {len(valid_dataset):,}")
print(f"      - Epochs: {NUM_EPOCHS}")
print(f"      - Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"      - Learning rate: {LEARNING_RATE}")
print(f"      - Warmup: {warmup_steps} steps")
print(f"      - FP16: Disabled (for debugging)")

# ============================================================================
# STEP 5: START TRAINING
# ============================================================================
print(f"\n{'='*70}")
print("Starting German P2G training...")
print("="*70 + "\n")

try:
    trainer.train()
    
    print("\n" + "="*70)
    print("✓ Training completed!")
    print("="*70)
    
    # Save
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save config
    config = {
        "model_name": MODEL_NAME,
        "language": "German",
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION,
        "learning_rate": LEARNING_RATE,
    }
    with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Cleanup
    import shutil, glob
    for checkpoint_dir in sorted(glob.glob(f"{OUTPUT_DIR}/checkpoint-*"))[:-1]:
        shutil.rmtree(checkpoint_dir)
    
    print(f"\n{'='*70}")
    print("🎉 GERMAN P2G TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Language: German (de)")
    print(f"{'='*70}\n")
    
except KeyboardInterrupt:
    print(f"\n⚠ Training interrupted")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()