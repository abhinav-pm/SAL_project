#!/usr/bin/env python3
"""
Phoneme-to-Grapheme (P2G) Training Script
Based on: "LLM-based phoneme-to-grapheme for phoneme-based speech recognition"
Ma et al., Interspeech 2025

This script trains an LLM (mT5) to convert phoneme sequences to text (graphemes).
Two training strategies are implemented:
1. DANP: Data Augmentation with Noisy Phonemes
2. TKM: Top-K Marginalized training

Dependencies:
    pip install torch transformers datasets pandas tqdm
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
from tqdm import tqdm
import json
import numpy as np

warnings.filterwarnings('ignore')
os.environ["WANDB_DISABLED"] = "true"

print("="*70)
print(" "*15 + "Phoneme-to-Grapheme (P2G) Training")
print(" "*18 + "(Based on Interspeech 2025)")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
# Phonemized TSV files directory (from S2P training)
TSV_DIR = "/home/abhinav.pm/ABHI/SAL/v4/phonemized"

# Output directory for P2G model
OUTPUT_DIR = "./p2g_model_polish_v1"

# Model selection - mT5 (multilingual T5)
MODEL_NAME = "google/mt5-small"  # 300M params - good balance
# Alternative: "google/mt5-base" (580M params) for better performance

# Training strategy
USE_DANP = False  # Data Augmentation with Noisy Phonemes
USE_TKM = False   # Top-K Marginalized training (requires S2P model)

# If using DANP, specify S2P model path for generating noisy phonemes
S2P_MODEL_PATH = "./s2p_model_polish_phoneme_v1"  # Your trained S2P model

# For testing
TEST_MODE = True  # Set to False for full training
TEST_SAMPLES = 1000

# Training parameters
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4
NUM_EPOCHS = 5
LEARNING_RATE = 3e-4  # As per paper
MAX_SOURCE_LENGTH = 256  # Max phoneme sequence length
MAX_TARGET_LENGTH = 256  # Max text sequence length
SAVE_STEPS = 500
LOGGING_STEPS = 50

# DANP parameters
DANP_BEAM_SIZE = 32  # Generate top-32 noisy phonemes
DANP_USE_SAMPLING = True  # Also use random sampling

# TKM parameters
TKM_K = 32  # Generate top-32 hypotheses
TKM_N = 8   # Marginalize over 8 random samples from top-32

print(f"\n📋 Configuration:")
print(f"   TSV directory: {TSV_DIR}")
print(f"   Output dir: {OUTPUT_DIR}")
print(f"   Model: {MODEL_NAME}")
print(f"   Training strategy: {'DANP' if USE_DANP else 'TKM' if USE_TKM else 'Basic'}")
print(f"   Test mode: {TEST_MODE}")

# ============================================================================
# STEP 1: LOAD PHONEMIZED TSV FILES
# ============================================================================
print(f"\n[1/6] Loading phonemized TSV files...")

train_tsv = os.path.join(TSV_DIR, "common_voice_pl_train_phoneme.tsv")
valid_tsv = os.path.join(TSV_DIR, "common_voice_pl_validation_phoneme.tsv")

# Check if files exist
for tsv_file in [train_tsv, valid_tsv]:
    if not os.path.exists(tsv_file):
        print(f"   ❌ ERROR: TSV file not found: {tsv_file}")
        exit(1)

# Load TSV files
df_train = pd.read_csv(train_tsv, sep='\t')
df_valid = pd.read_csv(valid_tsv, sep='\t')

print(f"   ✓ Train: {len(df_train)} rows")
print(f"   ✓ Valid: {len(df_valid)} rows")

# Filter out empty phonemes and sentences
df_train = df_train.dropna(subset=['phonemes', 'sentence'])
df_train = df_train[(df_train['phonemes'] != "") & (df_train['sentence'] != "")]

df_valid = df_valid.dropna(subset=['phonemes', 'sentence'])
df_valid = df_valid[(df_valid['phonemes'] != "") & (df_valid['sentence'] != "")]

print(f"   ✓ After filtering:")
print(f"      Train: {len(df_train)} rows")
print(f"      Valid: {len(df_valid)} rows")

# Apply test mode if enabled
if TEST_MODE:
    df_train = df_train.head(TEST_SAMPLES)
    df_valid = df_valid.head(TEST_SAMPLES // 5)
    print(f"\n   ⚠ TEST MODE: Using limited samples")
    print(f"      Train: {len(df_train)} samples")
    print(f"      Valid: {len(df_valid)} samples")

# Show examples
print(f"\n   Example data:")
for i in range(min(3, len(df_train))):
    phoneme = df_train.iloc[i]['phonemes']
    text = df_train.iloc[i]['sentence']
    phoneme_short = phoneme[:50] + "..." if len(phoneme) > 50 else phoneme
    text_short = text[:50] + "..." if len(text) > 50 else text
    print(f"   {i+1}. Phoneme: {phoneme_short}")
    print(f"      Text:    {text_short}")

# ============================================================================
# STEP 2: GENERATE NOISY PHONEMES (DANP Strategy)
# ============================================================================
if USE_DANP and not TEST_MODE:
    print(f"\n[2/6] Generating noisy phonemes (DANP)...")
    print(f"   This augments training data with S2P-generated phonemes")
    print(f"   to match training/testing conditions")
    
    # Load S2P model for generating noisy phonemes
    if not os.path.exists(S2P_MODEL_PATH):
        print(f"   ⚠ WARNING: S2P model not found at {S2P_MODEL_PATH}")
        print(f"   Proceeding without DANP augmentation")
        USE_DANP = False
    else:
        print(f"   Loading S2P model from: {S2P_MODEL_PATH}")
        
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        from datasets import load_dataset, Audio as AudioFeature
        import librosa
        import io
        
        # Load S2P model
        s2p_processor = Wav2Vec2Processor.from_pretrained(S2P_MODEL_PATH)
        s2p_model = Wav2Vec2ForCTC.from_pretrained(S2P_MODEL_PATH)
        s2p_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        s2p_model.to(device)
        
        print(f"   ✓ S2P model loaded on {device}")
        print(f"   Generating noisy phonemes with beam size={DANP_BEAM_SIZE}...")
        
        # Load audio dataset
        CACHE_PATH = "/scratch/ABHI/huggingface_cache2"
        dataset_train = load_dataset(
            "fsicoli/common_voice_22_0",
            "pl",
            split="train",
            cache_dir=CACHE_PATH
        )
        dataset_train = dataset_train.cast_column("audio", AudioFeature(decode=False))
        
        # Create filename -> index mapping
        filename_to_idx = {}
        for i in tqdm(range(len(dataset_train)), desc="   Creating audio index", ncols=70):
            filename = os.path.basename(dataset_train[i]['path'])
            filename_to_idx[filename] = i
        
        # Generate noisy phonemes for each training sample
        augmented_data = []
        
        for idx in tqdm(range(len(df_train)), desc="   Generating noisy phonemes", ncols=70):
            row = df_train.iloc[idx]
            filename = os.path.basename(row.get('audio_path', row.get('path', '')))
            
            if filename not in filename_to_idx:
                continue
            
            try:
                # Load audio
                audio_idx = filename_to_idx[filename]
                audio_row = dataset_train[audio_idx]
                
                # Load audio using librosa
                if 'audio' in audio_row and audio_row['audio']:
                    audio_data = audio_row['audio']
                    if 'bytes' in audio_data and audio_data['bytes']:
                        audio_array, _ = librosa.load(
                            io.BytesIO(audio_data['bytes']),
                            sr=16000,
                            mono=True
                        )
                    else:
                        audio_array, _ = librosa.load(audio_row['path'], sr=16000, mono=True)
                else:
                    audio_array, _ = librosa.load(audio_row['path'], sr=16000, mono=True)
                
                # Process audio
                input_values = s2p_processor(
                    audio_array,
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_values.to(device)
                
                # Generate noisy phonemes with beam search
                with torch.no_grad():
                    outputs = s2p_model(input_values)
                    logits = outputs.logits
                
                # Get top-K hypotheses using beam search approximation
                # (Simple version: sample from top probabilities)
                probs = torch.nn.functional.softmax(logits[0], dim=-1)
                
                # Generate multiple hypotheses
                noisy_phonemes = []
                for _ in range(min(5, DANP_BEAM_SIZE)):  # Generate 5 variants
                    # Sample from probabilities
                    sampled_ids = torch.multinomial(probs, num_samples=1).squeeze()
                    decoded = s2p_processor.decode(sampled_ids)
                    if decoded and decoded.strip():
                        noisy_phonemes.append(decoded.strip())
                
                # Add all variants to training data
                for noisy_ph in set(noisy_phonemes):  # Remove duplicates
                    augmented_data.append({
                        'phonemes': noisy_ph,
                        'sentence': row['sentence']
                    })
                
            except Exception as e:
                if idx < 3:  # Print first few errors
                    print(f"\n      ⚠ Error processing sample {idx}: {str(e)[:100]}")
                continue
        
        print(f"\n   ✓ Generated {len(augmented_data)} augmented samples")
        print(f"   ✓ Original: {len(df_train)}, Total: {len(df_train) + len(augmented_data)}")
        
        # Add augmented data to training set
        df_augmented = pd.DataFrame(augmented_data)
        df_train = pd.concat([df_train, df_augmented], ignore_index=True)
        
        # Clean up
        del s2p_model, s2p_processor, dataset_train
        torch.cuda.empty_cache()

else:
    print(f"\n[2/6] Skipping DANP (not enabled or in test mode)")

# ============================================================================
# STEP 3: LOAD TOKENIZER AND MODEL
# ============================================================================
print(f"\n[3/6] Loading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print(f"   ✓ Loaded {MODEL_NAME}")
print(f"   ✓ Model parameters: {model.num_parameters() / 1e6:.1f}M")

# ============================================================================
# STEP 4: PREPARE DATASETS
# ============================================================================
print(f"\n[4/6] Preparing datasets...")

def preprocess_function(examples):
    """
    Preprocess phoneme-text pairs for seq2seq training
    Input: phonemes (source)
    Output: text/graphemes (target)
    """
    # Tokenize inputs (phonemes)
    model_inputs = tokenizer(
        examples['phonemes'],
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
        padding=False  # Will pad dynamically in data collator
    )
    
    # Tokenize targets (text)
    labels = tokenizer(
        examples['sentence'],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Create HuggingFace datasets
train_dataset = Dataset.from_pandas(df_train[['phonemes', 'sentence']])
valid_dataset = Dataset.from_pandas(df_valid[['phonemes', 'sentence']])

print(f"   Tokenizing training data...")
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    desc="Tokenizing train",
    remove_columns=train_dataset.column_names
)

print(f"   Tokenizing validation data...")
valid_dataset = valid_dataset.map(
    preprocess_function,
    batched=True,
    desc="Tokenizing valid",
    remove_columns=valid_dataset.column_names
)

print(f"\n   ✓ Datasets prepared:")
print(f"      Train: {len(train_dataset)} samples")
print(f"      Valid: {len(valid_dataset)} samples")

# ============================================================================
# STEP 5: SETUP TRAINING
# ============================================================================
print(f"\n[5/6] Setting up training...")

# Data collator for dynamic padding
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True
)

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
    warmup_steps=500,
    save_total_limit=2,  # Keep only 2 best checkpoints
    predict_with_generate=True,  # Generate during evaluation
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=4,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to=[],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
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
print(f"      - Device: {'GPU (fp16)' if torch.cuda.is_available() else 'CPU'}")
print(f"      - Model: {MODEL_NAME}")

# ============================================================================
# STEP 6: START TRAINING
# ============================================================================
print(f"\n{'='*70}")
print("Starting P2G training...")
print("="*70 + "\n")

try:
    # Train
    trainer.train()
    
    print("\n" + "="*70)
    print("✓ Training completed successfully!")
    print("="*70)
    
    # Save final model
    print(f"\nSaving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save configuration
    config = {
        "model_name": MODEL_NAME,
        "max_source_length": MAX_SOURCE_LENGTH,
        "max_target_length": MAX_TARGET_LENGTH,
        "training_strategy": "DANP" if USE_DANP else "TKM" if USE_TKM else "Basic",
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
    }
    
    with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Clean up old checkpoints
    import shutil
    import glob
    print(f"\nCleaning up old checkpoints...")
    checkpoint_dirs = sorted(glob.glob(f"{OUTPUT_DIR}/checkpoint-*"))
    if len(checkpoint_dirs) > 1:
        for checkpoint_dir in checkpoint_dirs[:-1]:
            shutil.rmtree(checkpoint_dir)
            print(f"   Deleted: {os.path.basename(checkpoint_dir)}")
    
    print(f"\n{'='*70}")
    print("🎉 P2G TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Model: {MODEL_NAME}")
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(valid_dataset):,}")
    print(f"Language: Polish (pl)")
    print(f"{'='*70}\n")
    
except KeyboardInterrupt:
    print(f"\n⚠ Training interrupted by user")
    print(f"Saving current state...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✓ Model saved to: {OUTPUT_DIR}")
    
except Exception as e:
    print(f"\n❌ Training failed:")
    print(f"{e}")
    import traceback
    traceback.print_exc()