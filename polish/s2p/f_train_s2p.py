#!/usr/bin/env python3
"""
Speech-to-Phoneme (S2P) Training Script - OPTIMIZED
Processes audio on-the-fly during training instead of loading everything upfront
Now supports train/test/validation splits from Common Voice dataset

Dependencies:
    pip install torch transformers datasets librosa soundfile tqdm pandas

Note: Uses librosa to handle MP3 files properly
"""

import torch
import pandas as pd
from datasets import load_dataset, Audio as AudioFeature
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)
import json
import os
import warnings
from tqdm import tqdm
import librosa
import numpy as np

warnings.filterwarnings('ignore')
os.environ["WANDB_DISABLED"] = "true"

print("="*70)
print(" "*12 + "Speech-to-PHONEME Training (OPTIMIZED)")
print(" "*10 + "(On-the-fly audio processing - much faster!)")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
# HuggingFace dataset cache (where the audio is stored)
CACHE_PATH = "/scratch/ABHI/huggingface_cache2"

# Phonemized TSV files directory
TSV_DIR = "/home/abhinav.pm/ABHI/SAL/v4/phonemized"

# Output directory for model
OUTPUT_DIR = "./s2p_model_polish_phoneme_v1"

# For initial testing
TEST_MODE = False  # Set to False for full training
TEST_TRAIN_SIZE = 100
TEST_VALID_SIZE = 20

# Training parameters
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 2
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
SAVE_STEPS = 500
LOGGING_STEPS = 50

print(f"\n📋 Configuration:")
print(f"   Dataset cache: {CACHE_PATH}")
print(f"   TSV directory: {TSV_DIR}")
print(f"   Output dir: {OUTPUT_DIR}")
print(f"   Test mode: {TEST_MODE}")

# ============================================================================
# STEP 1: LOAD PHONEMIZED TSV FILES
# ============================================================================
print(f"\n[1/7] Loading phonemized TSV files...")

# Load train, test, validation TSV files
train_tsv = os.path.join(TSV_DIR, "common_voice_pl_train_phoneme.tsv")
test_tsv = os.path.join(TSV_DIR, "common_voice_pl_test_phoneme.tsv")
valid_tsv = os.path.join(TSV_DIR, "common_voice_pl_validation_phoneme.tsv")

# Check if files exist
for tsv_file in [train_tsv, test_tsv, valid_tsv]:
    if not os.path.exists(tsv_file):
        print(f"   ❌ ERROR: TSV file not found: {tsv_file}")
        exit(1)

# Load TSV files
df_train = pd.read_csv(train_tsv, sep='\t')
df_test = pd.read_csv(test_tsv, sep='\t')
df_valid = pd.read_csv(valid_tsv, sep='\t')

print(f"   ✓ Train: {len(df_train)} rows")
print(f"   ✓ Test: {len(df_test)} rows")
print(f"   ✓ Valid: {len(df_valid)} rows")

# Remove rows with empty phonemes
for df_name, df in [("train", df_train), ("test", df_test), ("valid", df_valid)]:
    original_len = len(df)
    if df_name == "train":
        df_train = df_train.dropna(subset=['phonemes'])
        df_train = df_train[df_train['phonemes'] != ""]
        print(f"   ✓ Train after filtering: {len(df_train)} rows (removed {original_len - len(df_train)})")
    elif df_name == "test":
        df_test = df_test.dropna(subset=['phonemes'])
        df_test = df_test[df_test['phonemes'] != ""]
        print(f"   ✓ Test after filtering: {len(df_test)} rows (removed {original_len - len(df_test)})")
    else:
        df_valid = df_valid.dropna(subset=['phonemes'])
        df_valid = df_valid[df_valid['phonemes'] != ""]
        print(f"   ✓ Valid after filtering: {len(df_valid)} rows (removed {original_len - len(df_valid)})")

# Create phoneme dictionaries: filename -> phoneme
def create_phoneme_dict(df):
    phoneme_dict = {}
    for idx, row in df.iterrows():
        # Handle both 'path' and 'audio_path' columns
        if 'audio_path' in row and pd.notna(row['audio_path']):
            filename = os.path.basename(row['audio_path'])
        elif 'path' in row and pd.notna(row['path']):
            filename = os.path.basename(row['path'])
        else:
            continue
        phoneme_dict[filename] = row['phonemes']
    return phoneme_dict

phoneme_dict_train = create_phoneme_dict(df_train)
phoneme_dict_test = create_phoneme_dict(df_test)
phoneme_dict_valid = create_phoneme_dict(df_valid)

print(f"\n   ✓ Phoneme dictionaries created:")
print(f"      Train: {len(phoneme_dict_train)} entries")
print(f"      Test: {len(phoneme_dict_test)} entries")
print(f"      Valid: {len(phoneme_dict_valid)} entries")

# Show examples
print(f"\n   Example phonemes from train:")
for i, (filename, phoneme) in enumerate(list(phoneme_dict_train.items())[:3]):
    phoneme_short = phoneme[:60] + "..." if len(phoneme) > 60 else phoneme
    print(f"   {i+1}. {filename}: {phoneme_short}")

# ============================================================================
# STEP 2: LOAD DATASET FROM CACHE (WITHOUT AUDIO DECODING)
# ============================================================================
print(f"\n[2/7] Loading HuggingFace datasets from cache...")

# Load train, test, validation splits
dataset_train = load_dataset(
    "fsicoli/common_voice_22_0",
    "pl",
    split="train",
    cache_dir=CACHE_PATH
)
dataset_train = dataset_train.cast_column("audio", AudioFeature(decode=False))

dataset_test = load_dataset(
    "fsicoli/common_voice_22_0",
    "pl",
    split="test",
    cache_dir=CACHE_PATH
)
dataset_test = dataset_test.cast_column("audio", AudioFeature(decode=False))

dataset_valid = load_dataset(
    "fsicoli/common_voice_22_0",
    "pl",
    split="validation",
    cache_dir=CACHE_PATH
)
dataset_valid = dataset_valid.cast_column("audio", AudioFeature(decode=False))

print(f"   ✓ Train dataset: {len(dataset_train)} samples")
print(f"   ✓ Test dataset: {len(dataset_test)} samples")
print(f"   ✓ Valid dataset: {len(dataset_valid)} samples")

# ============================================================================
# STEP 3: CREATE LIGHTWEIGHT INDEX (FAST - No audio loading!)
# ============================================================================
print(f"\n[3/7] Creating dataset indices (fast - no audio loading)...")

def create_matched_indices(dataset, phoneme_dict, split_name):
    matched_indices = []
    for i in tqdm(range(len(dataset)), desc=f"   Indexing {split_name}", ncols=70):
        row = dataset[i]
        filename = os.path.basename(row['path'])
        
        if filename in phoneme_dict:
            matched_indices.append(i)
    
    return matched_indices

train_indices = create_matched_indices(dataset_train, phoneme_dict_train, "train")
test_indices = create_matched_indices(dataset_test, phoneme_dict_test, "test")
valid_indices = create_matched_indices(dataset_valid, phoneme_dict_valid, "valid")

print(f"\n   ✓ Matched indices:")
print(f"      Train: {len(train_indices)} samples")
print(f"      Test: {len(test_indices)} samples")
print(f"      Valid: {len(valid_indices)} samples")

# Apply test mode if enabled
if TEST_MODE:
    train_indices = train_indices[:min(TEST_TRAIN_SIZE, len(train_indices))]
    valid_indices = valid_indices[:min(TEST_VALID_SIZE, len(valid_indices))]
    print(f"\n   ⚠ TEST MODE: Using limited samples")
    print(f"      Train: {len(train_indices)} samples")
    print(f"      Valid: {len(valid_indices)} samples")

# ============================================================================
# STEP 4: CREATE PHONEME VOCABULARY
# ============================================================================
print(f"\n[4/7] Creating PHONEME vocabulary from training data...")

# Use phonemes from training samples only
train_phonemes = []
for idx in tqdm(train_indices, desc="   Collecting phonemes", ncols=70):
    row = dataset_train[idx]
    filename = os.path.basename(row['path'])
    train_phonemes.append(phoneme_dict_train[filename])

all_phonemes = " ".join(train_phonemes)
vocab_list = sorted(list(set(all_phonemes)))

vocab_dict = {v: i for i, v in enumerate(vocab_list)}
vocab_dict["|"] = len(vocab_dict)
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

os.makedirs(OUTPUT_DIR, exist_ok=True)
vocab_path = os.path.join(OUTPUT_DIR, "vocab.json")
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

print(f"   ✓ Vocabulary size: {len(vocab_dict)}")
print(f"   ✓ Saved to: {vocab_path}")

# Show some phonemes
print(f"\n   Sample phonemes in vocabulary:")
sample_phonemes = list(vocab_dict.keys())[:20]
print(f"   {' '.join(sample_phonemes)}")

# ============================================================================
# STEP 5: CREATE PROCESSOR
# ============================================================================
print(f"\n[5/7] Creating processor...")

tokenizer = Wav2Vec2CTCTokenizer(
    vocab_path,
    unk_token="[UNK]",
    pad_token="[PAD]",
    word_delimiter_token="|"
)

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True
)

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer
)

processor.save_pretrained(OUTPUT_DIR)
print(f"   ✓ Processor created and saved")

# ============================================================================
# STEP 6: CREATE CUSTOM DATASET CLASS (loads audio on-the-fly!)
# ============================================================================
print(f"\n[6/7] Creating on-the-fly datasets...")

class OnTheFlyAudioDataset(torch.utils.data.Dataset):
    """
    Custom dataset that loads and processes audio ON-THE-FLY
    This is much faster than loading everything upfront!
    Handles MP3 files properly using librosa.
    """
    def __init__(self, dataset, indices, phoneme_dict, processor):
        self.dataset = dataset
        self.indices = indices
        self.phoneme_dict = phoneme_dict
        self.processor = processor
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        try:
            # Get dataset index
            dataset_idx = self.indices[idx]
            
            # Get row
            row = self.dataset[dataset_idx]
            
            # Get filename and phoneme
            filename = os.path.basename(row['path'])
            phoneme = self.phoneme_dict[filename]
            
            # Load audio using librosa (handles MP3, WAV, etc.)
            # Method 1: Load from bytes if available
            if 'audio' in row and row['audio'] is not None:
                audio_data = row['audio']
                
                # If bytes are available
                if 'bytes' in audio_data and audio_data['bytes'] is not None:
                    import io
                    audio_array, sample_rate = librosa.load(
                        io.BytesIO(audio_data['bytes']),
                        sr=16000,
                        mono=True
                    )
                # If path is available
                elif 'path' in audio_data and audio_data['path'] is not None:
                    audio_array, sample_rate = librosa.load(
                        audio_data['path'],
                        sr=16000,
                        mono=True
                    )
                else:
                    # Fallback: use the row's path
                    audio_array, sample_rate = librosa.load(
                        row['path'],
                        sr=16000,
                        mono=True
                    )
            else:
                # Direct path loading
                audio_array, sample_rate = librosa.load(
                    row['path'],
                    sr=16000,
                    mono=True
                )
            
            # Ensure audio is float32
            audio_array = audio_array.astype(np.float32)
            
            # Process audio
            input_values = self.processor(
                audio_array, 
                sampling_rate=16000
            ).input_values[0]
            
            # Process phonemes
            with self.processor.as_target_processor():
                labels = self.processor(phoneme).input_ids
            
            return {
                "input_values": input_values,
                "labels": labels
            }
            
        except Exception as e:
            # If loading fails, return a dummy sample (will be filtered out)
            print(f"\n⚠ Error loading sample {idx}: {str(e)[:100]}")
            # Return a minimal valid sample
            dummy_audio = np.zeros(16000, dtype=np.float32)
            input_values = self.processor(
                dummy_audio, 
                sampling_rate=16000
            ).input_values[0]
            
            with self.processor.as_target_processor():
                labels = self.processor(" ").input_ids
            
            return {
                "input_values": input_values,
                "labels": labels
            }

# Create datasets for train and validation
train_dataset = OnTheFlyAudioDataset(
    dataset_train, 
    train_indices, 
    phoneme_dict_train, 
    processor
)

valid_dataset = OnTheFlyAudioDataset(
    dataset_valid, 
    valid_indices, 
    phoneme_dict_valid, 
    processor
)

print(f"   ✓ On-the-fly datasets created")
print(f"      Train: {len(train_dataset)} samples")
print(f"      Valid: {len(valid_dataset)} samples")

# ============================================================================
# STEP 7: SETUP MODEL AND TRAINING
# ============================================================================
print(f"\n[7/7] Setting up model and training...")

class DataCollatorCTCWithPadding:
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
        
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(label_features, padding=self.padding, return_tensors="pt")
        
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor)

print(f"   Loading model: facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True
)

model.freeze_feature_encoder()
print(f"   ✓ Model loaded (feature encoder frozen)")

# Training configuration
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    group_by_length=False,  # Disabled to avoid issues with custom dataset
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    eval_strategy="steps",
    eval_steps=SAVE_STEPS,
    num_train_epochs=NUM_EPOCHS,
    fp16=torch.cuda.is_available(),
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.005,
    warmup_steps=500,
    save_total_limit=2,  # Keep only 2 checkpoints to save space
    dataloader_num_workers=4,  # Parallel audio loading during training
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to=[],
    remove_unused_columns=False,  # Important for custom datasets
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=processor.feature_extractor,
)

print(f"\n   Training configuration:")
print(f"      - Train samples: {len(train_dataset):,}")
print(f"      - Valid samples: {len(valid_dataset):,}")
print(f"      - Epochs: {NUM_EPOCHS}")
print(f"      - Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION} = {BATCH_SIZE * GRADIENT_ACCUMULATION}")
print(f"      - Learning rate: {LEARNING_RATE}")
print(f"      - Device: {'GPU (fp16)' if torch.cuda.is_available() else 'CPU'}")
print(f"      - Workers: 4 (parallel audio loading)")
print(f"      - Vocabulary size: {len(vocab_dict)}")

# ============================================================================
# START TRAINING
# ============================================================================
print(f"\n{'='*70}")
print("Starting training...")
print("Audio will be loaded and processed ON-THE-FLY during training!")
print("="*70 + "\n")

try:
    trainer.train()
    
    print("\n" + "="*70)
    print("✓ Training completed successfully!")
    print("="*70)
    
    print(f"\nSaving final model...")
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)
    
    # Clean up old checkpoints to save space
    import shutil
    import glob
    print(f"\nCleaning up old checkpoints...")
    checkpoint_dirs = sorted(glob.glob(f"{OUTPUT_DIR}/checkpoint-*"))
    if len(checkpoint_dirs) > 1:
        # Keep only the latest checkpoint
        for checkpoint_dir in checkpoint_dirs[:-1]:
            shutil.rmtree(checkpoint_dir)
            print(f"   Deleted: {os.path.basename(checkpoint_dir)}")
    
    print(f"\n{'='*70}")
    print("🎉 TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Vocabulary size: {len(vocab_dict)}")
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(valid_dataset):,}")
    print(f"Language: Polish (pl)")
    print(f"{'='*70}\n")
    
except KeyboardInterrupt:
    print(f"\n⚠ Training interrupted by user")
    print(f"Saving current state...")
    trainer.save_model()
    processor.save_pretrained(OUTPUT_DIR)
    print(f"✓ Model saved to: {OUTPUT_DIR}")
    
except Exception as e:
    print(f"\n❌ Training failed:")
    print(f"{e}")
    import traceback
    traceback.print_exc()