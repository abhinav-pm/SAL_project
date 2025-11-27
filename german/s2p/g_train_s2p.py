#!/usr/bin/env python3
"""
Speech-to-Phoneme (S2P) Training Script - OPTIMIZED & FIXED
Processes audio on-the-fly during training from a large dataset,
using pre-defined TSV files to define the train/validation splits.
"""

import torch
import pandas as pd
from datasets import load_from_disk
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
import numpy as np
import soundfile as sf
import io
import resampy
from datasets import Audio as AudioFeature

warnings.filterwarnings('ignore')
os.environ["WANDB_DISABLED"] = "true"

print("="*70)
print(" "*12 + "Speech-to-PHONEME Training (OPTIMIZED)")
print(" "*10 + "(Using pre-defined train/valid TSV splits)")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_PATH = "/scratch/ABHI/common_voice_de_190h" 

TRAIN_TSV_PATH = "/home/abhinav.pm/ABHI/SAL/v2/german_phoneme_smallset/common_voice_de_train_phoneme.tsv"
VALID_TSV_PATH = "/home/abhinav.pm/ABHI/SAL/v2/german_phoneme_smallset/common_voice_de_validation_phoneme.tsv"

OUTPUT_DIR = "./s2p_model_german_phoneme_split"

# For initial testing
TEST_MODE = False
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
print(f"   Audio Dataset Path: {DATASET_PATH}")
print(f"   Train Phonemes TSV: {TRAIN_TSV_PATH}")
print(f"   Valid Phonemes TSV: {VALID_TSV_PATH}")
print(f"   Output dir: {OUTPUT_DIR}")
print(f"   Test mode: {TEST_MODE}")

# ============================================================================
# STEP 1: LOAD PHONEMIZED TSV FILES
# ============================================================================
print(f"\n[1/6] Loading phonemized TSV files...")

def load_and_filter_tsv(path):
    if not os.path.exists(path):
        print(f"   ❌ ERROR: TSV file not found at '{path}'!")
        print(f"   Please phonemize your split TSVs and update the path.")
        exit(1)
    df = pd.read_csv(path, sep='\t')
    df = df.dropna(subset=['phonemes'])
    df = df[df['phonemes'] != ""]
    return df

df_train = load_and_filter_tsv(TRAIN_TSV_PATH)
df_valid = load_and_filter_tsv(VALID_TSV_PATH)

print(f"   ✓ Loaded {len(df_train)} training samples with phonemes")
print(f"   ✓ Loaded {len(df_valid)} validation samples with phonemes")

# Create separate phoneme dictionaries for train and valid
def create_phoneme_dict(df):
    phoneme_dict = {}
    for _, row in df.iterrows():
        filename = os.path.basename(row['path'])
        phoneme_dict[filename] = row['phonemes']
    return phoneme_dict

phoneme_dict_train = create_phoneme_dict(df_train)
phoneme_dict_valid = create_phoneme_dict(df_valid)

print(f"   ✓ Created phoneme dictionaries:")
print(f"      - Train: {len(phoneme_dict_train)} entries")
print(f"      - Valid: {len(phoneme_dict_valid)} entries")

# Show examples
print(f"\n   Example phonemes from train:")
for i, (filename, phoneme) in enumerate(list(phoneme_dict_train.items())[:3]):
    phoneme_short = phoneme[:60] + "..." if len(phoneme) > 60 else phoneme
    print(f"   {i+1}. {filename}: {phoneme_short}")

# ============================================================================
# STEP 2: LOAD THE FULL DATASET (WITHOUT AUDIO DECODING)
# ============================================================================
print(f"\n[2/6] Loading the full HuggingFace dataset...")
dataset = load_from_disk(DATASET_PATH)
dataset = dataset.cast_column("audio", AudioFeature(decode=False))
print(f"   ✓ Loaded {len(dataset)} total samples from disk")

# PATH VERIFICATION - CRITICAL
print(f"\n🔍 Verifying path compatibility...")
print(f"   TSV path example: {df_train['path'].iloc[0]}")
print(f"   TSV filename: {os.path.basename(df_train['path'].iloc[0])}")
print(f"   Dataset path example: {dataset[0]['path']}")
print(f"   Dataset filename: {os.path.basename(dataset[0]['path'])}")

# ============================================================================
# STEP 3: IDENTIFY TRAIN/VALID INDICES FROM TSVs (FAST)
# ============================================================================
print(f"\n[3/6] Mapping TSV filenames to dataset indices...")

train_filenames = set(df_train['path'].apply(os.path.basename))
valid_filenames = set(df_valid['path'].apply(os.path.basename))

train_indices = []
valid_indices = []

for i in tqdm(range(len(dataset)), desc="   Indexing", ncols=80):
    filename = os.path.basename(dataset[i]['path'])
    
    if filename in train_filenames:
        train_indices.append(i)
    elif filename in valid_filenames:
        valid_indices.append(i)

print(f"   ✓ Matched {len(train_indices)} training indices")
print(f"   ✓ Matched {len(valid_indices)} validation indices")

# Check for missing samples
if len(train_indices) != len(df_train):
    print(f"   ⚠ WARNING: Expected {len(df_train)} train samples, found {len(train_indices)}")
if len(valid_indices) != len(df_valid):
    print(f"   ⚠ WARNING: Expected {len(df_valid)} valid samples, found {len(valid_indices)}")

if TEST_MODE:
    print(f"\n   ⚠ TEST MODE: Slicing datasets for a quick run...")
    train_indices = train_indices[:TEST_TRAIN_SIZE]
    valid_indices = valid_indices[:TEST_VALID_SIZE]
    print(f"      - Using {len(train_indices)} training samples")
    print(f"      - Using {len(valid_indices)} validation samples")

# ============================================================================
# STEP 4: CREATE PHONEME VOCABULARY (FROM TRAINING SET ONLY)
# ============================================================================
print(f"\n[4/6] Creating PHONEME vocabulary from training data...")

train_phonemes = []
for _, row in df_train.iterrows():
    train_phonemes.append(row['phonemes'])

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
print(f"   Sample phonemes: {list(vocab_dict.keys())[:10]}")

# ============================================================================
# STEP 5: CREATE PROCESSOR
# ============================================================================
print(f"\n[5/6] Creating processor...")

tokenizer = Wav2Vec2CTCTokenizer(
    vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1, sampling_rate=16000, padding_value=0.0, 
    do_normalize=True, return_attention_mask=True)

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor, tokenizer=tokenizer)

processor.save_pretrained(OUTPUT_DIR)
print(f"   ✓ Processor created and saved")

# ============================================================================
# STEP 6: DEFINE ON-THE-FLY DATASET
# ============================================================================
print(f"\n[6/6] Defining on-the-fly dataset class...")

class OnTheFlyAudioDataset(torch.utils.data.Dataset):
    """
    Custom dataset that loads and processes audio ON-THE-FLY
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
            dataset_idx = self.indices[idx]
            row_dict = self.dataset[dataset_idx]
            
            filename = os.path.basename(row_dict['path'])
            
            # Check if phoneme exists
            if filename not in self.phoneme_dict:
                raise KeyError(f"Phoneme not found for {filename}")
            
            phoneme = self.phoneme_dict[filename]
            
            # Audio is loaded from the 'bytes' field
            audio_bytes = row_dict['audio']['bytes']
            audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Resample if needed
            if sample_rate != 16000:
                audio_array = resampy.resample(audio_array, sample_rate, 16000)
            
            # Process audio
            input_values = self.processor(
                audio_array, sampling_rate=16000).input_values[0]
            
            # Process phonemes
            with self.processor.as_target_processor():
                labels = self.processor(phoneme).input_ids
            
            return {"input_values": input_values, "labels": labels}
        
        except Exception as e:
            # Return dummy sample on error instead of crashing
            print(f"\n⚠ Error loading sample {idx}: {str(e)[:100]}")
            
            # Create minimal valid sample
            dummy_audio = np.zeros(16000, dtype=np.float32)
            input_values = self.processor(
                dummy_audio, sampling_rate=16000).input_values[0]
            
            with self.processor.as_target_processor():
                labels = self.processor(" ").input_ids
            
            return {"input_values": input_values, "labels": labels}

# Create datasets
train_dataset = OnTheFlyAudioDataset(dataset, train_indices, phoneme_dict_train, processor)
valid_dataset = OnTheFlyAudioDataset(dataset, valid_indices, phoneme_dict_valid, processor)

print(f"   ✓ On-the-fly datasets created")
print(f"      - Train: {len(train_dataset)} samples")
print(f"      - Valid: {len(valid_dataset)} samples")

# ============================================================================
# SETUP MODEL AND TRAINING
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

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR, 
    group_by_length=False,  # ✅ FIXED: Disabled for custom datasets
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
    save_total_limit=2,
    dataloader_num_workers=4,
    logging_dir=f"{OUTPUT_DIR}/logs",
    report_to=[],
    remove_unused_columns=False,  # ✅ FIXED: Critical for custom datasets
    disable_tqdm=False,
    logging_first_step=True
)

trainer = Trainer(
    model=model, 
    data_collator=data_collator, 
    args=training_args,
    train_dataset=train_dataset, 
    eval_dataset=valid_dataset,
    tokenizer=processor.feature_extractor
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
print("Starting training with validation...")
print("Audio will be loaded and processed ON-THE-FLY during training!")
print(f"{'='*70}\n")

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    trainer.train()
    # trainer.train(resume_from_checkpoint=True) #use this for resuming training from last checkpoint
    
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