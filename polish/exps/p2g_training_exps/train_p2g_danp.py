# #!/usr/bin/env python3
# """
# P2G Training with DANP Data
# - Fine-tunes an mT5 model on a large, augmented DANP dataset.
# - This script is used for training with the de-duplicated beam, sampling,
#   or combined beam+sampling datasets.

# Usage:
#     python g_train_p2g_danp.py \
#         --train_tsv ./p2g_final_datasets/train_danp_beam_plus_sampling_dedup.tsv \
#         --valid_tsv /path/to/your/validation_phoneme_cleaned.tsv \
#         --output_dir ./p2g_model_german_danp_combined \
#         --model_name google/mt5-small
# """

# import torch
# import pandas as pd
# from datasets import Dataset
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSeq2SeqLM,
#     DataCollatorForSeq2Seq,
#     Seq2SeqTrainingArguments,
#     Seq2SeqTrainer
# )
# import os
# import warnings
# import json
# import argparse

# warnings.filterwarnings('ignore')
# os.environ["WANDB_DISABLED"] = "true"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# def main():
#     parser = argparse.ArgumentParser(description="Train P2G model with DANP augmented data.")
#     parser.add_argument("--train_tsv", type=str, required=True, help="Path to the augmented training TSV (e.g., beam+sampling dedup).")
#     parser.add_argument("--valid_tsv", type=str, required=True, help="Path to the validation TSV.")
#     parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model.")
#     parser.add_argument("--model_name", type=str, default="google/mt5-small", help="Pre-trained model to fine-tune.")
    
#     # Training Hyperparameters
#     parser.add_argument("--batch_size", type=int, default=8)
#     parser.add_argument("--grad_accum", type=int, default=4)
#     parser.add_argument("--epochs", type=int, default=10)
#     parser.add_argument("--lr", type=float, default=3e-4)
#     parser.add_argument("--max_source_len", type=int, default=256)
#     parser.add_argument("--max_target_len", type=int, default=256)
    
#     args = parser.parse_args()

#     print("="*70)
#     print(" "*18 + "P2G Training with DANP")
#     print("="*70)
#     print(f"\n📋 Configuration:")
#     for arg, value in vars(args).items():
#         print(f"   {arg}: {value}")

#     # --- 1. Load Data ---
#     print("\n[1/5] Loading datasets...")
#     df_train = pd.read_csv(args.train_tsv, sep='\t')
#     df_valid = pd.read_csv(args.valid_tsv, sep='\t')

#     df_train = df_train.dropna(subset=['phonemes', 'sentence'])
#     df_valid = df_valid.dropna(subset=['phonemes', 'sentence'])

#     print(f"   ✓ Train: {len(df_train):,} samples")
#     print(f"   ✓ Valid: {len(df_valid):,} samples")

#     # --- 2. Load Model & Tokenizer ---
#     print("\n[2/5] Loading model and tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name, legacy=False)
#     model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
#     print(f"   ✓ Model: {args.model_name} ({model.num_parameters() / 1e6:.1f}M params)")

#     # --- 3. Prepare Datasets ---
#     print("\n[3/5] Preparing and tokenizing datasets...")
#     def preprocess_function(examples):
#         model_inputs = tokenizer(
#             examples['phonemes'], max_length=args.max_source_len, truncation=True
#         )
#         with tokenizer.as_target_tokenizer():
#             labels = tokenizer(
#                 examples['sentence'], max_length=args.max_target_len, truncation=True
#             )
#         model_inputs["labels"] = labels["input_ids"]
#         return model_inputs

#     train_dataset = Dataset.from_pandas(df_train[['phonemes', 'sentence']])
#     valid_dataset = Dataset.from_pandas(df_valid[['phonemes', 'sentence']])

#     train_dataset = train_dataset.map(preprocess_function, batched=True, desc="Tokenizing Train", remove_columns=train_dataset.column_names)
#     valid_dataset = valid_dataset.map(preprocess_function, batched=True, desc="Tokenizing Valid", remove_columns=valid_dataset.column_names)

#     print(f"   ✓ Train: {len(train_dataset):,} samples")
#     print(f"   ✓ Valid: {len(valid_dataset):,} samples")

#     # --- 4. Setup Training ---
#     print("\n[4/5] Setting up training...")
#     data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
#     training_args = Seq2SeqTrainingArguments(
#         output_dir=args.output_dir,
#         eval_strategy="steps",
#         eval_steps=500,
#         save_steps=500,
#         logging_steps=50,
#         learning_rate=args.lr,
#         per_device_train_batch_size=args.batch_size,
#         per_device_eval_batch_size=args.batch_size,
#         gradient_accumulation_steps=args.grad_accum,
#         num_train_epochs=args.epochs,
#         weight_decay=0.01,
#         warmup_ratio=0.1,
#         save_total_limit=2,
#         predict_with_generate=True,
#         generation_max_length=args.max_target_len,
#         fp16=torch.cuda.is_available(),
#         logging_dir=f"{args.output_dir}/logs",
#         report_to=[],
#         load_best_model_at_end=True,
#         metric_for_best_model="eval_loss",
#         greater_is_better=False,
#     )

#     trainer = Seq2SeqTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=valid_dataset,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#     )

#     # --- 5. Start Training ---
#     print("\n[5/5] Starting DANP training...")
#     try:
#         trainer.train()
#         print("\n🎉 DANP Training Complete!")
#         trainer.save_model()
#         tokenizer.save_pretrained(args.output_dir)
#         print(f"   Model saved to: {args.output_dir}")
#     except Exception as e:
#         print(f"\n❌ Training failed: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()








"""
P2G Training with DANP Data (or Baseline)
- Fine-tunes an mT5 model on a standard or augmented DANP dataset.
- Includes a --smoke_test_size flag for quick debugging runs.

Usage:
    # Full run
    python g_train_p2g_danp.py \
        --train_tsv ./p2g_final_datasets/train_danp_beam_plus_sampling_dedup.tsv \
        --valid_tsv /path/to/validation_phoneme_cleaned.tsv \
        --output_dir ./p2g_model_danp_combined

    # Smoke test on 100 samples
    python g_train_p2g_danp.py \
        --train_tsv ./p2g_final_datasets/train_danp_beam_plus_sampling_dedup.tsv \
        --valid_tsv /path/to/validation_phoneme_cleaned.tsv \
        --output_dir ./p2g_model_danp_smoke_test \
        --smoke_test_size 100
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
import argparse

warnings.filterwarnings('ignore')
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    parser = argparse.ArgumentParser(description="Train P2G model with DANP augmented data.")
    parser.add_argument("--train_tsv", type=str, required=True, help="Path to the training TSV (can be baseline or DANP).")
    parser.add_argument("--valid_tsv", type=str, required=True, help="Path to the validation TSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model.")
    parser.add_argument("--model_name", type=str, default="google/mt5-small", help="Pre-trained model to fine-tune.")
    
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_source_len", type=int, default=256)
    parser.add_argument("--max_target_len", type=int, default=256)
    
    # New Argument for quick testing
    parser.add_argument(
        "--smoke_test_size",
        type=int,
        default=None,
        help="If set, runs a quick test on a small subset of N samples. E.g., --smoke_test_size 100"
    )
    
    args = parser.parse_args()

    print("="*70)
    print(" "*18 + "P2G Training (DANP / Baseline)")
    print("="*70)
    print(f"\n📋 Configuration:")
    for arg, value in vars(args).items():
        print(f"   {arg}: {value}")

    # --- 1. Load Data ---
    print("\n[1/5] Loading datasets...")
    df_train = pd.read_csv(args.train_tsv, sep='\t')
    df_valid = pd.read_csv(args.valid_tsv, sep='\t')

    # --- SLICING LOGIC FOR SMOKE TEST ---
    if args.smoke_test_size:
        print(f"\n🔥🔥🔥 SMOKE TEST MODE ENABLED 🔥🔥🔥")
        print(f"   Using only {args.smoke_test_size} training samples and {max(10, args.smoke_test_size // 5)} validation samples.")
        df_train = df_train.head(args.smoke_test_size)
        df_valid = df_valid.head(max(10, args.smoke_test_size // 5))
    # --- END OF SLICING LOGIC ---

    df_train = df_train.dropna(subset=['phonemes', 'sentence'])
    df_valid = df_valid.dropna(subset=['phonemes', 'sentence'])

    print(f"   ✓ Train samples to be used: {len(df_train):,}")
    print(f"   ✓ Valid samples to be used: {len(df_valid):,}")

    # --- 2. Load Model & Tokenizer ---
    print("\n[2/5] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    print(f"   ✓ Model: {args.model_name} ({model.num_parameters() / 1e6:.1f}M params)")

    # --- 3. Prepare Datasets ---
    # print("\n[3/5] Preparing and tokenizing datasets...")
    # def preprocess_function(examples):
    #     model_inputs = tokenizer(
    #         examples['phonemes'], max_length=args.max_source_len, truncation=True
    #     )
    #     with tokenizer.as_target_tokenizer():
    #         labels = tokenizer(
    #             examples['sentence'], max_length=args.max_target_len, truncation=True
    #         )
    #     model_inputs["labels"] = labels["input_ids"]
    #     return model_inputs


    print("\n[3/5] Preparing and tokenizing datasets...")

    def preprocess_function(examples):
        # Dynamic padding - no padding during tokenization
        model_inputs = tokenizer(
            examples['phonemes'], 
            max_length=args.max_source_len, 
            truncation=True,
            padding=False  # Let data collator handle padding
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples['sentence'], 
                max_length=args.max_target_len, 
                truncation=True,
                padding=False  # Let data collator handle padding
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = Dataset.from_pandas(df_train[['phonemes', 'sentence']])
    valid_dataset = Dataset.from_pandas(df_valid[['phonemes', 'sentence']])

    train_dataset = train_dataset.map(
        preprocess_function, 
        batched=True, 
        desc="Tokenizing Train", 
        remove_columns=train_dataset.column_names
    )
    valid_dataset = valid_dataset.map(
        preprocess_function, 
        batched=True, 
        desc="Tokenizing Valid", 
        remove_columns=valid_dataset.column_names
    )

    # # --- 4. Setup Training ---
    # print("\n[4/5] Setting up training...")
    # data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir=args.output_dir,
    #     eval_strategy="steps",
    #     eval_steps=500,
    #     save_steps=500,
    #     logging_steps=50,
    #     learning_rate=args.lr,
    #     per_device_train_batch_size=args.batch_size,
    #     per_device_eval_batch_size=args.batch_size,
    #     gradient_accumulation_steps=args.grad_accum,
    #     num_train_epochs=args.epochs,
    #     weight_decay=0.01,
    #     warmup_ratio=0.1,
    #     save_total_limit=2,
    #     predict_with_generate=True,
    #     generation_max_length=args.max_target_len,
    #     fp16=torch.cuda.is_available(),
    #     logging_dir=f"{args.output_dir}/logs",
    #     report_to=[],
    #     load_best_model_at_end=True,
    #     metric_for_best_model="eval_loss",
    #     greater_is_better=False,
    # )



    print("\n[4/5] Setting up training...")

    # Calculate total steps for reference
    total_train_steps = (len(df_train) // args.batch_size // args.grad_accum) * args.epochs
    print(f"   📊 Total training steps: {total_train_steps}")
    print(f"   📊 Warmup steps (if using 50): {50}")
    print(f"   📊 Initial LR: {args.lr}")

    data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, 
    model=model,
    padding=True,  # Dynamic padding
    label_pad_token_id=-100
)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=50,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=50,  # Changed from warmup_ratio
        lr_scheduler_type="linear",  # Explicit scheduler
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{args.output_dir}/logs",
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # ADD THIS DIAGNOSTIC:
    print("\n🔍 Optimizer & Scheduler Diagnostic:")
    print(f"   Number of optimizers: {len(trainer.optimizer.param_groups)}")
    print(f"   Initial LR from optimizer: {trainer.optimizer.param_groups[0]['lr']}")
    if trainer.lr_scheduler is not None:
        print(f"   Scheduler type: {type(trainer.lr_scheduler).__name__}")
        print(f"   Scheduler last_epoch: {trainer.lr_scheduler.last_epoch}")
        # Try to get current LR
        current_lr = trainer.lr_scheduler.get_last_lr()[0]
        print(f"   Current LR from scheduler: {current_lr}")
    else:
        print("   ⚠️  NO SCHEDULER FOUND!")

    # --- Debug: Print scheduler info ---
    print(f"\n🔍 Scheduler check:")
    print(f"   LR scheduler type: {training_args.lr_scheduler_type}")
    print(f"   Initial LR: {training_args.learning_rate}")
    print(f"   Warmup steps: {training_args.warmup_steps}")

    # # --- 5. Start Training ---
    # print("\n[5/5] Starting training...")
    # try:
    #     trainer.train()
    #     print("\n🎉 Training Complete!")
    #     trainer.save_model()
    #     tokenizer.save_pretrained(args.output_dir)
    #     print(f"   Model saved to: {args.output_dir}")
    # except Exception as e:
    #     print(f"\n❌ Training failed: {e}")
    #     import traceback
    #     traceback.print_exc()





    print("\n[5/5] Starting training...")

    # ============ CRITICAL DEBUG CODE ============
    print("\n🔬 DEEP DIAGNOSTIC CHECK:")
    print("=" * 60)

    # Get a sample batch
    sample_dataloader = trainer.get_train_dataloader()
    sample_batch = next(iter(sample_dataloader))

    print(f"1️⃣ Batch shapes:")
    print(f"   input_ids: {sample_batch['input_ids'].shape}")
    print(f"   labels: {sample_batch['labels'].shape}")

    print(f"\n2️⃣ First sample in batch:")
    print(f"   Input (raw): {sample_batch['input_ids'][0][:20]}")
    print(f"   Labels (raw): {sample_batch['labels'][0][:20]}")

    print(f"\n3️⃣ Label statistics:")
    labels_tensor = sample_batch['labels']
    print(f"   Total label tokens: {labels_tensor.numel()}")
    print(f"   Number of -100 (ignored): {(labels_tensor == -100).sum().item()}")
    print(f"   Number of valid labels: {(labels_tensor != -100).sum().item()}")
    print(f"   ⚠️  Percentage ignored: {(labels_tensor == -100).sum().item() / labels_tensor.numel() * 100:.1f}%")

    print(f"\n4️⃣ Decoded examples:")
    for i in range(min(2, len(sample_batch['input_ids']))):
        input_text = tokenizer.decode(sample_batch['input_ids'][i], skip_special_tokens=True)
        # Decode labels, filtering out -100
        valid_labels = [l for l in sample_batch['labels'][i].tolist() if l != -100]
        label_text = tokenizer.decode(valid_labels, skip_special_tokens=True) if valid_labels else "[EMPTY]"
        
        print(f"   Sample {i+1}:")
        print(f"      Input:  {input_text[:100]}")
        print(f"      Target: {label_text[:100]}")

    print(f"\n5️⃣ Manual forward pass test:")
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    sample_batch = {k: v.to(model.device) for k, v in sample_batch.items()}

    with torch.no_grad():
        outputs = model(**sample_batch)
        print(f"   Loss: {outputs.loss.item() if outputs.loss is not None else 'None'}")
        print(f"   Loss is NaN: {torch.isnan(outputs.loss).item() if outputs.loss is not None else 'N/A'}")

    print("=" * 60)
    # ============ END DIAGNOSTIC CODE ============

    try:
        trainer.train()
        print("\n🎉 Training Complete!")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        print(f"   Model saved to: {args.output_dir}")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()