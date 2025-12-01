# #v1 with full data

# #!/usr/bin/env python3
# """
# P2G Training with TKM Data (Final, Robust Version)
# - Fine-tunes an mT5 model using the Top-K Marginalized (TKM) objective.
# - Uses a custom TKMDataset for the structured training data.
# - Uses a standard Hugging Face Dataset for the simple validation data.
# - Uses a custom TKMTrainer to implement the marginalized loss ONLY for training.

# Usage:
#     python h_train_p2g_tkm.py \
#         --train_tsv ./p2g_final_datasets/polish_tkm_k32.tsv \
#         --valid_tsv ./polish_phonemized/dev_5h_phoneme.tsv \
#         --output_dir ./p2g_model_polish_tkm_final \
#         --model_name google/mt5-small \
#         --k_train 8 \
#         --grad_accum 32
# """

# import torch
# import pandas as pd
# from datasets import Dataset as HfDataset
# from torch.utils.data import Dataset as TorchDataset
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
# import random

# warnings.filterwarnings('ignore')
# os.environ["WANDB_DISABLED"] = "true"
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# # --- Custom Dataset for TKM Training Data ---
# class TKMDataset(TorchDataset):
#     def __init__(self, tsv_path, tokenizer, max_source_len, max_target_len, k_train):
#         print(f"[*] Loading TKM training data from {tsv_path}")
#         self.df = pd.read_csv(tsv_path, sep='\t')
#         self.tokenizer = tokenizer
#         self.max_source_len = max_source_len
#         self.max_target_len = max_target_len
#         self.k_train = k_train
#         print(f"    ✓ Loaded {len(self.df)} samples.")
#         print(f"    ✓ Will randomly sample {k_train} hypotheses per sample during training.")

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
        
#         # Parse the JSON string of hypotheses
#         hypotheses = json.loads(row['hypotheses_json'])
        
#         # Randomized TKM: randomly sample k_train from the pool
#         if len(hypotheses) > self.k_train:
#             selected_hypotheses = random.sample(hypotheses, self.k_train)
#         else:
#             selected_hypotheses = hypotheses
        
#         phoneme_texts = [h['phonemes'] for h in selected_hypotheses]
        
#         # Use heuristic scores from file, convert to probabilities
#         scores = torch.tensor([h['score'] for h in selected_hypotheses], dtype=torch.float32)
#         s2p_probs = torch.nn.functional.softmax(scores, dim=0)

#         # Tokenize source phonemes (a batch of K)
#         source_tokenized = self.tokenizer(
#             phoneme_texts,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_source_len,
#             return_tensors="pt"
#         )
        
#         # Tokenize the single target sentence
#         with self.tokenizer.as_target_tokenizer():
#             target_tokenized = self.tokenizer(
#                 row['sentence'],
#                 padding="max_length",
#                 truncation=True,
#                 max_length=self.max_target_len,
#                 return_tensors="pt"
#             )

#         return {
#             "input_ids": source_tokenized.input_ids,
#             "attention_mask": source_tokenized.attention_mask,
#             "labels": target_tokenized.input_ids.squeeze(0),
#             "decoder_attention_mask": target_tokenized.attention_mask.squeeze(0),
#             "s2p_probs": s2p_probs
#         }

# # --- Custom Trainer for Marginalized Loss ---
# class TKMTrainer(Seq2SeqTrainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         # This custom loss logic is only used during TRAINING.
#         # The 's2p_probs' key will only exist in the training dataloader's output.
#         if "s2p_probs" in inputs:
#             s2p_probs = inputs.pop("s2p_probs").squeeze(0) # Shape: [K_train]
            
#             # Run the model once with a batch of K hypotheses
#             outputs = model(**inputs)
#             logits = outputs.get("logits") # Shape: [K_train, target_len, vocab_size]
            
#             # Reshape for cross_entropy
#             logits_flat = logits.view(-1, model.config.vocab_size)
            
#             # Repeat labels K_train times to match logits
#             labels = inputs.get("labels").repeat(len(s2p_probs), 1) # Shape: [K_train, target_len]
#             labels_flat = labels.view(-1)

#             # Calculate per-token loss without reduction
#             loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
#             per_token_loss = loss_fct(logits_flat, labels_flat)
            
#             # Reshape and calculate per-sequence loss
#             per_token_loss = per_token_loss.view(len(s2p_probs), -1)
#             per_sequence_loss = per_token_loss.mean(dim=1) # Shape: [K_train]
            
#             # ** THE MARGINALIZATION STEP **
#             marginalized_loss = torch.sum(s2p_probs * per_sequence_loss)
            
#             return (marginalized_loss, outputs) if return_outputs else marginalized_loss
#         else:
#             # For EVALUATION, the dataloader provides standard inputs.
#             # Use the default loss calculation from the parent Seq2SeqTrainer class.
#             return super().compute_loss(model, inputs, return_outputs)

# def main():
#     parser = argparse.ArgumentParser(description="Train P2G model with TKM.")
#     parser.add_argument("--train_tsv", type=str, required=True, help="Path to the TKM training TSV.")
#     parser.add_argument("--valid_tsv", type=str, required=True, help="Path to the standard, clean validation TSV.")
#     parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model.")
#     parser.add_argument("--model_name", type=str, default="google/mt5-small")
#     parser.add_argument("--k_train", type=int, default=8, help="Number of hypotheses to sample for training (n in paper).")
    
#     # Hyperparameters
#     parser.add_argument("--batch_size", type=int, default=1, help="TKM requires per-device batch size of 1.")
#     parser.add_argument("--grad_accum", type=int, default=32)
#     parser.add_argument("--epochs", type=int, default=10)
#     parser.add_argument("--lr", type=float, default=3e-4)
#     parser.add_argument("--max_source_len", type=int, default=256)
#     parser.add_argument("--max_target_len", type=int, default=256)
    
#     args = parser.parse_args()

#     print("="*70)
#     print(" "*20 + "P2G Training with TKM")
#     print("="*70)

#     # --- 1. Load Model & Tokenizer ---
#     print("\n[1/4] Loading model and tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name, legacy=False)
#     model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

#     # --- 2. Create Datasets ---
#     print("\n[2/4] Creating TKM and Validation datasets...")
#     # Custom dataset for the structured TKM training data
#     train_dataset = TKMDataset(args.train_tsv, tokenizer, args.max_source_len, args.max_target_len, args.k_train)
    
#     # Standard Hugging Face dataset for the simple validation data
#     df_valid = pd.read_csv(args.valid_tsv, sep='\t')
#     df_valid = df_valid.dropna(subset=['phonemes', 'sentence'])
#     valid_dataset = HfDataset.from_pandas(df_valid[['phonemes', 'sentence']])

#     def preprocess_validation(examples):
#         model_inputs = tokenizer(examples['phonemes'], max_length=args.max_source_len, truncation=True)
#         with tokenizer.as_target_tokenizer():
#             labels = tokenizer(examples['sentence'], max_length=args.max_target_len, truncation=True)
#         model_inputs["labels"] = labels["input_ids"]
#         return model_inputs

#     valid_dataset = valid_dataset.map(preprocess_validation, batched=True, desc="Tokenizing Validation")
    
#     print(f"   ✓ Train dataset (TKM): {len(train_dataset)} samples")
#     print(f"   ✓ Valid dataset (Standard): {len(valid_dataset)} samples")

#     # --- 3. Setup Training ---
#     print("\n[3/4] Setting up TKM training...")
    
#     if args.batch_size != 1:
#         print("⚠️ WARNING: TKM requires a per-device batch size of 1. Overriding to 1.")
#         args.batch_size = 1
        
#     training_args = Seq2SeqTrainingArguments(
#         output_dir=args.output_dir,
#         eval_strategy="steps",
#         eval_steps=500,
#         save_steps=500,
#         logging_steps=50,
#         learning_rate=args.lr,
#         per_device_train_batch_size=args.batch_size,
#         per_device_eval_batch_size=8, # Can use a larger batch for standard validation
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
#         remove_unused_columns=False,
#     )

#     trainer = TKMTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=valid_dataset,
#         tokenizer=tokenizer,
#         data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
#     )
    
#     # --- 4. Start Training ---
#     print("\n[4/4] Starting TKM training...")
#     try:
#         trainer.train()
#         print("\n🎉 TKM Training Complete!")
#         trainer.save_model()
#         tokenizer.save_pretrained(args.output_dir)
#         print(f"   Model saved to: {args.output_dir}")
#     except Exception as e:
#         print(f"\n❌ Training failed: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()








#v2
#small data
#!/usr/bin/env python3
"""
P2G Training with TKM Data (Final, Robust Version)
- Fine-tunes an mT5 model using the Top-K Marginalized (TKM) objective.
- Includes a --smoke_test_size flag for quick debugging runs.

Usage:
    # Full run
    python h_train_p2g_tkm.py \
        --train_tsv ./p2g_final_datasets/train_tkm_k32.tsv \
        --valid_tsv /path/to/validation_phoneme_cleaned.tsv \
        --output_dir ./p2g_model_tkm

    # Smoke test on 100 samples
    python h_train_p2g_tkm.py \
        --train_tsv ./p2g_final_datasets/train_tkm_k32.tsv \
        --valid_tsv /path/to/validation_phoneme_cleaned.tsv \
        --output_dir ./p2g_model_tkm_smoke_test \
        --smoke_test_size 100

    python3 train_p2g_tkm.py \
    --train_tsv /home/abhinav.pm/ABHI/SAL2/german/130_hours_exps/p2g_training_20_data/train_tkm_k32.tsv \
    --valid_tsv /home/abhinav.pm/ABHI/SAL2/german/130_hours_exps/phonemized/validation_5_phoneme.tsv \
    --output_dir ./p2g_model_tkm_smoke_test \
    --smoke_test_size 5000

"""

import torch
import pandas as pd
from datasets import Dataset as HfDataset
from torch.utils.data import Dataset as TorchDataset
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
import random

warnings.filterwarnings('ignore')
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- Custom Dataset for TKM Training Data ---
class TKMDataset(TorchDataset):
    def __init__(self, data_input, tokenizer, max_source_len, max_target_len, k_train):
        if isinstance(data_input, str):
            print(f"[*] Loading TKM training data from {data_input}")
            self.df = pd.read_csv(data_input, sep='\t')
        elif isinstance(data_input, pd.DataFrame):
            self.df = data_input
        else:
            raise TypeError("data_input must be a file path or a pandas DataFrame")

        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.k_train = k_train
        print(f"    ✓ Loaded {len(self.df)} samples for TKMDataset.")
        print(f"    ✓ Will randomly sample up to {k_train} hypotheses per sample.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        hypotheses = json.loads(row['hypotheses_json'])
        
        if len(hypotheses) > self.k_train:
            selected_hypotheses = random.sample(hypotheses, self.k_train)
        else:
            selected_hypotheses = hypotheses
        
        phoneme_texts = [h['phonemes'] for h in selected_hypotheses]
        scores = torch.tensor([h['score'] for h in selected_hypotheses], dtype=torch.float32)
        s2p_probs = torch.nn.functional.softmax(scores, dim=0)

        source_tokenized = self.tokenizer(
            phoneme_texts, padding="max_length", truncation=True, max_length=self.max_source_len, return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer():
            target_tokenized = self.tokenizer(
                row['sentence'], padding="max_length", truncation=True, max_length=self.max_target_len, return_tensors="pt"
            )

        return {
            "input_ids": source_tokenized.input_ids,
            "attention_mask": source_tokenized.attention_mask,
            "labels": target_tokenized.input_ids.squeeze(0),
            "s2p_probs": s2p_probs
        }

# --- Custom Trainer for Marginalized Loss ---
# class TKMTrainer(Seq2SeqTrainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         if "s2p_probs" in inputs:
#             s2p_probs = inputs.pop("s2p_probs").squeeze(0)
#             outputs = model(**inputs)
#             logits = outputs.get("logits")
#             logits_flat = logits.view(-1, model.config.vocab_size)
            
#             labels = inputs.get("labels").repeat(len(s2p_probs), 1)
#             labels_flat = labels.view(-1)
            
#             loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
#             per_token_loss = loss_fct(logits_flat, labels_flat)
            
#             per_token_loss = per_token_loss.view(len(s2p_probs), -1)
#             per_sequence_loss = per_token_loss.mean(dim=1)
            
#             marginalized_loss = torch.sum(s2p_probs * per_sequence_loss)
            
#             return (marginalized_loss, outputs) if return_outputs else marginalized_loss
#         else:
#             return super().compute_loss(model, inputs, return_outputs)



# --- Custom Trainer for Marginalized Loss (FINAL CORRECTED VERSION) ---
class TKMTrainer(Seq2SeqTrainer):
    # Fix 1: Add **kwargs to accept extra arguments from the trainer loop
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if "s2p_probs" in inputs:
            s2p_probs = inputs.pop("s2p_probs")

            # Fix 2: Reshape the inputs to be 2D before passing to the model
            # The dataloader adds a batch dimension of 1. We need to remove it.
            # E.g., from [1, 8, 256] to [8, 256]
            if inputs["input_ids"].shape[0] == 1:
                inputs["input_ids"] = inputs["input_ids"].squeeze(0)
                inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)
                s2p_probs = s2p_probs.squeeze(0)

            outputs = model(**inputs)
            logits = outputs.get("logits")
            logits_flat = logits.view(-1, model.config.vocab_size)
            
            # The labels are 2D: [1, target_len]. We need to repeat them K times.
            labels = inputs.get("labels").repeat(len(s2p_probs), 1)
            labels_flat = labels.view(-1)
            
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            per_token_loss = loss_fct(logits_flat, labels_flat)
            
            per_token_loss = per_token_loss.view(len(s2p_probs), -1)
            # Make the mean calculation robust to padding
            per_sequence_loss = per_token_loss.sum(dim=1) / (labels != -100).sum(dim=1).add(1e-8)
            
            marginalized_loss = torch.sum(s2p_probs * per_sequence_loss)
            
            return (marginalized_loss, outputs) if return_outputs else marginalized_loss
        else:
            # For evaluation, pass the extra arguments to the parent class
            return super().compute_loss(model, inputs, return_outputs, **kwargs)
def main():
    parser = argparse.ArgumentParser(description="Train P2G model with TKM.")
    parser.add_argument("--train_tsv", type=str, required=True, help="Path to the TKM training TSV.")
    parser.add_argument("--valid_tsv", type=str, required=True, help="Path to the standard, clean validation TSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained model.")
    parser.add_argument("--model_name", type=str, default="google/mt5-small")
    parser.add_argument("--k_train", type=int, default=8, help="Number of hypotheses to sample for training (n in paper).")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=1, help="TKM requires per-device batch size of 1.")
    parser.add_argument("--grad_accum", type=int, default=32)
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
    print(" "*20 + "P2G Training with TKM")
    print("="*70)

    # --- 1. Load Model & Tokenizer ---
    print("\n[1/4] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    # --- 2. Create Datasets ---
    print("\n[2/4] Creating TKM and Validation datasets...")

    # Load dataframes first
    df_train_tkm = pd.read_csv(args.train_tsv, sep='\t')
    df_valid_clean = pd.read_csv(args.valid_tsv, sep='\t')

    # --- SLICING LOGIC FOR SMOKE TEST ---
    if args.smoke_test_size:
        print(f"\n🔥🔥🔥 SMOKE TEST MODE ENABLED 🔥🔥🔥")
        print(f"   Using only {args.smoke_test_size} training samples and {max(10, args.smoke_test_size // 5)} validation samples.")
        df_train_tkm = df_train_tkm.head(args.smoke_test_size)
        df_valid_clean = df_valid_clean.head(max(10, args.smoke_test_size // 5))
    # --- END OF SLICING LOGIC ---
    
    # Custom dataset for the structured TKM training data
    train_dataset = TKMDataset(df_train_tkm, tokenizer, args.max_source_len, args.max_target_len, args.k_train)
    
    # Standard Hugging Face dataset for the simple validation data
    df_valid_clean = df_valid_clean.dropna(subset=['phonemes', 'sentence'])
    valid_dataset = HfDataset.from_pandas(df_valid_clean[['phonemes', 'sentence']])

    def preprocess_validation(examples):
        model_inputs = tokenizer(examples['phonemes'], max_length=args.max_source_len, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples['sentence'], max_length=args.max_target_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    valid_dataset = valid_dataset.map(preprocess_validation, batched=True, desc="Tokenizing Validation", remove_columns=valid_dataset.column_names)
    
    print(f"   ✓ Train dataset (TKM): {len(train_dataset)} samples")
    print(f"   ✓ Valid dataset (Standard): {len(valid_dataset)} samples")

    # --- 3. Setup Training ---
    print("\n[3/4] Setting up TKM training...")
    
    if args.batch_size != 1:
        print("⚠️ WARNING: TKM requires a per-device batch size of 1. Overriding to 1.")
        args.batch_size = 1
        
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=50,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8, # Can use a larger batch for standard validation
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        fp16=torch.cuda.is_available(),
        logging_dir=f"{args.output_dir}/logs",
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )

    trainer = TKMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)
    )
    
    # --- 4. Start Training ---
    print("\n[4/4] Starting TKM training...")
    try:
        trainer.train()
        print("\n🎉 TKM Training Complete!")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        print(f"   Model saved to: {args.output_dir}")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()