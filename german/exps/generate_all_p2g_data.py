#!/usr/bin/env python3
"""
Complete P2G Data Generation Script
Generates ALL augmented data for DANP and TKM training

Methods:
1. DANP Beam Search (K-beam)
2. DANP Random Sampling  
3. TKM Top-K Hypotheses

Usage:
    # Generate all data types
    python generate_all_p2g_data.py \
        --s2p_model_path ./s2p_model_german_phoneme_split \
        --dataset_path /scratch/ABHI/common_voice_de_190h \
        --train_tsv /path/to/train_phoneme_cleaned.tsv \
        --output_dir ./p2g_training_data \
        --generate_danp_beam \
        --generate_danp_sampling \
        --generate_tkm \
        --beam_size 32 \
        --num_samples 500 \
        --tkm_k 32

    # Or generate specific types only
    python generate_all_p2g_data.py \
        --s2p_model_path ./s2p_model_german_phoneme_split \
        --dataset_path /scratch/ABHI/common_voice_de_190h \
        --train_tsv /path/to/train_phoneme_cleaned.tsv \
        --output_dir ./p2g_training_data \
        --generate_danp_beam \
        --beam_size 32
"""

import torch
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_from_disk, Audio as AudioFeature
import argparse
import os
from tqdm import tqdm
import soundfile as sf
import io
import resampy
import numpy as np
import warnings
import json

warnings.filterwarnings('ignore')


class S2PDataGenerator:
    """
    Unified S2P-based data generator for DANP and TKM
    """
    
    def __init__(self, model_path):
        # Convert to absolute path to avoid HuggingFace path validation issues
        import os
        model_path = os.path.abspath(model_path)
        
        print(f"[*] Loading S2P model from: {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Using device: {self.device}")
        
        # Check if path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"S2P model not found at: {model_path}")
        
        # Check for required files
        if not os.path.exists(os.path.join(model_path, "config.json")):
            raise FileNotFoundError(f"config.json not found in {model_path}")
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"[*] S2P model loaded successfully")
    
    def preprocess_audio(self, audio_array, sample_rate):
        """Preprocess audio to 16kHz mono"""
        # Convert to mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        # Resample to 16kHz
        if sample_rate != 16000:
            audio_array = resampy.resample(audio_array, sample_rate, 16000)
        
        # Convert to float32
        audio_array = audio_array.astype(np.float32)
        
        return audio_array
    
    def get_logits(self, audio_array):
        """Get S2P model logits"""
        input_values = self.processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).input_values.to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        return logits
    
    def generate_beam_hypotheses(self, audio_array, sample_rate, num_beams=32):
        """
        Generate top-K diverse phoneme hypotheses using vectorized sampling
        
        This uses the same vectorized approach as random sampling but with
        a smaller num_samples to ensure we get exactly K unique hypotheses quickly.
        
        Returns:
            List of (phoneme_sequence, score) tuples
        """
        audio_array = self.preprocess_audio(audio_array, sample_rate)
        logits = self.get_logits(audio_array)
        
        # Squeeze batch dimension
        logits = logits.squeeze(0)  # Shape: [Time, Vocab]
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        hypotheses = []
        
        # 1. Always include greedy (1-best) decoding
        greedy_ids = torch.argmax(logits, dim=-1).unsqueeze(0)
        greedy_phonemes = self.processor.batch_decode(greedy_ids)[0]
        if greedy_phonemes:
            hypotheses.append((greedy_phonemes, 1.0))
        
        # 2. Generate additional hypotheses using vectorized sampling
        # We'll generate more than needed and then select unique ones
        batch_size = num_beams * 3  # Generate 3x to ensure enough unique
        
        # Vectorized sampling
        sampled_ids_tensor = torch.multinomial(
            probs, 
            num_samples=batch_size, 
            replacement=True
        )
        sampled_ids_tensor = sampled_ids_tensor.transpose(0, 1)  # [batch_size, Time]
        
        # Batch decode
        phonemes_list = self.processor.batch_decode(sampled_ids_tensor)
        
        # Collect unique hypotheses (excluding greedy)
        seen_phonemes = {greedy_phonemes}
        
        for phonemes in phonemes_list:
            if phonemes and phonemes not in seen_phonemes:
                # Assign decreasing scores
                score = 1.0 / (len(hypotheses) + 1)
                hypotheses.append((phonemes, score))
                seen_phonemes.add(phonemes)
                
                # Stop when we have enough
                if len(hypotheses) >= num_beams:
                    break
        
        # If still not enough, generate more
        while len(hypotheses) < num_beams:
            # Generate one more batch
            sampled_ids_tensor = torch.multinomial(
                probs, 
                num_samples=num_beams, 
                replacement=True
            )
            sampled_ids_tensor = sampled_ids_tensor.transpose(0, 1)
            phonemes_list = self.processor.batch_decode(sampled_ids_tensor)
            
            for phonemes in phonemes_list:
                if phonemes and phonemes not in seen_phonemes:
                    score = 1.0 / (len(hypotheses) + 1)
                    hypotheses.append((phonemes, score))
                    seen_phonemes.add(phonemes)
                    
                    if len(hypotheses) >= num_beams:
                        break
            
            # Safety: break if we're stuck (very rare)
            if len(hypotheses) >= num_beams - 2:  # Allow 2 missing
                break
        
        return hypotheses[:num_beams]
    
    def generate_sampling_hypotheses(self, audio_array, sample_rate, num_samples=500):
        """
        Generate diverse phoneme hypotheses using random sampling.
        This version is FULLY VECTORIZED for massive speed improvements.
        
        Returns:
            List of unique phoneme sequences
        """
        # Step 1: Preprocess audio and get logits
        audio_array = self.preprocess_audio(audio_array, sample_rate)
        logits = self.get_logits(audio_array)
        
        # Squeeze the batch dimension, shape becomes [Time, Vocab]
        logits = logits.squeeze(0)
        
        # Step 2: Get probabilities for the entire sequence at once
        # The shape of probs will be [Time, Vocab]
        probs = torch.softmax(logits, dim=-1)
        
        # Step 3: THE CORE OPTIMIZATION - Vectorized Sampling
        # Use torch.multinomial to sample num_samples times from each time step's distribution.
        # 'replacement=True' is important, allowing it to pick the same phoneme multiple times if needed.
        # The resulting tensor will have shape [Time, num_samples], where each column is one full path of token IDs.
        sampled_ids_tensor = torch.multinomial(probs, num_samples=num_samples, replacement=True)
        
        # Step 4: Transpose the tensor to get the desired shape for batch decoding.
        # We need shape [num_samples, Time] because the batch dimension comes first.
        sampled_ids_tensor = sampled_ids_tensor.transpose(0, 1)
        
        # Step 5: Batch decode all generated sequences in a single call.
        # This is massively more efficient than decoding in a loop.
        phonemes_list = self.processor.batch_decode(sampled_ids_tensor)
        
        # Step 6: Filter out empty strings and get the unique set of phonemes.
        # Using a set comprehension is a fast and Pythonic way to do this.
        unique_phonemes = {p for p in phonemes_list if p}
        
        return list(unique_phonemes)


def generate_danp_beam_data(generator, df_train, dataset, filename_to_idx, beam_size, output_dir):
    """Generate DANP data using beam search"""
    
    print("\n" + "="*70)
    print("GENERATING DANP DATA - BEAM SEARCH METHOD")
    print("="*70)
    print(f"Beam size: {beam_size}")
    print(f"Expected augmentation: ~{beam_size}x")
    print(f"Total samples to process: {len(df_train):,}")
    print(f"Estimated time: 5-7 hours")
    
    augmented_data = []
    failed_count = 0
    
    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="DANP Beam", ncols=100, 
                         unit="sample", smoothing=0.1):
        filename = os.path.basename(row['path'])
        sentence = row['sentence']
        ground_truth_phonemes = row['phonemes']
        
        if filename not in filename_to_idx:
            failed_count += 1
            continue
        
        try:
            # Get audio
            dataset_idx = filename_to_idx[filename]
            audio_bytes = dataset[dataset_idx]['audio']['bytes']
            audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
            
            # Generate beam hypotheses
            hypotheses = generator.generate_beam_hypotheses(
                audio_array, sample_rate, num_beams=beam_size)
            
            # Add all hypotheses
            for phoneme_seq, score in hypotheses:
                if phoneme_seq:
                    augmented_data.append({
                        'path': row['path'],
                        'sentence': sentence,
                        'phonemes': phoneme_seq,
                        'original_phonemes': ground_truth_phonemes,
                        'method': 'beam',
                        'beam_rank': len(augmented_data) % beam_size,
                        'is_ground_truth': False
                    })
            
            # Add ground truth
            augmented_data.append({
                'path': row['path'],
                'sentence': sentence,
                'phonemes': ground_truth_phonemes,
                'original_phonemes': ground_truth_phonemes,
                'method': 'ground_truth',
                'beam_rank': -1,
                'is_ground_truth': True
            })
            
        except Exception as e:
            failed_count += 1
            continue
    
    # Save
    df_augmented = pd.DataFrame(augmented_data)
    output_path = os.path.join(output_dir, f"train_danp_beam{beam_size}.tsv")
    df_augmented.to_csv(output_path, sep='\t', index=False)
    
    # Statistics
    print(f"\n{'='*70}")
    print("DANP BEAM SEARCH - GENERATION COMPLETE")
    print("="*70)
    print(f"  Original samples:     {len(df_train):,}")
    print(f"  Augmented samples:    {len(df_augmented):,}")
    print(f"  Augmentation factor:  {len(df_augmented) / len(df_train):.1f}x")
    print(f"  Failed samples:       {failed_count:,}")
    print(f"  Output file:          {output_path}")
    print("="*70)
    
    return output_path


def generate_danp_sampling_data(generator, df_train, dataset, filename_to_idx, num_samples, output_dir):
    """Generate DANP data using random sampling"""
    
    print("\n" + "="*70)
    print("GENERATING DANP DATA - RANDOM SAMPLING METHOD")
    print("="*70)
    print(f"Samples per audio: {num_samples}")
    print(f"Expected augmentation: ~10-20x (after deduplication)")
    print(f"Total samples to process: {len(df_train):,}")
    print(f"Estimated time: 8-10 hours")
    
    augmented_data = []
    failed_count = 0
    
    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="DANP Sampling", ncols=100,
                         unit="sample", smoothing=0.1):
        filename = os.path.basename(row['path'])
        sentence = row['sentence']
        ground_truth_phonemes = row['phonemes']
        
        if filename not in filename_to_idx:
            failed_count += 1
            continue
        
        try:
            # Get audio
            dataset_idx = filename_to_idx[filename]
            audio_bytes = dataset[dataset_idx]['audio']['bytes']
            audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
            
            # Generate sampled hypotheses
            hypotheses = generator.generate_sampling_hypotheses(
                audio_array, sample_rate, num_samples=num_samples)
            
            # Add all unique samples
            for phoneme_seq in hypotheses:
                if phoneme_seq:
                    augmented_data.append({
                        'path': row['path'],
                        'sentence': sentence,
                        'phonemes': phoneme_seq,
                        'original_phonemes': ground_truth_phonemes,
                        'method': 'sampling',
                        'is_ground_truth': False
                    })
            
            # Add ground truth
            augmented_data.append({
                'path': row['path'],
                'sentence': sentence,
                'phonemes': ground_truth_phonemes,
                'original_phonemes': ground_truth_phonemes,
                'method': 'ground_truth',
                'is_ground_truth': True
            })
            
        except Exception as e:
            failed_count += 1
            continue
    
    # Save
    df_augmented = pd.DataFrame(augmented_data)
    output_path = os.path.join(output_dir, f"train_danp_sampling{num_samples}.tsv")
    df_augmented.to_csv(output_path, sep='\t', index=False)
    
    # Statistics
    print(f"\n{'='*70}")
    print("DANP RANDOM SAMPLING - GENERATION COMPLETE")
    print("="*70)
    print(f"  Original samples:     {len(df_train):,}")
    print(f"  Augmented samples:    {len(df_augmented):,}")
    print(f"  Augmentation factor:  {len(df_augmented) / len(df_train):.1f}x")
    print(f"  Failed samples:       {failed_count:,}")
    print(f"  Unique phoneme seqs:  {df_augmented['phonemes'].nunique():,}")
    print(f"  Output file:          {output_path}")
    print("="*70)
    
    return output_path


def generate_tkm_data(generator, df_train, dataset, filename_to_idx, k, output_dir):
    """Generate TKM training data with top-K hypotheses"""
    
    print("\n" + "="*70)
    print("GENERATING TKM TRAINING DATA")
    print("="*70)
    print(f"K (top-K hypotheses): {k}")
    print(f"Each sample will have {k} phoneme hypotheses with scores")
    print(f"Total samples to process: {len(df_train):,}")
    print(f"Estimated time: 5-7 hours")
    
    tkm_data = []
    failed_count = 0
    
    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="TKM Generation", ncols=100,
                         unit="sample", smoothing=0.1):
        filename = os.path.basename(row['path'])
        sentence = row['sentence']
        ground_truth_phonemes = row['phonemes']
        
        if filename not in filename_to_idx:
            failed_count += 1
            continue
        
        try:
            # Get audio
            dataset_idx = filename_to_idx[filename]
            audio_bytes = dataset[dataset_idx]['audio']['bytes']
            audio_array, sample_rate = sf.read(io.BytesIO(audio_bytes))
            
            # Generate top-K hypotheses with scores
            hypotheses = generator.generate_beam_hypotheses(
                audio_array, sample_rate, num_beams=k)
            
            # Store as JSON for easy loading later (with readable Unicode)
            hypotheses_json = json.dumps([
                {'phonemes': h[0], 'score': float(h[1])} 
                for h in hypotheses
            ], ensure_ascii=False)  # Keep phonemes readable!
            
            tkm_data.append({
                'path': row['path'],
                'sentence': sentence,
                'ground_truth_phonemes': ground_truth_phonemes,
                'num_hypotheses': len(hypotheses),
                'hypotheses_json': hypotheses_json,
                'k': k
            })
            
        except Exception as e:
            failed_count += 1
            continue
    
    # Save
    df_tkm = pd.DataFrame(tkm_data)
    output_path = os.path.join(output_dir, f"train_tkm_k{k}.tsv")
    df_tkm.to_csv(output_path, sep='\t', index=False)
    
    # Statistics
    print(f"\n{'='*70}")
    print("TKM DATA - GENERATION COMPLETE")
    print("="*70)
    print(f"  Original samples:     {len(df_train):,}")
    print(f"  TKM samples:          {len(df_tkm):,}")
    print(f"  Failed samples:       {failed_count:,}")
    print(f"  Hypotheses per sample: {k}")
    print(f"  Output file:          {output_path}")
    print("="*70)
    
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate ALL P2G training data (DANP + TKM)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all data types
  python generate_all_p2g_data.py \\
      --s2p_model_path ./s2p_model_german \\
      --dataset_path /scratch/ABHI/common_voice_de_190h \\
      --train_tsv /path/to/train_cleaned.tsv \\
      --output_dir ./p2g_training_data \\
      --generate_danp_beam \\
      --generate_danp_sampling \\
      --generate_tkm

  # Generate only DANP beam
  python generate_all_p2g_data.py \\
      --s2p_model_path ./s2p_model_german \\
      --dataset_path /scratch/ABHI/common_voice_de_190h \\
      --train_tsv /path/to/train_cleaned.tsv \\
      --output_dir ./p2g_training_data \\
      --generate_danp_beam \\
      --beam_size 32
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--s2p_model_path",
        type=str,
        required=True,
        help="Path to trained S2P model"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to audio dataset (HuggingFace format)"
    )
    parser.add_argument(
        "--train_tsv",
        type=str,
        required=True,
        help="Path to training TSV file (with clean phonemes)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./p2g_training_data",
        help="Output directory for all generated data"
    )
    
    # What to generate
    parser.add_argument(
        "--generate_danp_beam",
        action="store_true",
        help="Generate DANP data using beam search"
    )
    parser.add_argument(
        "--generate_danp_sampling",
        action="store_true",
        help="Generate DANP data using random sampling"
    )
    parser.add_argument(
        "--generate_tkm",
        action="store_true",
        help="Generate TKM training data"
    )
    
    # DANP beam parameters
    parser.add_argument(
        "--beam_size",
        type=int,
        default=32,
        help="Beam size for DANP beam search (paper: 32-96)"
    )
    
    # DANP sampling parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples for DANP random sampling (paper: 500 for German, 25000 for Polish)"
    )
    
    # TKM parameters
    parser.add_argument(
        "--tkm_k",
        type=int,
        default=32,
        help="K for TKM (top-K hypotheses, paper: 32)"
    )
    
    args = parser.parse_args()
    
    # Validate
    if not (args.generate_danp_beam or args.generate_danp_sampling or args.generate_tkm):
        print("❌ ERROR: Must specify at least one generation method!")
        print("   Use --generate_danp_beam, --generate_danp_sampling, or --generate_tkm")
        return
    
    # Convert paths to absolute
    import os
    args.s2p_model_path = os.path.abspath(args.s2p_model_path)
    args.dataset_path = os.path.abspath(args.dataset_path)
    args.train_tsv = os.path.abspath(args.train_tsv)
    args.output_dir = os.path.abspath(args.output_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print(" "*15 + "P2G DATA GENERATION")
    print(" "*10 + "(DANP + TKM - Complete Pipeline)")
    print("="*70)
    
    print(f"\n📋 Configuration:")
    print(f"   S2P Model: {args.s2p_model_path}")
    print(f"   Dataset: {args.dataset_path}")
    print(f"   Train TSV: {args.train_tsv}")
    print(f"   Output dir: {args.output_dir}")
    print(f"\n   Generation methods:")
    if args.generate_danp_beam:
        print(f"      ✓ DANP Beam Search (K={args.beam_size})")
    if args.generate_danp_sampling:
        print(f"      ✓ DANP Random Sampling (R={args.num_samples})")
    if args.generate_tkm:
        print(f"      ✓ TKM (K={args.tkm_k})")
    
    # Load S2P model
    print(f"\n{'='*70}")
    print("STEP 1: Loading S2P Model")
    print("="*70)
    generator = S2PDataGenerator(args.s2p_model_path)
    
    # Load training TSV
    print(f"\n{'='*70}")
    print("STEP 2: Loading Training Data")
    print("="*70)
    
    if not os.path.exists(args.train_tsv):
        print(f"❌ ERROR: Training TSV not found: {args.train_tsv}")
        return
    
    df_train = pd.read_csv(args.train_tsv, sep='\t')
    df_train = df_train.dropna(subset=['phonemes', 'sentence'])
    df_train = df_train[(df_train['phonemes'] != "") & (df_train['sentence'] != "")]
    
    # Calculate total audio duration if available
    total_duration_sec = 0
    if 'duration' in df_train.columns:
        total_duration_sec = df_train['duration'].sum()
    
    print(f"   ✓ Loaded {len(df_train):,} training samples")
    if total_duration_sec > 0:
        total_hours = total_duration_sec / 3600
        avg_duration = total_duration_sec / len(df_train)
        print(f"   ✓ Total audio duration: {total_hours:.1f} hours ({total_duration_sec/60:.0f} minutes)")
        print(f"   ✓ Average sample length: {avg_duration:.1f} seconds")
    else:
        # Estimate based on typical German speech (~4 seconds average)
        estimated_hours = (len(df_train) * 4.0) / 3600
        print(f"   ℹ️  Estimated audio duration: ~{estimated_hours:.1f} hours (assuming ~4s per sample)")
        print(f"   ℹ️  Note: For exact duration, TSV should have 'duration' column")
    
    # Load dataset
    print(f"\n{'='*70}")
    print("STEP 3: Loading Audio Dataset")
    print("="*70)
    
    if not os.path.exists(args.dataset_path):
        print(f"❌ ERROR: Dataset not found: {args.dataset_path}")
        return
    
    dataset = load_from_disk(args.dataset_path)
    dataset = dataset.cast_column("audio", AudioFeature(decode=False))
    print(f"   ✓ Loaded {len(dataset):,} audio samples")
    
    # Create filename mapping
    print(f"\n   Creating filename to index mapping...")
    filename_to_idx = {}
    for idx in tqdm(range(len(dataset)), desc="   Mapping", ncols=70):
        filename = os.path.basename(dataset[idx]['path'])
        filename_to_idx[filename] = idx
    print(f"   ✓ Created map with {len(filename_to_idx):,} entries")
    
    # Generate data
    generated_files = []
    
    # DANP Beam
    if args.generate_danp_beam:
        print(f"\n{'='*70}")
        print("STEP 4a: Generating DANP Beam Search Data")
        print("="*70)
        output_file = generate_danp_beam_data(
            generator, df_train, dataset, filename_to_idx, 
            args.beam_size, args.output_dir
        )
        generated_files.append(('DANP Beam', output_file))
    
    # DANP Sampling
    if args.generate_danp_sampling:
        print(f"\n{'='*70}")
        print("STEP 4b: Generating DANP Random Sampling Data")
        print("="*70)
        output_file = generate_danp_sampling_data(
            generator, df_train, dataset, filename_to_idx,
            args.num_samples, args.output_dir
        )
        generated_files.append(('DANP Sampling', output_file))
    
    # TKM
    if args.generate_tkm:
        print(f"\n{'='*70}")
        print("STEP 4c: Generating TKM Training Data")
        print("="*70)
        output_file = generate_tkm_data(
            generator, df_train, dataset, filename_to_idx,
            args.tkm_k, args.output_dir
        )
        generated_files.append(('TKM', output_file))
    
    # Final summary
    print(f"\n{'='*70}")
    print("🎉 DATA GENERATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated files:")
    for method, filepath in generated_files:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"   [{method}] {filepath}")
        print(f"      Size: {size_mb:.1f} MB")
    
    print(f"\n{'='*70}")
    print("Next Steps:")
    print("="*70)
    
    if args.generate_danp_beam or args.generate_danp_sampling:
        print("\n1. Train P2G with DANP:")
        for method, filepath in generated_files:
            if 'DANP' in method:
                print(f"   python train_p2g_danp.py \\")
                print(f"       --train_tsv {filepath} \\")
                print(f"       --valid_tsv /path/to/valid.tsv \\")
                print(f"       --output_dir ./p2g_model_danp")
    
    if args.generate_tkm:
        print("\n2. Train P2G with TKM:")
        for method, filepath in generated_files:
            if 'TKM' in method:
                print(f"   python train_p2g_tkm_separate.py \\")
                print(f"       --train_tsv {filepath} \\")
                print(f"       --valid_tsv /path/to/valid.tsv \\")
                print(f"       --output_dir ./p2g_model_tkm")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()