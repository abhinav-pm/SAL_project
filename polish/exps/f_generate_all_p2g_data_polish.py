#!/usr/bin/env python3
"""
Complete P2G Data Generation Script for POLISH
(Optimized for Direct Disk Loading & Incremental Saving)

Generates augmented data for:
1. DANP Beam Search
2. DANP Random Sampling
3. TKM (Top-K Hypotheses)

Usage:
    python generate_all_p2g_polish.py \
        --s2p_model_path ./s2p_model_polish_phoneme_v1 \
        --clips_dir /scratch/priyanka/common_voice_polish23/cv-corpus-23.0-2025-09-05/pl/clips \
        --train_tsv ./phonemized/train_130_phoneme.tsv \
        --output_dir ./p2g_training_polish_130h \
        --generate_danp_sampling \
        --num_samples 500
"""

import torch
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import argparse
import os
from tqdm import tqdm
import librosa
import numpy as np
import warnings
import json
import resampy

warnings.filterwarnings('ignore')


class S2PDataGenerator:
    def __init__(self, model_path):
        import os
        model_path = os.path.abspath(model_path)
        print(f"[*] Loading S2P model from: {model_path}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Using device: {self.device}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"S2P model not found at: {model_path}")
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print(f"[*] S2P model loaded successfully")
    
    def preprocess_audio(self, audio_array, sample_rate):
        # Ensure mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        # Resample to 16kHz if needed (though librosa usually handles this on load)
        if sample_rate != 16000:
            audio_array = resampy.resample(audio_array, sample_rate, 16000)
            
        audio_array = audio_array.astype(np.float32)
        return audio_array
    
    def get_logits(self, audio_array):
        input_values = self.processor(
            audio_array, sampling_rate=16000, return_tensors="pt", padding=True
        ).input_values.to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_values).logits
        return logits
    
    def generate_beam_hypotheses(self, audio_array, sample_rate, num_beams=32):
        audio_array = self.preprocess_audio(audio_array, sample_rate)
        logits = self.get_logits(audio_array)
        logits = logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        hypotheses = []
        
        greedy_ids = torch.argmax(logits, dim=-1).unsqueeze(0)
        greedy_phonemes = self.processor.batch_decode(greedy_ids)[0]
        if greedy_phonemes:
            hypotheses.append((greedy_phonemes, 1.0))
        
        batch_size = num_beams * 3
        sampled_ids_tensor = torch.multinomial(probs, num_samples=batch_size, replacement=True)
        sampled_ids_tensor = sampled_ids_tensor.transpose(0, 1)
        phonemes_list = self.processor.batch_decode(sampled_ids_tensor)
        seen_phonemes = {greedy_phonemes}
        
        for phonemes in phonemes_list:
            if phonemes and phonemes not in seen_phonemes:
                score = 1.0 / (len(hypotheses) + 1)
                hypotheses.append((phonemes, score))
                seen_phonemes.add(phonemes)
                if len(hypotheses) >= num_beams:
                    break
        
        # Fill up if needed
        while len(hypotheses) < num_beams:
            sampled_ids_tensor = torch.multinomial(probs, num_samples=num_beams, replacement=True)
            sampled_ids_tensor = sampled_ids_tensor.transpose(0, 1)
            phonemes_list = self.processor.batch_decode(sampled_ids_tensor)
            for phonemes in phonemes_list:
                if phonemes and phonemes not in seen_phonemes:
                    score = 1.0 / (len(hypotheses) + 1)
                    hypotheses.append((phonemes, score))
                    seen_phonemes.add(phonemes)
                    if len(hypotheses) >= num_beams:
                        break
            if len(hypotheses) >= num_beams - 2:
                break
        return hypotheses[:num_beams]
    
    def generate_sampling_hypotheses(self, audio_array, sample_rate, num_samples=500):
        audio_array = self.preprocess_audio(audio_array, sample_rate)
        logits = self.get_logits(audio_array)
        logits = logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        
        # Vectorized sampling
        sampled_ids_tensor = torch.multinomial(probs, num_samples=num_samples, replacement=True)
        sampled_ids_tensor = sampled_ids_tensor.transpose(0, 1)
        
        phonemes_list = self.processor.batch_decode(sampled_ids_tensor)
        unique_phonemes = {p for p in phonemes_list if p}
        return list(unique_phonemes)


def generate_danp_beam_data(generator, df_train, clips_dir, beam_size, output_dir):
    print("\n" + "="*70)
    print("GENERATING DANP DATA - BEAM SEARCH METHOD (INCREMENTAL SAVING)")
    print("="*70)
    
    output_path = os.path.join(output_dir, f"train_danp_beam{beam_size}.tsv")
    if os.path.exists(output_path):
        os.remove(output_path)
        
    augmented_data_batch = []
    failed_count = 0
    total_saved = 0
    SAVE_EVERY = 200
    
    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="DANP Beam"):
        filename = os.path.basename(row['path'])
        audio_path = os.path.join(clips_dir, filename)
        
        if not os.path.exists(audio_path):
            failed_count += 1
            continue
        
        try:
            # Load directly from disk (handles MP3)
            audio_array, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            
            hypotheses = generator.generate_beam_hypotheses(audio_array, sample_rate, num_beams=beam_size)
            
            for phoneme_seq, score in hypotheses:
                if phoneme_seq:
                    augmented_data_batch.append({
                        'path': row['path'],
                        'sentence': row['sentence'],
                        'phonemes': phoneme_seq,
                        'original_phonemes': row['phonemes'],
                        'method': 'beam',
                        'beam_rank': len(augmented_data_batch) % beam_size,
                        'is_ground_truth': False
                    })
            
            augmented_data_batch.append({
                'path': row['path'],
                'sentence': row['sentence'],
                'phonemes': row['phonemes'],
                'original_phonemes': row['phonemes'],
                'method': 'ground_truth',
                'beam_rank': -1,
                'is_ground_truth': True
            })
            
            # Incremental Save
            if (idx + 1) % SAVE_EVERY == 0:
                df_chunk = pd.DataFrame(augmented_data_batch)
                write_header = not os.path.exists(output_path)
                df_chunk.to_csv(output_path, sep='\t', mode='a', index=False, header=write_header)
                total_saved += len(df_chunk)
                augmented_data_batch = []
                
        except Exception as e:
            failed_count += 1
            continue
            
    if augmented_data_batch:
        df_chunk = pd.DataFrame(augmented_data_batch)
        write_header = not os.path.exists(output_path)
        df_chunk.to_csv(output_path, sep='\t', mode='a', index=False, header=write_header)
        total_saved += len(df_chunk)

    print(f"Done. Saved to {output_path} (Rows: {total_saved})")
    return output_path


def generate_danp_sampling_data(generator, df_train, clips_dir, num_samples, output_dir):
    print("\n" + "="*70)
    print("GENERATING DANP DATA - RANDOM SAMPLING METHOD (INCREMENTAL SAVING)")
    print("="*70)
    
    output_path = os.path.join(output_dir, f"train_danp_sampling{num_samples}.tsv")
    if os.path.exists(output_path):
        os.remove(output_path)
    
    augmented_data_batch = []
    failed_count = 0
    total_saved = 0
    SAVE_EVERY = 100 
    
    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="DANP Sampling"):
        filename = os.path.basename(row['path'])
        audio_path = os.path.join(clips_dir, filename)
        
        if not os.path.exists(audio_path):
            failed_count += 1
            continue
        
        try:
            # Load directly from disk
            audio_array, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            
            hypotheses = generator.generate_sampling_hypotheses(audio_array, sample_rate, num_samples=num_samples)
            
            for phoneme_seq in hypotheses:
                if phoneme_seq:
                    augmented_data_batch.append({
                        'path': row['path'],
                        'sentence': row['sentence'],
                        'phonemes': phoneme_seq,
                        'original_phonemes': row['phonemes'],
                        'method': 'sampling',
                        'is_ground_truth': False
                    })
            
            augmented_data_batch.append({
                'path': row['path'],
                'sentence': row['sentence'],
                'phonemes': row['phonemes'],
                'original_phonemes': row['phonemes'],
                'method': 'ground_truth',
                'is_ground_truth': True
            })
            
            if (idx + 1) % SAVE_EVERY == 0:
                df_chunk = pd.DataFrame(augmented_data_batch)
                write_header = not os.path.exists(output_path)
                df_chunk.to_csv(output_path, sep='\t', mode='a', index=False, header=write_header)
                total_saved += len(df_chunk)
                augmented_data_batch = []
                
        except Exception as e:
            failed_count += 1
            continue
            
    if augmented_data_batch:
        df_chunk = pd.DataFrame(augmented_data_batch)
        write_header = not os.path.exists(output_path)
        df_chunk.to_csv(output_path, sep='\t', mode='a', index=False, header=write_header)
        total_saved += len(df_chunk)

    print(f"Done. Saved to {output_path} (Rows: {total_saved})")
    return output_path


def generate_tkm_data(generator, df_train, clips_dir, k, output_dir):
    print("\n" + "="*70)
    print("GENERATING TKM TRAINING DATA (INCREMENTAL SAVING)")
    print("="*70)
    
    output_path = os.path.join(output_dir, f"train_tkm_k{k}.tsv")
    if os.path.exists(output_path):
        os.remove(output_path)
        
    tkm_data_batch = []
    failed_count = 0
    total_saved = 0
    SAVE_EVERY = 500
    
    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="TKM Generation"):
        filename = os.path.basename(row['path'])
        audio_path = os.path.join(clips_dir, filename)
        
        if not os.path.exists(audio_path):
            failed_count += 1
            continue
        
        try:
            # Load directly from disk
            audio_array, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            
            hypotheses = generator.generate_beam_hypotheses(audio_array, sample_rate, num_beams=k)
            
            hypotheses_json = json.dumps([
                {'phonemes': h[0], 'score': float(h[1])} for h in hypotheses
            ], ensure_ascii=False)
            
            tkm_data_batch.append({
                'path': row['path'],
                'sentence': row['sentence'],
                'ground_truth_phonemes': row['phonemes'],
                'num_hypotheses': len(hypotheses),
                'hypotheses_json': hypotheses_json,
                'k': k
            })
            
            if (idx + 1) % SAVE_EVERY == 0:
                df_chunk = pd.DataFrame(tkm_data_batch)
                write_header = not os.path.exists(output_path)
                df_chunk.to_csv(output_path, sep='\t', mode='a', index=False, header=write_header)
                total_saved += len(df_chunk)
                tkm_data_batch = []
                
        except Exception as e:
            failed_count += 1
            continue

    if tkm_data_batch:
        df_chunk = pd.DataFrame(tkm_data_batch)
        write_header = not os.path.exists(output_path)
        df_chunk.to_csv(output_path, sep='\t', mode='a', index=False, header=write_header)
        total_saved += len(df_chunk)

    print(f"Done. Saved to {output_path} (Rows: {total_saved})")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate P2G training data for Polish (Incremental)")
    parser.add_argument("--s2p_model_path", type=str, required=True)
    parser.add_argument("--clips_dir", type=str, required=True, help="Directory containing .mp3 clips")
    parser.add_argument("--train_tsv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./p2g_training_polish")
    parser.add_argument("--generate_danp_beam", action="store_true")
    parser.add_argument("--generate_danp_sampling", action="store_true")
    parser.add_argument("--generate_tkm", action="store_true")
    parser.add_argument("--beam_size", type=int, default=32)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--tkm_k", type=int, default=32)
    
    args = parser.parse_args()
    
    import os
    args.s2p_model_path = os.path.abspath(args.s2p_model_path)
    args.train_tsv = os.path.abspath(args.train_tsv)
    args.output_dir = os.path.abspath(args.output_dir)
    args.clips_dir = os.path.abspath(args.clips_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check audio dir
    if not os.path.exists(args.clips_dir):
        print(f"❌ ERROR: Clips directory not found at {args.clips_dir}")
        return

    print(f"📁 Clips directory: {args.clips_dir}")
    
    generator = S2PDataGenerator(args.s2p_model_path)
    
    # 1. Load TSV
    if not os.path.exists(args.train_tsv):
        print(f"Error: Train TSV not found at {args.train_tsv}")
        return
        
    df_train = pd.read_csv(args.train_tsv, sep='\t')
    df_train = df_train.dropna(subset=['phonemes', 'sentence'])
    df_train = df_train[(df_train['phonemes'] != "") & (df_train['sentence'] != "")]
    print(f"Loaded {len(df_train)} training samples.")
    
    # 2. Generate
    if args.generate_danp_beam:
        generate_danp_beam_data(generator, df_train, args.clips_dir, args.beam_size, args.output_dir)
        
    if args.generate_danp_sampling:
        generate_danp_sampling_data(generator, df_train, args.clips_dir, args.num_samples, args.output_dir)
        
    if args.generate_tkm:
        generate_tkm_data(generator, df_train, args.clips_dir, args.tkm_k, args.output_dir)

if __name__ == "__main__":
    main()