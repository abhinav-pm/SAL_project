#!/usr/bin/env python3
"""
Phoneme-to-Grapheme (P2G) Inference Script
Converts phoneme sequences to text using trained P2G model

Usage:
    python predict_p2g.py --model_path ./p2g_model_polish_v1 --phonemes "ʃ ɛ ɕ tʂ ɛ ɕ"
    python predict_p2g.py --model_path ./p2g_model_polish_v1 --phoneme_file phonemes.txt
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

class P2GPredictor:
    """Phoneme-to-Grapheme prediction class"""
    
    def __init__(self, model_path):
        """Load P2G model and tokenizer"""
        print(f"[*] Loading P2G model from: {model_path}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print("[*] Model loaded successfully")
    
    def predict(self, phonemes, max_length=256, num_beams=4):
        """
        Convert phonemes to text
        
        Args:
            phonemes (str): Phoneme sequence (IPA symbols)
            max_length (int): Maximum generation length
            num_beams (int): Beam size for beam search
        
        Returns:
            str: Predicted text
        """
        # Tokenize input
        inputs = self.tokenizer(
            phonemes,
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # Decode
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text
    
    def predict_batch(self, phonemes_list, max_length=256, num_beams=4, batch_size=8):
        """
        Convert multiple phoneme sequences to text
        
        Args:
            phonemes_list (list): List of phoneme sequences
            max_length (int): Maximum generation length
            num_beams (int): Beam size
            batch_size (int): Batch size for processing
        
        Returns:
            list: List of predicted texts
        """
        results = []
        
        for i in range(0, len(phonemes_list), batch_size):
            batch = phonemes_list[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            # Decode
            texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results.extend(texts)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Phoneme-to-Grapheme inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to P2G model")
    parser.add_argument("--phonemes", type=str, help="Phoneme sequence to convert")
    parser.add_argument("--phoneme_file", type=str, help="File with phonemes (one per line)")
    parser.add_argument("--max_length", type=int, default=256, help="Max generation length")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam size")
    
    args = parser.parse_args()
    
    # Load predictor
    predictor = P2GPredictor(args.model_path)
    
    # Single phoneme prediction
    if args.phonemes:
        print(f"\n{'='*70}")
        print("Phoneme-to-Grapheme Conversion")
        print("="*70)
        print(f"Input (Phonemes):  {args.phonemes}")
        
        text = predictor.predict(args.phonemes, args.max_length, args.num_beams)
        
        print(f"Output (Text):     {text}")
        print("="*70)
    
    # Batch prediction from file
    elif args.phoneme_file:
        print(f"\n[*] Reading phonemes from: {args.phoneme_file}")
        
        with open(args.phoneme_file, 'r', encoding='utf-8') as f:
            phonemes_list = [line.strip() for line in f if line.strip()]
        
        print(f"[*] Found {len(phonemes_list)} phoneme sequences")
        print(f"[*] Converting to text...")
        
        texts = predictor.predict_batch(
            phonemes_list,
            max_length=args.max_length,
            num_beams=args.num_beams
        )
        
        # Save results
        output_file = args.phoneme_file.replace('.txt', '_output.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            for phoneme, text in zip(phonemes_list, texts):
                f.write(f"Phonemes: {phoneme}\n")
                f.write(f"Text: {text}\n")
                f.write("-" * 70 + "\n")
        
        print(f"\n{'='*70}")
        print(f"✓ Converted {len(texts)} phoneme sequences")
        print(f"✓ Results saved to: {output_file}")
        print("="*70)
        
        # Show first few examples
        print(f"\nFirst 3 examples:")
        for i in range(min(3, len(texts))):
            print(f"\n{i+1}. Phonemes: {phonemes_list[i]}")
            print(f"   Text: {texts[i]}")
    
    else:
        print("Error: Provide either --phonemes or --phoneme_file")
        parser.print_help()


if __name__ == "__main__":
    main()