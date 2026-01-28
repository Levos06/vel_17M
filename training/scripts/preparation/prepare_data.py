#!/usr/bin/env python3
"""
Data Preparation Script for Stage 2
Processes local parquet files from stanford_train and tokenizes them
Saves as memory-mapped binary file for efficient training
"""

import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os

def main():
    # Configuration
    base_dir = Path(__file__).parent.parent.parent
    data_dir = base_dir / "data" / "datasets" / "stanford_train"
    output_file = base_dir / "data" / "processed" / "train.bin"
    
    print("=" * 70)
    print("DeepSeek-Inspired Transformer - Data Preparation (Stage 2)")
    print("=" * 70)
    print(f"\nSource Directory: {data_dir}")
    print(f"Output: {output_file}")
    print()
    
    # Initialize GPT-2 tokenizer
    print("Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print()
    
    # Get local parquet files
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} not found.")
        return
        
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    files.sort()
    
    if not files:
        print(f"No parquet files found in {data_dir}")
        return
        
    print(f"Found {len(files)} parquet files.")
    
    # Tokenize and collect all token IDs
    all_tokens = []
    total_chars = 0
    total_docs = 0
    
    for filename in files:
        file_path = os.path.join(data_dir, filename)
        print(f"Processing {filename}...")
        
        try:
            df = pd.read_parquet(file_path)
            if 'text' not in df.columns:
                print(f"Warning: 'text' column not found in {filename}. Skipping.")
                continue
                
            texts = df['text'].astype(str).tolist()
            total_docs += len(texts)
            
            for text in tqdm(texts, desc=f"Tokenizing {filename}", leave=False):
                total_chars += len(text)
                # Tokenize the text
                tokens = tokenizer.encode(text, add_special_tokens=False)
                all_tokens.extend(tokens)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    # Convert to numpy array with uint16 dtype
    print("\nConverting to NumPy array...")
    token_array = np.array(all_tokens, dtype=np.uint16)
    
    # Save as memory-mapped binary file
    print(f"Saving to {output_file}...")
    token_array.tofile(output_file)
    
    # Display statistics
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    file_size_gb = file_size_mb / 1024
    
    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Total files processed: {len(files)}")
    print(f"Total documents processed: {total_docs:,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Total tokens: {len(all_tokens):,}")
    print(f"File size: {file_size_mb:.2f} MB ({file_size_gb:.3f} GB)")
    if total_docs > 0:
        print(f"Average tokens per document: {len(all_tokens) / total_docs:.1f}")
    if len(all_tokens) > 0:
        print(f"Compression ratio: {total_chars / len(all_tokens):.2f} chars/token")
    print()
    print(f"✓ Saved to: {output_file}")
    print(f"✓ Ready for Stage 2 training!")
    print("=" * 70)

if __name__ == "__main__":
    main()
