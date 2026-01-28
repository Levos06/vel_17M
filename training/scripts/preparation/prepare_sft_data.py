#!/usr/bin/env python3
import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os

def main():
    base_dir = Path(__file__).parent.parent.parent
    input_file = base_dir / "data" / "datasets" / "train-00000-of-00001.parquet"
    output_file = base_dir / "data" / "processed" / "train_sft_packed.bin"
    max_length = 512
    
    print("=" * 70)
    print("SFT Data Preparation (Optimized Single Binary)")
    print("=" * 70)
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Load dataset
    df = pd.read_parquet(input_file)
    print(f"Found {len(df):,} samples.")
    
    # We will save as [X(512 tokens), Y(512 tokens)] concatenated
    # Using int32 to safely store both token IDs and -100 masking
    all_data = []
    
    prefix = "### Instruction:\n"
    response_marker = "\n\n### Response:\n"
    eos_id = tokenizer.eos_token_id
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        instruction = str(row['question'])
        response = str(row['answer'])
        
        prompt_text = f"{prefix}{instruction}{response_marker}"
        full_text = f"{prompt_text}{response}{tokenizer.eos_token}"
        
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
        
        padding_len = max_length - len(full_ids)
        x_ids = full_ids + [eos_id] * padding_len
        
        prompt_len = min(len(prompt_ids), len(full_ids))
        y_ids = ([-100] * prompt_len) + full_ids[prompt_len:]
        y_ids = y_ids + ([-100] * padding_len)
        
        # Concat X and Y (512 + 512 = 1024 tokens)
        all_data.extend(x_ids)
        all_data.extend(y_ids)
        
    print("\nSaving to binary file...")
    data_array = np.array(all_data, dtype=np.int32)
    data_array.tofile(output_file)
    
    print(f"✓ Created {output_file} ({os.path.getsize(output_file)/(1024*1024):.2f} MB)")
    print("✓ Each sample is a block of 1024 tokens (512 inputs + 512 labels).")

if __name__ == "__main__":
    main()
