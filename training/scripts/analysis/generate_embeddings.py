import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

def generate_embeddings(data_dir, output_dir, model_name='all-MiniLM-L6-v2', batch_size=64):
    """
    Reads parquet files from data_dir, generates embeddings for the 'text' column,
    and saves them as .npy files in output_dir.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Initialize the model
    print(f"Loading model: {model_name}...")
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = SentenceTransformer(model_name, device=device)
    print(f"Model loaded on {device}")

    # Get all parquet files
    files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    files.sort()

    if not files:
        print(f"No parquet files found in {data_dir}")
        return

    for filename in files:
        file_path = os.path.join(data_dir, filename)
        output_file = os.path.join(output_dir, filename)

        if os.path.exists(output_file):
            print(f"Skipping {filename}, embeddings already exist.")
            continue

        print(f"Processing {filename}...")
        try:
            df = pd.read_parquet(file_path)
            if 'text' not in df.columns:
                print(f"Error: 'text' column not found in {filename}")
                continue

            texts = df['text'].astype(str).tolist()
            
            # Generate embeddings
            embeddings = model.encode(
                texts, 
                batch_size=batch_size, 
                show_progress_bar=True, 
                convert_to_numpy=True
            )

            # Add embeddings as a columns (stored as list for parquet compatibility)
            df['embedding'] = embeddings.tolist()

            # Save as parquet
            df.to_parquet(output_file)
            print(f"Saved {len(df)} rows with embeddings to {output_file}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    from pathlib import Path
    # Configure paths
    base_dir = Path(__file__).parent.parent.parent
    DATA_DIR = base_dir / "data" / "datasets" / "stanford"
    OUTPUT_DIR = base_dir / "data" / "embeddings"

    # Run the generator
    generate_embeddings(str(DATA_DIR), str(OUTPUT_DIR))
