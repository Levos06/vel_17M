import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import random
from tqdm import tqdm

def visualize_embeddings(stanford_dir, embeddings_dir, output_dir, n_samples=10000):
    """
    Randomly samples n_samples from each embedding file, performs t-SNE,
    and saves results with original text.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Get all embedding files
    emb_files = [f for f in os.listdir(embeddings_dir) if f.endswith('_embeddings.npy')]
    emb_files.sort()

    if not emb_files:
        print(f"No embedding files found in {embeddings_dir}")
        return

    for emb_name in emb_files:
        # Match with original parquet file
        base_name = emb_name.replace('_embeddings.npy', '.parquet')
        parquet_path = os.path.join(stanford_dir, base_name)
        emb_path = os.path.join(embeddings_dir, emb_name)
        
        output_name = emb_name.replace('_embeddings.npy', '_tsne.parquet')
        output_path = os.path.join(output_dir, output_name)

        if os.path.exists(output_path):
            print(f"Skipping {emb_name}, t-SNE results already exist.")
            continue

        if not os.path.exists(parquet_path):
            print(f"Warning: Original parquet {base_name} not found for {emb_name}. Skipping.")
            continue

        print(f"Processing {base_name}...")
        try:
            # Load embeddings and parquet
            embeddings = np.load(emb_path)
            df = pd.read_parquet(parquet_path)

            if len(embeddings) != len(df):
                print(f"Error: Mismatch in row counts for {base_name} ({len(df)}) and embeddings ({len(embeddings)})")
                continue

            # Random sampling
            n = min(n_samples, len(df))
            indices = random.sample(range(len(df)), n)
            
            sampled_embeddings = embeddings[indices]
            sampled_df = df.iloc[indices].copy()

            # Perform t-SNE
            print(f"Running t-SNE for {n} samples...")
            tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
            reduced_vectors = tsne.fit_transform(sampled_embeddings)

            # Add t-SNE results to the sampled dataframe
            sampled_df['tsne_x'] = reduced_vectors[:, 0]
            sampled_df['tsne_y'] = reduced_vectors[:, 1]

            # Save results
            sampled_df.to_parquet(output_path)
            print(f"Saved t-SNE results to {output_path}")

        except Exception as e:
            print(f"Error processing {emb_name}: {e}")

if __name__ == "__main__":
    from pathlib import Path
    base_dir = Path(__file__).parent.parent.parent
    STANFORD_DIR = base_dir / "data" / "datasets" / "stanford"
    EMBEDDINGS_DIR = base_dir / "data" / "embeddings"
    OUTPUT_DIR = base_dir / "data" / "results" / "tsne_results_10k"

    visualize_embeddings(str(STANFORD_DIR), str(EMBEDDINGS_DIR), str(OUTPUT_DIR))
