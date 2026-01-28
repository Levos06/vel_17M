import pandas as pd
import os
from pathlib import Path

def sample_data(parquet_path, n=100):
    # Read the parquet file
    df = pd.read_parquet(parquet_path)
    
    # Sample n random rows
    sampled_df = df.sample(n=min(n, len(df)), random_state=42)
    
    # Select and rename columns if necessary
    # Based on previous analysis, columns are: 'instruction', 'response', 'instruction_source'
    output_df = sampled_df[['instruction', 'response']]
    
    # Base directory
    base_dir = Path(__file__).parent.parent.parent
    
    # Save to CSV
    csv_path = base_dir / "data" / "results" / "random_samples.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(csv_path, index=False)
    print(f"Saved 100 random rows to {csv_path}")
    
    # Save to Markdown
    md_path = base_dir / "data" / "results" / "random_samples.md"
    with open(md_path, 'w') as f:
        f.write("# Random Samples from Dataset\n\n")
        f.write(output_df.to_markdown(index=False))
    print(f"Saved 100 random rows to {md_path}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent
    it_dir = base_dir / "data" / "datasets" / "it"
    first_file = it_dir / "train-00000-of-00003-929c6c373c0473cd.parquet"
    
    if first_file.exists():
        sample_data(str(first_file))
    else:
        print(f"Error: File {first_file} not found.")
