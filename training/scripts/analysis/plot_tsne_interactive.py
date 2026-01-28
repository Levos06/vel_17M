import pandas as pd
import plotly.express as px
import os

def create_interactive_plot(tsne_dir, output_html):
    # Get all tsne parquet files
    files = [f for f in os.listdir(tsne_dir) if f.endswith('_tsne.parquet')]
    files.sort()
    
    if not files:
        print(f"No t-SNE result files found in {tsne_dir}")
        return
    
    first_file = files[0]
    file_path = os.path.join(tsne_dir, first_file)
    print(f"Loading {first_file}...")
    
    df = pd.read_parquet(file_path)
    
    # Create interactive scatter plot
    print("Generating interactive plot...")
    fig = px.scatter(
        df, 
        x='tsne_x', 
        y='tsne_y', 
        hover_data=['text'],
        title=f"t-SNE Visualization: {first_file}",
        template="plotly_dark"
    )
    
    # Update layout for better readability
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter"
        )
    )
    
    # Save as HTML
    fig.write_html(output_html)
    print(f"Interactive plot saved to {output_html}")
    print("You can open this file in your browser to explore the data.")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TSNE_DIR = os.path.join(BASE_DIR, 'tsne_results')
    OUTPUT_HTML = os.path.join(BASE_DIR, 'tsne_interactive_plot.html')
    
    create_interactive_plot(TSNE_DIR, OUTPUT_HTML)
