#!/usr/bin/env python3
"""
Plot training metrics from CSV file
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_metrics(metrics_file=None, save_dir=None):
    base_dir = Path(__file__).parent.parent.parent
    if metrics_file is None:
        metrics_file = base_dir / "logs" / "training_metrics.csv"
    if save_dir is None:
        save_dir = base_dir / "logs"
    """Plot training metrics and save figures"""
    
    # Load metrics
    df = pd.read_csv(metrics_file)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss curve
    axes[0, 0].plot(df['step'], df['loss'], linewidth=2, color='#2E86AB')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Learning rate schedule
    axes[0, 1].plot(df['step'], df['lr'], linewidth=2, color='#A23B72')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_title('Learning Rate Schedule')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Plot 3: Throughput
    axes[1, 0].plot(df['step'], df['tokens_per_sec'], linewidth=2, color='#F18F01')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Tokens/sec')
    axes[1, 0].set_title('Training Throughput')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Training time
    axes[1, 1].plot(df['step'], df['elapsed_hours'], linewidth=2, color='#C73E1D')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Hours')
    axes[1, 1].set_title('Elapsed Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_path = Path(save_dir) / "training_plots.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plots to: {save_path}")
    
    # Also create a simple loss-only plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['step'], df['loss'], linewidth=2.5, color='#2E86AB')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_path = Path(save_dir) / "loss_curve.png"
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved loss curve to: {loss_path}")
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("TRAINING STATISTICS")
    print("=" * 50)
    print(f"Initial loss: {df['loss'].iloc[0]:.4f}")
    print(f"Final loss: {df['loss'].iloc[-1]:.4f}")
    print(f"Loss reduction: {df['loss'].iloc[0] - df['loss'].iloc[-1]:.4f}")
    print(f"Average throughput: {df['tokens_per_sec'].mean():.0f} tokens/sec")
    print(f"Total training time: {df['elapsed_hours'].iloc[-1]:.2f} hours")
    print("=" * 50)

if __name__ == "__main__":
    import sys
    
    base_dir = Path(__file__).parent.parent.parent
    
    if len(sys.argv) > 1:
        metrics_file = Path(sys.argv[1])
    else:
        metrics_file = base_dir / "logs" / "training_metrics.csv"
    
    if metrics_file.exists():
        plot_training_metrics(str(metrics_file))
    else:
        print(f"Error: Metrics file not found: {metrics_file}")
        print("Run training first to generate metrics.")
