#!/usr/bin/env python3
"""
Load and use the trained transformer model for inference.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from pathlib import Path

# Import model architecture from training script
# We need to add the training scripts directory to path
import sys
import importlib.util

def load_model_class():
    """Dynamically load DeepSeekTransformer class from training script"""
    script_path = Path(__file__).parent / "training" / "scripts" / "training" / "train_sft.py"
    spec = importlib.util.spec_from_file_location("train_sft", script_path)
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    return train_module.DeepSeekTransformer

DeepSeekTransformer = load_model_class()

def load_model(model_path=None, device=None):
    """
    Load the trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint. Defaults to models/final/model.pt
        device: Device to load model on. Defaults to 'mps' on Mac, 'cuda' on Linux, 'cpu' otherwise
    
    Returns:
        model: Loaded model in eval mode
        tokenizer: GPT-2 tokenizer
        device: Device used
    """
    if model_path is None:
        model_path = Path(__file__).parent / "models" / "final" / "model.pt"
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    
    print(f"Loading model from {model_path}...")
    print(f"Using device: {device}")
    
    # Initialize model (adjust config based on your training)
    # Default config matches train_sft.py: dim=256, n_layers=4, n_heads=4, max_seq_len=512
    model = DeepSeekTransformer(
        vocab_size=50257,
        dim=256,
        n_layers=4,
        n_heads=4,
        max_seq_len=512
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle both dict and direct state_dict formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    print("Model loaded successfully!")
    return model, tokenizer, device


def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_p=0.9, device="mps"):
    """
    Generate text from a prompt.
    
    Args:
        model: Loaded model
        tokenizer: GPT-2 tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        device: Device to run on
    
    Returns:
        Generated text
    """
    model.eval()
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    eos_id = tokenizer.eos_token_id
    
    generated = tokens
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to max context length
            tokens_crop = generated[:, -512:] if generated.shape[1] > 512 else generated
            
            # Forward pass
            logits, _ = model(tokens_crop)
            logits = logits[:, -1, :] / temperature
            
            # Top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == eos_id:
                break
    
    # Decode
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and use the trained transformer model")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="The meaning of life is", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, device = load_model(args.model)
    
    # Generate
    print(f"\nPrompt: {args.prompt}")
    print("Generating...\n")
    
    output = generate(
        model, tokenizer, args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device
    )
    
    print("=" * 70)
    print("Generated text:")
    print("=" * 70)
    print(output)
    print("=" * 70)
