#!/usr/bin/env python3
"""
Training Script for DeepSeek-inspired Transformer
Implements: RMSNorm, RoPE, SwiGLU, Multi-Head Attention
Optimized for Apple M2 (MPS backend)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import time
import os
from pathlib import Path
from transformers import GPT2Tokenizer

# ============================================================================
# Model Components
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return (x / rms) * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        
        # Compute theta for each dimension pair
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos and sin for max sequence length
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, q, k):
        """Apply rotary embeddings to queries and keys"""
        seq_len = q.shape[1]
        
        # Get cos and sin for current sequence length
        cos = self.cos_cached[:seq_len, :]
        sin = self.sin_cached[:seq_len, :]
        
        # Apply rotation
        q_rot = self.apply_rotary_emb(q, cos, sin)
        k_rot = self.apply_rotary_emb(k, cos, sin)
        
        return q_rot, k_rot
    
    def apply_rotary_emb(self, x, cos, sin):
        """Apply rotary embedding to input tensor"""
        # x: [batch, seq_len, n_heads, head_dim]
        # cos, sin: [seq_len, head_dim]
        
        # Reshape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        # Standard RoPE rotation: (x * cos) + (rotate_half(x) * sin)
        # where rotate_half swaps and negates halves
        x1, x2 = x.chunk(2, dim=-1)
        x_rotated = torch.cat((-x2, x1), dim=-1)
        
        # Apply rotation
        return (x * cos) + (x_rotated * sin)


class SwiGLU(nn.Module):
    """SwiGLU Activation (from PaLM/LLaMA)"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x):
        # SwiGLU(x) = Swish(W1·x) ⊗ (W3·x) · W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with RoPE"""
    def __init__(self, dim, n_heads, max_seq_len=2048):
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # Rotary embeddings
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        # [batch, seq_len, dim] -> [batch, seq_len, n_heads, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        q, k = self.rope(q, k)
        
        # Transpose for attention computation
        # [batch, seq_len, n_heads, head_dim] -> [batch, n_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = attn @ v
        
        # Reshape back
        # [batch, n_heads, seq_len, head_dim] -> [batch, seq_len, dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm and modern components"""
    def __init__(self, dim, n_heads, max_seq_len=2048):
        super().__init__()
        self.attn = MultiHeadAttention(dim, n_heads, max_seq_len)
        self.ffn = SwiGLU(dim, hidden_dim=4 * dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
    
    def forward(self, x, mask=None):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class DeepSeekTransformer(nn.Module):
    """DeepSeek-inspired Transformer Language Model"""
    def __init__(
        self,
        vocab_size=50257,
        dim=512,
        n_layers=8,
        n_heads=8,
        max_seq_len=1024
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, max_seq_len)
            for _ in range(n_layers)
        ])
        
        # Output norm and projection
        self.norm_out = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights between embedding and output projection
        self.lm_head.weight = self.token_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())
        print(f"Model initialized with {self.n_params:,} parameters")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape
        
        # Token embeddings
        x = self.token_emb(idx)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=idx.device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Output normalization
        x = self.norm_out(x)
        
        # Language model head
        logits = self.lm_head(x)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1)
            )
        
        return logits, loss


# ============================================================================
# Data Loading
# ============================================================================

class TokenDataset(Dataset):
    """Memory-mapped token dataset"""
    def __init__(self, bin_file, context_length=1024):
        self.context_length = context_length
        
        # Load tokens using memmap
        self.tokens = np.memmap(bin_file, dtype=np.uint16, mode='r')
        self.n_tokens = len(self.tokens)
        
        print(f"Loaded {self.n_tokens:,} tokens from {bin_file}")
    
    def __len__(self):
        # Number of possible contexts
        return self.n_tokens - self.context_length
    
    def __getitem__(self, idx):
        # Get a chunk of tokens
        chunk = self.tokens[idx:idx + self.context_length + 1]
        
        # Input and target (shifted by 1)
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        
        return x, y


def get_batch(dataset, batch_size, device):
    """Get a random batch from dataset"""
    # Random indices
    indices = torch.randint(0, len(dataset), (batch_size,))
    
    # Get samples
    xs, ys = [], []
    for idx in indices:
        x, y = dataset[idx]
        xs.append(x)
        ys.append(y)
    
    # Stack into batches
    x_batch = torch.stack(xs).to(device)
    y_batch = torch.stack(ys).to(device)
    
    return x_batch, y_batch


# ============================================================================
# Training
# ============================================================================

def get_lr(step, warmup_steps=500, max_steps=50000, max_lr=3e-4, min_lr=3e-5):
    """Cosine learning rate schedule with warmup"""
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step + 1) / warmup_steps
    elif step >= max_steps:
        # Minimum learning rate
        return min_lr
    else:
        # Cosine decay
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


def generate_samples(model, tokenizer, step, device, num_samples=10):
    """Generate sample texts to monitor training progress"""
    import random
    
    # Random prompts for variety
    prompts = [
        "Once upon a time",
        "The future of",
        "In a world where",
        "Scientists have discovered",
        "The most important thing",
        "When I was young",
        "Technology has changed",
        "People often say",
        "The secret to success",
        "In the year 2050",
        "Education is",
        "Artificial intelligence will",
        "The problem with",
        "Many experts believe",
        "According to research"
    ]
    
    print("\n" + "=" * 70)
    print(f"GENERATION SAMPLES (Step {step})")
    print("=" * 70)
    
    model.eval()
    selected_prompts = random.sample(prompts, min(num_samples, len(prompts)))
    
    for i, prompt in enumerate(selected_prompts, 1):
        try:
            generated = generate(model, tokenizer, prompt, max_new_tokens=60, temperature=0.9, device=device)
            # Only show the generated part (remove prompt)
            generated_only = generated[len(prompt):].strip()
            print(f"\n{i}. Prompt: '{prompt}'")
            print(f"   → {generated_only[:150]}...")  # First 150 chars
        except Exception as e:
            print(f"\n{i}. Prompt: '{prompt}'")
            print(f"   → Error: {e}")
    
    print("\n" + "=" * 70 + "\n")
    model.train()


def train():
    # ========================================================================
    # Configuration
    # ========================================================================
    
    # Model (reduced to ~10M parameters for faster training)
    vocab_size = 50257
    dim = 256           # Reduced from 512
    n_layers = 4        # Reduced from 8
    n_heads = 4         # Reduced from 8
    max_seq_len = 512   # Reduced from 1024
    
    # Training
    batch_size = 16       # Can increase with smaller model + shorter context
    grad_accum_steps = 4  # Effective batch size = 64
    num_train_steps = 10000 # Number of steps to train in THIS run
    eval_interval = 100
    save_interval = 300  # Save every ~2 hours
    generation_interval = 150  # Generate samples every ~1 hour
    max_checkpoints = 5  # Keep only last 5 checkpoints
    
    # Optimizer
    learning_rate = 4e-5
    weight_decay = 0.1
    betas = (0.9, 0.95)
    
    # Base directory (training/)
    base_dir = Path(__file__).parent.parent.parent
    
    # Data
    data_file = base_dir / "data" / "processed" / "train.bin"
    
    # Device
    device = torch.device("mps")
    print(f"Using device: {device}")
    print()
    
    # ========================================================================
    # Setup
    # ========================================================================
    
    # Create checkpoint directory
    checkpoint_dir = base_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup logging
    log_dir = base_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # CSV for metrics
    metrics_file = log_dir / "training_metrics.csv"
    with open(metrics_file, 'w') as f:
        f.write("step,loss,lr,tokens_per_sec,eta_hours,elapsed_hours\n")
    
    # Text log file
    log_file = log_dir / "training.log"
    log_fp = open(log_file, 'w')
    
    def log_print(message):
        """Print to both console and file"""
        print(message)
        log_fp.write(message + '\n')
        log_fp.flush()
    
    log_print(f"Using device: {device}")
    log_print("")
    
    # Load dataset
    log_print("Loading dataset...")
    dataset = TokenDataset(data_file, context_length=max_seq_len)
    log_print("")
    
    # Load tokenizer for generation
    log_print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    log_print("")
    
    # Initialize model
    log_print("Initializing model...")
    model = DeepSeekTransformer(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len
    ).to(device)
    print()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas
    )
    
    # ========================================================================
    # Phase 2 Initialization
    # ========================================================================
    
    start_step = 0
    tokens_processed = 0
    
    base_model_path = base_dir / "checkpoints" / "final_model.pt"
    if base_model_path.exists():
        log_print(f"Initializing from base model: {base_model_path}")
        checkpoint = torch.load(base_model_path, map_location=device)
        # Handle both checkpoint formats (with or without 'model_state_dict')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        log_print("Base model loaded successfully.")
    else:
        log_print(f"Warning: Base model {base_model_path} not found. Starting from scratch.")
    
    # ========================================================================
    # Metal Compiler Warmup
    # ========================================================================
    
    log_print("Warming up Metal compiler...")
    model.train()
    for _ in range(10):
        x_warmup, y_warmup = get_batch(dataset, batch_size, device)
        _, loss = model(x_warmup, y_warmup)
        loss.backward()
        optimizer.zero_grad()
    log_print("Warmup complete!")
    log_print("")
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    
    target_steps = start_step + num_train_steps
    
    log_print("=" * 70)
    log_print("TRAINING START")
    log_print("=" * 70)
    log_print(f"Resuming from step: {start_step:,}")
    log_print(f"Training for: {num_train_steps:,} steps")
    log_print(f"Target step: {target_steps:,}")
    log_print(f"Batch size: {batch_size} (adjusted for MPS memory)")
    log_print(f"Gradient accumulation: {grad_accum_steps}")
    log_print(f"Effective batch size: {batch_size * grad_accum_steps}")
    log_print(f"Tokens per batch: {batch_size * grad_accum_steps * max_seq_len:,}")
    log_print("=" * 70)
    log_print("")
    
    training_start_time = time.time()
    step_losses = []
    
    for step in range(start_step, target_steps):
        step_start_time = time.time()
        
        # Learning rate schedule
        lr = get_lr(step - start_step, warmup_steps=150, max_steps=num_train_steps, max_lr=learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Gradient accumulation loop
        model.train()
        optimizer.zero_grad()
        
        accum_loss = 0.0
        for micro_step in range(grad_accum_steps):
            # Get batch
            x, y = get_batch(dataset, batch_size, device)
            
            # Forward pass
            _, loss = model(x, y)
            
            # Scale loss by accumulation steps
            loss = loss / grad_accum_steps
            accum_loss += loss.item()
            
            # Backward pass
            loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Track metrics
        step_losses.append(accum_loss)
        tokens_processed += batch_size * grad_accum_steps * max_seq_len
        
        # Logging
        if (step + 1) % eval_interval == 0 or step == 0:
            elapsed = time.time() - training_start_time
            step_time = time.time() - step_start_time
            
            # Calculate tokens/sec
            tokens_per_sec = (batch_size * grad_accum_steps * max_seq_len) / step_time
            
            # Estimate time remaining
            steps_completed_this_run = (step + 1) - start_step
            steps_remaining = target_steps - (step + 1)
            avg_step_time = elapsed / steps_completed_this_run
            eta_seconds = steps_remaining * avg_step_time
            eta_hours = eta_seconds / 3600
            elapsed_hours = elapsed / 3600
            
            # Average loss over last interval
            avg_loss = sum(step_losses[-eval_interval:]) / min(len(step_losses), eval_interval)
            
            # Log to console and file
            log_message = (f"Step {step+1:6d} | "
                          f"Loss {avg_loss:.4f} | "
                          f"LR {lr:.2e} | "
                          f"Tokens/s {tokens_per_sec:7.0f} | "
                          f"ETA {eta_hours:.1f}h")
            log_print(log_message)
            
            # Save metrics to CSV
            with open(metrics_file, 'a') as f:
                f.write(f"{step+1},{avg_loss:.6f},{lr:.8f},{tokens_per_sec:.2f},{eta_hours:.2f},{elapsed_hours:.2f}\n")
        
        # Generate samples periodically
        if (step + 1) % generation_interval == 0 and step > 0:
            generate_samples(model, tokenizer, step + 1, device, num_samples=3)
        
        # Checkpointing
        if (step + 1) % save_interval == 0:
            checkpoint_path = checkpoint_dir / f"step_{step+1}.pt"
            torch.save({
                'step': step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': accum_loss,
                'tokens_processed': tokens_processed,
            }, checkpoint_path)
            log_print(f"  → Saved checkpoint: {checkpoint_path}")
            
            # Keep only last N checkpoints
            checkpoints = sorted(checkpoint_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split('_')[1]))
            if len(checkpoints) > max_checkpoints:
                for old_ckpt in checkpoints[:-max_checkpoints]:
                    old_ckpt.unlink()
                    log_print(f"  → Deleted old checkpoint: {old_ckpt.name}")
    
    # ========================================================================
    # Training Complete
    # ========================================================================
    
    total_time = time.time() - training_start_time
    
    log_print("")
    log_print("=" * 70)
    log_print("TRAINING COMPLETE")
    log_print("=" * 70)
    log_print(f"Total time: {total_time / 3600:.2f} hours")
    log_print(f"Total tokens: {tokens_processed:,}")
    log_print(f"Average tokens/sec: {tokens_processed / total_time:.0f}")
    if step_losses:
        log_print(f"Final loss: {step_losses[-1]:.4f}")
    log_print("=" * 70)
    
    # Save final model
    final_path = checkpoint_dir / "final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'dim': dim,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'max_seq_len': max_seq_len,
        },
        'tokens_processed': tokens_processed,
    }, final_path)
    log_print(f"✓ Saved final model: {final_path}")
    log_print(f"✓ Training metrics saved to: {metrics_file}")
    log_print(f"✓ Training log saved to: {log_file}")
    
    # Close log file
    log_fp.close()
    
    return model


# ============================================================================
# Inference
# ============================================================================

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, device="mps"):
    """Generate text from a prompt"""
    model.eval()
    
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    # Generate
    for _ in range(max_new_tokens):
        # Crop to max context length
        tokens_crop = tokens if tokens.size(1) <= model.max_seq_len else tokens[:, -model.max_seq_len:]
        
        # Forward pass
        logits, _ = model(tokens_crop)
        
        # Get logits for last token
        logits = logits[:, -1, :] / temperature
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        tokens = torch.cat([tokens, next_token], dim=1)
        
        # Stop if EOS token (50256 for GPT-2)
        if next_token.item() == 50256:
            break
    
    # Decode
    generated = tokenizer.decode(tokens[0].cpu().numpy())
    return generated


def inference_test(model, device="mps"):
    """Test inference with sample prompts"""
    print("\n" + "=" * 70)
    print("INFERENCE TEST")
    print("=" * 70)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In a world where",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 70)
        generated = generate(model, tokenizer, prompt, max_new_tokens=50, device=device)
        print(generated)
        print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Train model
    model = train()
    
    # Test inference
    inference_test(model, device="mps")
    
    print("\n✓ All done!")
