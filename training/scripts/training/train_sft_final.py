#!/usr/bin/env python3
"""
SFT Training Script based on original train.py
Preserves same efficiency and architecture.
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
# Model Components (EXACTLY AS IN train.py)
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    def forward(self, q, k):
        seq_len = q.shape[1]
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)
        q1, q2 = q.chunk(2, dim=-1)
        q_rotated = torch.cat((-q2, q1), dim=-1)
        q_rot = (q * cos) + (q_rotated * sin)
        k1, k2 = k.chunk(2, dim=-1)
        k_rotated = torch.cat((-k2, k1), dim=-1)
        k_rot = (k * cos) + (k_rotated * sin)
        return q_rot, k_rot

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len=2048):
        super().__init__()
        self.dim, self.n_heads = dim, n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        q, k = self.rope(q, k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None: attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len=2048):
        super().__init__()
        self.attn = MultiHeadAttention(dim, n_heads, max_seq_len)
        self.ffn = SwiGLU(dim, hidden_dim=4 * dim)
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class DeepSeekTransformer(nn.Module):
    def __init__(self, vocab_size=50257, dim=512, n_layers=8, n_heads=8, max_seq_len=1024):
        super().__init__()
        self.vocab_size, self.dim, self.max_seq_len = vocab_size, dim, max_seq_len
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads, max_seq_len) for _ in range(n_layers)])
        self.norm_out = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        batch_size, seq_len = idx.shape
        x = self.token_emb(idx)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=idx.device)).view(1, 1, seq_len, seq_len)
        for block in self.blocks: x = block(x, mask)
        logits = self.lm_head(self.norm_out(x))
        loss = None
        if targets is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = targets[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1), ignore_index=-100)
        return logits, loss

# ============================================================================
# Optimized Data Loading
# ============================================================================

class SFTPackedDataset(Dataset):
    """Dataset reading from our optimized [X,Y] packed binary file"""
    def __init__(self, bin_file, context_length=512):
        self.context_length = context_length
        # SFT packed file has [X(512 tokens), Y(512 tokens)]
        self.tokens = np.memmap(bin_file, dtype=np.int32, mode='r')
        self.num_samples = len(self.tokens) // (context_length * 2)
        print(f"Loaded {self.num_samples:,} SFT samples from {bin_file}")
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.context_length * 2
        # Extract X and Y (Labels are already shifted/masked during preparation)
        x = torch.from_numpy(self.tokens[start : start + self.context_length].astype(np.int64))
        y = torch.from_numpy(self.tokens[start + self.context_length : start + self.context_length * 2].astype(np.int64))
        return x, y

def get_batch(dataset, batch_size, device):
    indices = torch.randint(0, len(dataset), (batch_size,))
    xs, ys = [], []
    for idx in indices:
        x, y = dataset[idx]
        xs.append(x); ys.append(y)
    return torch.stack(xs).to(device), torch.stack(ys).to(device)

def get_lr(step, warmup_steps=100, max_steps=5000, max_lr=1e-5, min_lr=1e-6):
    if step < warmup_steps: return max_lr * (step + 1) / warmup_steps
    if step >= max_steps: return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * decay_ratio))

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=60, device="mps"):
    model.eval()
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        logits, _ = model(tokens[:, -512:])
        probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
        if next_token.item() == 50256: break
    model.train()
    return tokenizer.decode(tokens[0].cpu().numpy())

# ============================================================================
# Training (EXACTLY AS IN train.py)
# ============================================================================

def train():
    # Base Config (Must match final_model.pt)
    vocab_size, dim, n_layers, n_heads, max_seq_len = 50257, 256, 4, 4, 512
    batch_size, grad_accum_steps, num_train_steps = 16, 4, 5000
    learning_rate = 1e-5
    
    device = torch.device("mps")
    print(f"Using device: {device}\n")
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    checkpoint_dir = base_dir / "checkpoints"
    log_dir = base_dir / "logs"
    checkpoint_dir.mkdir(exist_ok=True); log_dir.mkdir(exist_ok=True)
    
    metrics_file = log_dir / "sft_metrics.csv"
    with open(metrics_file, 'w') as f: f.write("step,loss,lr\n")
    log_file = log_dir / "sft_training.log"
    log_fp = open(log_file, 'w')
    def log_print(m): print(m); log_fp.write(m + '\n'); log_fp.flush()

    # Load Data & Model
    dataset = SFTPackedDataset(str(base_dir / "data" / "processed" / "train_sft_packed.bin"), context_length=max_seq_len)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = DeepSeekTransformer(vocab_size, dim, n_layers, n_heads, max_seq_len).to(device)
    
    base_model_path = base_dir / "checkpoints" / "final_model.pt"
    if base_model_path.exists():
        log_print(f"Loading base model: {base_model_path}")
        ckpt = torch.load(base_model_path, map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Metal Warmup
    log_print("Warming up Metal...")
    model.train()
    for _ in range(10):
        x, y = get_batch(dataset, batch_size, device)
        _, loss = model(x, y); loss.backward(); optimizer.zero_grad()
    log_print("✓ Warmup complete")

    # Training Loop
    start_time = time.time()
    step_losses = []
    
    for step in range(num_train_steps):
        lr = get_lr(step, max_steps=num_train_steps, max_lr=learning_rate)
        for param_group in optimizer.param_groups: param_group['lr'] = lr
        
        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(grad_accum_steps):
            x, y = get_batch(dataset, batch_size, device)
            _, loss = model(x, y)
            loss = loss / grad_accum_steps
            accum_loss += loss.item()
            loss.backward()
        
        optimizer.step()
        step_losses.append(accum_loss)
        
        # Logging
        if (step + 1) % 20 == 0 or step == 0:
            elapsed = (time.time() - start_time) / 60
            avg_loss = sum(step_losses[-20:]) / len(step_losses[-20:])
            log_print(f"Step {step+1:4d} | Loss {avg_loss:.4f} | LR {lr:.2e} | {elapsed:.1f}m")
            with open(metrics_file, 'a') as f: f.write(f"{step+1},{avg_loss:.6f},{lr:.8f}\n")
        
        # Samples & Checkpoints
        if (step + 1) % 100 == 0:
            sample = generate(model, tokenizer, "### Instruction:\nWhat is a black hole?\n\n### Response:\n", device=device)
            log_print(f"\nSample Step {step+1}:\n{sample}\n" + "="*50)
            torch.save({'model_state_dict': model.state_dict(), 'step': step+1}, checkpoint_dir / f"sft_step_{step+1}.pt")
            
    torch.save(model.state_dict(), checkpoint_dir / "sft_final_model.pt")
    log_print("✓ Training COMPLETE")
    log_fp.close()

if __name__ == "__main__":
    train()
