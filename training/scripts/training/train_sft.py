#!/usr/bin/env python3
"""
OPTIMIZED SFT Training Script
- Internal Loss Calculation (Prevents huge logit transfers)
- Single Binary Data Loading
- Metal Warmup & Correct Metrics
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
from tqdm import tqdm

# ============================================================================
# Model Components
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
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    def forward(self, q, k):
        T = q.shape[1]
        cos = self.cos_cached[:T, :].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:T, :].unsqueeze(0).unsqueeze(2)
        q_rot = (q * cos) + (torch.cat((-q.chunk(2, -1)[1], q.chunk(2, -1)[0]), -1) * sin)
        k_rot = (k * cos) + (torch.cat((-k.chunk(2, -1)[1], k.chunk(2, -1)[0]), -1) * sin)
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
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
    def forward(self, x, mask=None):
        B, T, C = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim) for t in (q, k, v)]
        q, k = self.rope(q, k)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None: attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        return self.out_proj((attn @ v).transpose(1, 2).contiguous().view(B, T, C))

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, max_seq_len):
        super().__init__()
        self.attn = MultiHeadAttention(dim, n_heads, max_seq_len)
        self.ffn = SwiGLU(dim, hidden_dim=4 * dim)
        self.norm1, self.norm2 = RMSNorm(dim), RMSNorm(dim)
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class DeepSeekTransformer(nn.Module):
    def __init__(self, vocab_size=50257, dim=256, n_layers=4, n_heads=4, max_seq_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, n_heads, max_seq_len) for _ in range(n_layers)])
        self.norm_out = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx)
        mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1, 1, T, T)
        for block in self.blocks: x = block(x, mask)
        logits = self.lm_head(self.norm_out(x))
        if targets is not None:
            # Shift for causal training: targets are ALREADY aligned in SFT mode via dataset
            # But here we assume targets is a parallel tensor, so we shift
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = targets[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
            return logits, loss
        return logits, None

# ============================================================================
# Optimized Data Loading
# ============================================================================

class SFTPackedDataset(Dataset):
    def __init__(self, bin_file, context_length=512):
        self.ctx = context_length
        self.data = np.memmap(bin_file, dtype=np.int32, mode='r')
        self.num_samples = len(self.data) // (context_length * 2)
        print(f"Loaded {self.num_samples:,} samples from {bin_file}")
    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        start = idx * self.ctx * 2
        chunk = self.data[start:start + self.ctx * 2]
        x = torch.from_numpy(chunk[:self.ctx].astype(np.int64))
        y = torch.from_numpy(chunk[self.ctx:].astype(np.int64))
        return x, y

def get_lr(step, warmup=100, max_steps=5000, max_lr=1e-5, min_lr=1e-6):
    if step < warmup: return max_lr * (step + 1) / warmup
    if step >= max_steps: return min_lr
    ratio = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * ratio))

@torch.no_grad()
def generate_sample(model, tokenizer, prompt, device):
    model.eval()
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(100):
        logits = model(tokens[:, -512:])
        next_tok = torch.multinomial(F.softmax(logits[:, -1, :] / 0.8, dim=-1), 1)
        tokens = torch.cat([tokens, next_tok], dim=1)
        if next_tok.item() == tokenizer.eos_token_id: break
    model.train()
    return tokenizer.decode(tokens[0].cpu().numpy())

# ============================================================================
# Training
# ============================================================================

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base_dir = Path(__file__).parent.parent.parent
    log_dir, ckpt_dir = base_dir / "logs", base_dir / "checkpoints"
    for d in (log_dir, ckpt_dir): d.mkdir(exist_ok=True)
    
    log_fp = open(log_dir / "sft.log", 'a')
    def log_print(m): print(m); log_fp.write(m + '\n'); log_fp.flush()

    log_print(f"Device: {device} | Optimized SFT Start")
    dataset = SFTPackedDataset(str(base_dir / "data" / "processed" / "train_sft_packed.bin"))
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    model = DeepSeekTransformer().to(device)
    base_model = base_dir / "checkpoints" / "final_model.pt"
    if base_model.exists():
        ckpt = torch.load(base_model, map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        log_print("✓ Base weights loaded")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

    # Warmup Metal
    if device.type == "mps":
        log_print("Warmup Metal...")
        model.train()
        for _ in range(5):
            x, y = dataset[0]; x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)
            _, loss = model(x, y); loss.backward(); optimizer.zero_grad()
    
    batch_size, grad_accum = 16, 4
    total_steps = (len(dataset) // batch_size)
    
    start_time = time.time()
    global_step = 0
    
    for epoch in range(1):
        for step in range(total_steps):
            lr = get_lr(global_step, max_steps=total_steps // grad_accum)
            for pg in optimizer.param_groups: pg['lr'] = lr
            
            optimizer.zero_grad()
            accum_loss = 0
            for _ in range(grad_accum):
                # Using random fetch like original train.py for speed comparison
                indices = torch.randint(0, len(dataset), (batch_size,))
                xs, ys = [], []
                for idx in indices:
                    x, y = dataset[idx]; xs.append(x); ys.append(y)
                bx, by = torch.stack(xs).to(device), torch.stack(ys).to(device)
                
                _, loss = model(bx, by)
                loss = loss / grad_accum
                loss.backward()
                accum_loss += loss.item()
            
            optimizer.step()
            global_step += 1
            
            if global_step % 10 == 0 or global_step == 1:
                elapsed = (time.time() - start_time) / 60
                log_print(f"Step {global_step:4d} | Loss {accum_loss:.4f} | LR {lr:.1e} | {elapsed:.1f}m")
            
            if global_step % 100 == 0:
                print("\n" + "="*50)
                print(generate_sample(model, tokenizer, "### Instruction:\nWhat is entropy?\n\n### Response:\n", device))
                print("="*50 + "\n")
                torch.save(model.state_dict(), ckpt_dir / f"sft_step_{global_step}.pt")

    torch.save(model.state_dict(), ckpt_dir / "sft_final.pt")
    log_print("✓ Done!")

if __name__ == "__main__":
    train()
