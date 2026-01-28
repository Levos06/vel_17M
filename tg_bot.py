#!/usr/bin/env python3
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
import logging

# ============================================================================
# Model Components (Copied from train_sft.py)
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
        return logits, None

# ============================================================================
# Bot Implementation
# ============================================================================

from pathlib import Path

TOKEN = "8261989884:AAFaZPjfcC-_YQDLHGrGVOJMVRTttjiMTuw"
MODEL_PATH = Path(__file__).parent / "models" / "final" / "model.pt"

logging.basicConfig(level=logging.INFO)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = DeepSeekTransformer().to(device)

print(f"Loading model from {MODEL_PATH}...")
ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt)
model.eval()
print("Model loaded successfully.")

bot = Bot(token=TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    await message.answer("Hello! I'm a transformer-based bot. Ask me anything, and I'll try to answer.")

model_lock = asyncio.Lock()

def generate_response_sync(text: str) -> str:
    """Synchronous generation function to be run in a separate thread."""
    prompt = f"### Instruction:\n{text}\n\n### Response:\n"
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    generated_text = ""
    
    # Generation parameters
    max_new_tokens = 300
    temperature = 0.8
    top_p = 0.9
    repetition_penalty = 1.2
    eos_id = tokenizer.eos_token_id
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(tokens[:, -512:])
            logits = logits[:, -1, :] / temperature
            
            # Repetition penalty
            for token_id in set(tokens[0].tolist()):
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

            # Top-p (Nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[0, indices_to_remove] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            
            tokens = torch.cat([tokens, next_tok], dim=1)
            char = tokenizer.decode(next_tok[0].cpu().numpy())
            generated_text += char
            
            if next_tok.item() == eos_id:
                break
    
    response = generated_text.strip()
    end_token = tokenizer.eos_token if tokenizer.eos_token else "<|endoftext|>"
    if response.endswith(end_token):
        response = response[:-len(end_token)].strip()
    
    return response if response else "Sorry, I couldn't generate an answer."

@dp.message()
async def handle_message(message: types.Message):
    # Immediate feedback
    status_msg = await message.answer("🔄 Processing your request, please wait...")
    
    async with model_lock:
        try:
            # Offload heavy computation to a separate thread to keep the bot responsive
            response = await asyncio.to_thread(generate_response_sync, message.text)
            await status_msg.edit_text(response)
        except Exception as e:
            logging.error(f"Error during generation: {e}")
            await status_msg.edit_text("⚠️ An error occurred during generation.")

async def main():
    print("Bot is starting...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Bot stopped.")
