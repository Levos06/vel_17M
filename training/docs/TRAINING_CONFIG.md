# Training Configuration Summary

## Model Architecture
- **Parameters**: ~60M
- **Layers**: 8
- **Attention Heads**: 8
- **Embedding Dim**: 512
- **Context Length**: 1024 tokens
- **Components**: RMSNorm, RoPE, SwiGLU

## Optimized Training Settings (for 16GB M2)

```python
batch_size = 32              # Optimized for 16GB RAM
grad_accum_steps = 2         # Effective batch size = 64
max_steps = 50000            
eval_interval = 100          # Log every 100 steps
save_interval = 300          # Save checkpoint every ~2 hours
generation_interval = 50     # Generate samples every ~15-20 min
max_checkpoints = 5          # Keep only last 5 checkpoints
```

## Performance Estimates

| Metric | Value |
|--------|-------|
| Tokens/sec | 2,500-3,000 |
| Time per step | ~26 seconds |
| Training time (50K steps) | ~8-9 days |
| Tokens processed | 3.28 billion |

## Disk Space Usage

| Component | Size |
|-----------|------|
| train.bin (dataset) | ~1-2 GB |
| 5 checkpoints @ 720 MB each | ~3.6 GB |
| final_model.pt | ~720 MB |
| **Total** | **~5-6 GB** |

## Checkpointing Schedule

Every **300 steps** (~2 hours):
- Saves checkpoint with model + optimizer state
- Automatically deletes oldest checkpoint
- Always keeps last 5 checkpoints

Example timeline:
```
Step 300:   2h    → Save checkpoint_1
Step 600:   4h    → Save checkpoint_2
Step 900:   6h    → Save checkpoint_3
Step 1200:  8h    → Save checkpoint_4
Step 1500:  10h   → Save checkpoint_5
Step 1800:  12h   → Save checkpoint_6, DELETE checkpoint_1
Step 2100:  14h   → Save checkpoint_7, DELETE checkpoint_2
...
```

## Text Generation Schedule

Every **50 steps** (~20 minutes):
- Generates 10 random texts
- Shows model progress in real-time
- Different prompts each time

## Memory Usage

```
Model weights:       ~237 MB
Optimizer state:     ~474 MB
Gradients:           ~237 MB
Activations (b=32):  ~300 MB
───────────────────────────
Total during training: ~1.4 GB (out of 16 GB)
```

## Quick Reference

**To start training**:
```bash
python train.py
```

**Expected output**:
```
Step    100 | Loss 8.2341 | LR 3.00e-04 | Tokens/s   2567 | ETA 8.5h

======================================================================
GENERATION SAMPLES (Step 50)
======================================================================
1. Prompt: 'Once upon a time'
   → [generated text]
...
======================================================================

  → Saved checkpoint: checkpoints/step_300.pt
```

**To resume from checkpoint**:
```python
# Modify train.py to load checkpoint before training loop
checkpoint = torch.load("checkpoints/step_1500.pt")
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_step = checkpoint['step']
```

## Full Timeline (50K steps)

| Day | Step | Time | Checkpoints | Events |
|-----|------|------|-------------|--------|
| 1 | 3,300 | 24h | step_3300.pt | 66 generations, 11 checkpoints |
| 2 | 6,600 | 48h | step_6600.pt | 132 generations total |
| 3 | 9,900 | 72h | step_9900.pt | 198 generations total |
| 4 | 13,200 | 96h | step_13200.pt | 264 generations total |
| 5 | 16,500 | 120h | step_16500.pt | 330 generations total |
| 6 | 19,800 | 144h | step_19800.pt | 396 generations total |
| 7 | 23,100 | 168h | step_23100.pt | 462 generations total |
| 8 | 26,400 | 192h | step_26400.pt | 528 generations total |
| 9 | 29,700 | 216h | step_29700.pt | 594 generations total |
| **DONE** | **50,000** | **~9 days** | **final_model.pt** | **1000 generations** |
