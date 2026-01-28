#!/usr/bin/env python3
"""
Upload model to HuggingFace Hub
"""

from huggingface_hub import HfApi, upload_folder
from pathlib import Path
import torch
from safetensors.torch import save_file
import json

def main():
    # Paths
    model_path = Path("models/final/model.pt")
    output_dir = Path("hf_model")
    
    print("=" * 70)
    print("Preparing model for HuggingFace upload")
    print("=" * 70)
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Load model checkpoint
    print(f"\n1. Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    # Extract state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
    else:
        state_dict = checkpoint
        config = {}
    
    print(f"   Model state_dict keys: {len(state_dict)}")
    
    # Save as safetensors (recommended format)
    print("\n2. Converting to safetensors format...")
    safetensors_path = output_dir / "model.safetensors"
    
    # Handle shared weights (lm_head.weight and token_emb.weight share memory)
    # Create a copy to avoid shared memory issues
    state_dict_copy = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            state_dict_copy[key] = value.clone()
        else:
            state_dict_copy[key] = value
    
    save_file(state_dict_copy, safetensors_path)
    print(f"   Saved to {safetensors_path}")
    
    # Create config.json
    print("\n3. Creating config.json...")
    model_config = {
        "vocab_size": 50257,
        "dim": 256,
        "n_layers": 4,
        "n_heads": 4,
        "max_seq_len": 512,
        "architecture": "DeepSeekTransformer",
        "components": ["RMSNorm", "RoPE", "SwiGLU"],
        **config  # Add any additional config from checkpoint
    }
    
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
    print(f"   Saved to {config_path}")
    
    # Create README.md for HuggingFace
    print("\n4. Creating README.md...")
    readme_content = """---
license: mit
tags:
- transformer
- text-generation
- deepseek
- rmsnorm
- rope
- swiglu
- pytorch
---

# vel_17M

Transformer модель с архитектурой DeepSeek (RMSNorm, RoPE, SwiGLU) для генерации текста.

## Характеристики

- **Параметры**: ~15M (SFT версия)
- **Архитектура**: Transformer с RMSNorm, RoPE, SwiGLU
- **Контекст**: 512 токенов
- **Словарь**: GPT-2 tokenizer (50,257 токенов)
- **Обучение**: Pre-training + Supervised Fine-Tuning

## Использование

### С помощью load_model.py

```python
from load_model import load_model, generate

model, tokenizer, device = load_model()
text = generate(model, tokenizer, "The meaning of life is", max_new_tokens=100)
print(text)
```

### Прямая загрузка (после публикации)

```python
import torch
from transformers import GPT2Tokenizer
from safetensors.torch import load_file

# Загрузите модель
state_dict = load_file("model.safetensors")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Инициализируйте архитектуру (см. training/scripts/training/train_sft.py)
# и загрузите state_dict
```

## Архитектура

- **RMSNorm**: Root Mean Square Layer Normalization (более эффективная альтернатива LayerNorm)
- **RoPE**: Rotary Position Embeddings для лучшей экстраполяции длины
- **SwiGLU**: Gated activation function (SwiGLU = Swish(W1·x) ⊗ (W3·x) · W2)
- **Multi-Head Attention**: Стандартная causal attention

## Обучение

Модель обучена на:
1. FineWeb-Edu dataset для pre-training
2. Supervised Fine-Tuning на инструкциях

Подробности обучения см. в [training/README.md](https://github.com/Levos06/vel_17M/blob/main/training/README.md)

## Параметры генерации

- `temperature` (0.1-2.0): Контроль случайности
- `top_p` (0.0-1.0): Nucleus sampling
- `max_new_tokens`: Максимальное количество токенов

## Требования

- Python 3.8+
- PyTorch 2.1.0+
- transformers >= 4.30.0
- safetensors

## Лицензия

MIT License

## Ссылки

- **GitHub**: [Levos06/vel_17M](https://github.com/Levos06/vel_17M)
- **Архитектура**: Вдохновлена DeepSeek и LLaMA
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"   Saved to {readme_path}")
    
    # Upload to HuggingFace
    print("\n5. Uploading to HuggingFace...")
    repo_id = "levos06/vel_17M"
    
    try:
        api = HfApi()
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload model files"
        )
        print(f"\n✓ Successfully uploaded to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"\n✗ Error uploading: {e}")
        print("\nMake sure you are logged in:")
        print("  huggingface-cli login")
        return
    
    print("\n" + "=" * 70)
    print("Upload complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
