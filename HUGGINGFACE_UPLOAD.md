# Инструкция по загрузке модели на HuggingFace

## Что нужно сделать

### 1. Установить зависимости

```bash
pip install huggingface_hub transformers
```

### 2. Авторизоваться в HuggingFace

```bash
huggingface-cli login
```

Введите ваш токен доступа (можно получить на https://huggingface.co/settings/tokens)

### 3. Создать репозиторий на HuggingFace

1. Перейдите на https://huggingface.co/new
2. Создайте новый репозиторий с именем `vel_17M`
3. Выберите тип "Model"
4. Установите видимость (Public или Private)

### 4. Подготовить модель для загрузки

Создайте скрипт `upload_to_hf.py`:

```python
from huggingface_hub import HfApi, upload_folder
from pathlib import Path
import torch

# Путь к модели
model_path = Path("models/final/model.pt")

# Загрузите модель
checkpoint = torch.load(model_path, map_location="cpu")

# Сохраните в формате safetensors (рекомендуется)
from safetensors.torch import save_file

# Конвертируйте state_dict в safetensors
output_dir = Path("hf_model")
output_dir.mkdir(exist_ok=True)

# Сохраните state_dict
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# Сохраните как safetensors
save_file(state_dict, output_dir / "model.safetensors")

# Создайте config.json
config = {
    "vocab_size": 50257,
    "dim": 256,
    "n_layers": 4,
    "n_heads": 4,
    "max_seq_len": 512,
    "architecture": "DeepSeekTransformer",
    "components": ["RMSNorm", "RoPE", "SwiGLU"]
}

import json
with open(output_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

# Создайте README.md для HF
readme_content = """---
license: mit
tags:
- transformer
- text-generation
- deepseek
- rmsnorm
- rope
- swiglu
---

# vel_17M

Transformer модель с архитектурой DeepSeek (RMSNorm, RoPE, SwiGLU) для генерации текста.

## Характеристики

- **Параметры**: ~15M (SFT версия)
- **Архитектура**: Transformer с RMSNorm, RoPE, SwiGLU
- **Контекст**: 512 токенов
- **Словарь**: GPT-2 tokenizer (50,257 токенов)

## Использование

```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained("Levos06/vel_17M")
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Использует GPT-2 tokenizer
```

## Архитектура

- RMSNorm вместо LayerNorm
- Rotary Position Embeddings (RoPE)
- SwiGLU активация
- Multi-Head Attention

## Обучение

Модель обучена на FineWeb-Edu dataset с последующим Supervised Fine-Tuning.
"""

with open(output_dir / "README.md", "w") as f:
    f.write(readme_content)

print(f"Модель подготовлена в {output_dir}")
```

### 5. Загрузить на HuggingFace

```python
from huggingface_hub import HfApi, upload_folder

api = HfApi()
repo_id = "Levos06/vel_17M"

# Загрузите папку
upload_folder(
    folder_path="hf_model",
    repo_id=repo_id,
    repo_type="model"
)

print(f"Модель загружена в {repo_id}")
```

Или используйте команду:

```bash
huggingface-cli upload Levos06/vel_17M hf_model/ --repo-type model
```

## Альтернативный способ (через git)

```bash
# Клонируйте репозиторий
git clone https://huggingface.co/Levos06/vel_17M
cd vel_17M

# Скопируйте файлы модели
cp ../hf_model/* .

# Закоммитьте и запушьте
git add .
git commit -m "Add model files"
git push
```

## Примечания

- Модель будет доступна по адресу: https://huggingface.co/Levos06/vel_17M
- Рекомендуется использовать формат safetensors вместо .pt для безопасности
- Добавьте README.md с описанием модели для лучшей видимости
- Укажите лицензию в README (например, MIT)
