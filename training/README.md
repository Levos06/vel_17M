# Training Directory

Эта папка содержит все материалы, связанные с обучением модели.

## Структура

```
training/
├── data/                    # Данные
│   ├── datasets/           # Исходные датасеты
│   │   ├── stanford/       # Stanford dataset
│   │   ├── stanford_train/ # Обработанный Stanford dataset
│   │   ├── it/             # Italian dataset
│   │   └── *.parquet       # Файлы датасетов
│   ├── processed/          # Обработанные данные для обучения
│   │   ├── train.bin       # Токенизированные данные для основного обучения
│   │   └── train_sft_packed.bin  # Данные для SFT
│   ├── embeddings/         # Сгенерированные эмбеддинги
│   └── results/            # Результаты обработки (t-SNE, выборки и т.д.)
│
├── checkpoints/            # Чекпоинты обучения
│   ├── step_*.pt          # Чекпоинты основного обучения
│   ├── sft_step_*.pt      # Чекпоинты SFT
│   └── final_model.pt     # Финальная модель основного обучения
│
├── logs/                   # Логи и метрики
│   ├── training.log       # Лог основного обучения
│   ├── training_metrics.csv  # Метрики обучения
│   ├── sft_training.log    # Лог SFT
│   └── sft_metrics.csv     # Метрики SFT
│
├── scripts/                # Скрипты обучения
│   ├── preparation/        # Подготовка данных
│   │   ├── prepare_data.py         # Подготовка данных для основного обучения
│   │   ├── prepare_sft_data.py     # Подготовка данных для SFT
│   │   ├── filter_dataset.py       # Фильтрация датасета
│   │   └── sample_data.py          # Выборка данных
│   │
│   ├── training/          # Скрипты обучения
│   │   ├── train.py                # Основное обучение transformer
│   │   ├── train_sft.py            # Supervised Fine-Tuning
│   │   ├── train_sft_final.py      # Финальная версия SFT
│   │   └── setfit_train.py         # Обучение SetFit модели для фильтрации
│   │
│   └── analysis/          # Анализ и визуализация
│       ├── generate_embeddings.py  # Генерация эмбеддингов
│       ├── visualize_embeddings.py # Визуализация эмбеддингов
│       ├── plot_metrics.py         # Построение графиков метрик
│       ├── plot_tsne_interactive.py # Интерактивная визуализация t-SNE
│       └── loss.py                 # Анализ loss функции
│
└── docs/                   # Документация
    ├── LOGGING.md
    ├── TRAINING_CONFIG.md
    ├── TRAINING_PROCESS.md
    ├── OPTIMIZED_CONFIG.md
    └── PROJECT_STRUCTURE.md
```

## Процесс обучения

### Этап 1: Подготовка данных

1. **Фильтрация датасета** (опционально):
   ```bash
   python training/scripts/preparation/filter_dataset.py
   ```
   Фильтрует датасет для академических текстов.

2. **Подготовка данных для основного обучения**:
   ```bash
   python training/scripts/preparation/prepare_data.py
   ```
   Создает `data/processed/train.bin` из датасетов.

3. **Подготовка данных для SFT**:
   ```bash
   python training/scripts/preparation/prepare_sft_data.py
   ```
   Создает `data/processed/train_sft_packed.bin`.

### Этап 2: Основное обучение

```bash
python training/scripts/training/train.py
```

Обучает transformer модель с нуля:
- Использует `data/processed/train.bin`
- Сохраняет чекпоинты в `checkpoints/step_*.pt`
- Финальная модель: `checkpoints/final_model.pt`
- Логи: `logs/training.log` и `logs/training_metrics.csv`

### Этап 3: Supervised Fine-Tuning (SFT)

```bash
python training/scripts/training/train_sft_final.py
```

Дообучает модель на инструкциях:
- Использует `data/processed/train_sft_packed.bin`
- Загружает базовую модель из `checkpoints/final_model.pt`
- Сохраняет чекпоинты в `checkpoints/sft_step_*.pt`
- Логи: `logs/sft_training.log` и `logs/sft_metrics.csv`

### Этап 4: Обучение фильтра (опционально)

```bash
python training/scripts/training/setfit_train.py
```

Обучает SetFit модель для фильтрации академических текстов:
- Сохраняет модель в `../models/academic_filter_model/`

## Анализ результатов

### Построение графиков метрик:
```bash
python training/scripts/analysis/plot_metrics.py
```

### Генерация эмбеддингов:
```bash
python training/scripts/analysis/generate_embeddings.py
```

### Визуализация эмбеддингов:
```bash
python training/scripts/analysis/visualize_embeddings.py
```

## Конфигурация

Все пути в скриптах настроены относительно директории `training/`. Скрипты автоматически определяют базовую директорию через `Path(__file__).parent.parent.parent`.

## Примечания

- Все большие файлы (датасеты, чекпоинты) игнорируются Git через `.gitignore`
- Финальная модель копируется в `../models/final/model.pt` для использования
- Логи и метрики сохраняются в `logs/` для анализа
- TensorBoard runs находятся в `logs/runs/`
