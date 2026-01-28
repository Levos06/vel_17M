# Структура проекта

Этот документ описывает организацию файлов и папок в проекте.

## Основные директории

### `data/`
Содержит все данные проекта:
- **`data/datasets/`** - исходные и обработанные датасеты
  - `stanford/` - Stanford dataset
  - `it/` - Italian dataset  
  - `stanford_train/` - обработанный Stanford dataset
  - `*.parquet` - датасеты в формате Parquet
  - `*.bin` - токенизированные данные в бинарном формате
- **`data/results/`** - результаты обработки данных
  - `tsne_results/` - результаты t-SNE визуализации
  - `tsne_results_10k/` - результаты t-SNE для 10k выборок

### `models/`
Сохраненные обученные модели:
- `academic_filter_model/` - SetFit модель для фильтрации академических текстов

### `checkpoints/`
Чекпоинты обучения:
- `checkpoint-100/` - HuggingFace checkpoint
- `step_*.pt` - чекпоинты основного обучения
- `sft_step_*.pt` - чекпоинты supervised fine-tuning
- `final_model.pt` - финальная модель

### `logs/`
Логи и метрики обучения:
- `training.log` - лог основного обучения
- `training_metrics.csv` - метрики обучения
- `sft_training.log`, `sft_metrics.csv` - логи и метрики SFT

### `outputs/`
Сгенерированные выходные данные:
- **`outputs/embeddings/`** - эмбеддинги в формате .npy
- **`outputs/visualizations/`** - графики и визуализации (HTML, изображения)

### `scripts/`
Вспомогательные скрипты:
- `loss.py` - анализ и визуализация loss функции

### `runs/`
TensorBoard runs для визуализации метрик

## Корневая директория

### Основные скрипты
- `train.py` - основной скрипт обучения transformer
- `train_sft.py` - supervised fine-tuning
- `train_sft_final.py` - финальная версия SFT
- `prepare_data.py` - подготовка данных
- `prepare_sft_data.py` - подготовка данных для SFT
- `setfit_train.py` - обучение SetFit модели
- `filter_dataset.py` - фильтрация датасета
- `generate_embeddings.py` - генерация эмбеддингов
- `visualize_embeddings.py` - визуализация эмбеддингов
- `plot_metrics.py` - построение графиков метрик
- `plot_tsne_interactive.py` - интерактивная визуализация t-SNE
- `sample_data.py` - выборка данных
- `tg_bot.py` - Telegram бот

### Документация
- `README.md` - основная документация
- `QUICKSTART.md` - быстрый старт
- `LOGGING.md` - документация по логированию
- `TRAINING_CONFIG.md` - конфигурация обучения
- `TRAINING_PROCESS.md` - процесс обучения
- `OPTIMIZED_CONFIG.md` - оптимизированная конфигурация

## Игнорируемые файлы (.gitignore)

Следующие типы файлов игнорируются Git:
- Python кэш (`__pycache__/`, `*.pyc`)
- Данные (`*.parquet`, `*.bin`, `*.pt`, `*.npy`)
- Логи (`*.log`, `*.csv`)
- Системные файлы (`.DS_Store`)
- IDE файлы (`.vscode/`, `.idea/`)

## Примечания

- Большие файлы (датасеты, чекпоинты, эмбеддинги) не должны попадать в Git
- Все временные файлы должны быть в соответствующих папках
- Скрипты должны быть в корне или в `scripts/` для удобства запуска
