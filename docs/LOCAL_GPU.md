# Локальный запуск на компьютере с NVIDIA GPU

Проект не требует DataSphere, S3, Docker или облачного notebook. Все пути
относительны корню клонированного репозитория:

```text
DiatomDINO/
├── data/       # архивы и датасеты
└── artifacts/  # веса и отчёты
```

## 1. Окружение

Рекомендуется Python 3.10–3.12 и отдельное виртуальное окружение.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Сначала установите `torch` и `torchvision` из официального PyTorch selector для
версии CUDA, поддерживаемой локальным NVIDIA driver. Затем установите проект:

```bash
python -m pip install -e ".[faiss,dev]"
```

Если для ОС/GPU нет подходящего `faiss-cpu` wheel, можно оставить NumPy fallback
для небольшой gallery либо установить FAISS отдельно через conda.

## 2. Безопасная проверка

Команда ничего не скачивает и не запускает обучение. Она только читает версии,
CUDA-состояние и свободное место:

```bash
python -m scripts.check_environment --minimum-free-gb 100
```

Значение `100` является примером, а не гарантированной оценкой полного объёма:
перед скачиванием следует сравнить свободное место с актуальными размерами
четырёх Kaggle-архивов и оставить запас для PNG-копий и checkpoints.

## 3. Локальные данные и обучение

Все команды выполняются из корня репозитория:

```bash
python -m scripts.prepare_data all --config configs/data.yaml --dry-run
python -m scripts.prepare_data all --config configs/data.yaml
python -m scripts.run_train_detector --config configs/detector.yaml
python -m scripts.run_train_classifier --config configs/classifier.yaml
python -m scripts.run_build_gallery --config configs/classifier_benchmark.yaml
python -m scripts.run_test_classifier --config configs/classifier_benchmark.yaml
python -m scripts.run_test_supermodel --config configs/inference.yaml
```

Для другой GPU можно менять параметры без редактирования YAML:

```bash
python -m scripts.run_train_detector --config configs/detector.yaml \
  --set training.batch=4 --set training.workers=2

python -m scripts.run_train_classifier --config configs/classifier.yaml \
  --set loader.num_workers=2 --set loader.eval_batch_size=32
```

При нехватке VRAM сначала уменьшаются `batch`, `eval_batch_size` и `imgsz`, а не
размер датасета или состав benchmark.

## 4. Переносимость

- Windows и Linux поддерживаются через `pathlib`.
- Сгенерированный YOLO `data.yaml` содержит абсолютные локальные пути и должен
  пересоздаваться после переноса готового `data/` на другой компьютер.
- В Git не попадают `data/`, `artifacts/`, `outputs/` и `.venv/`.
- ClearML выключен и для локального запуска не требуется.
