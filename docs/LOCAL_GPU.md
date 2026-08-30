# JupyterLab на локальной NVIDIA GPU

DiatomDINO рассчитан на JupyterLab в отдельном Python-окружении. DataSphere,
S3, Docker и облачный notebook не требуются. Данные и результаты остаются
локально:

```text
DiatomDINO/
├── .cache/     # torch/huggingface/matplotlib cache, ignored by Git
├── data/       # archives, datasetDiatom and splits, ignored by Git
└── artifacts/  # checkpoints and reports, ignored by Git
```

## 1. Драйвер и Python

До установки Python-пакетов проверьте наличие NVIDIA driver. Рекомендуется
Python 3.10-3.12 и отдельное виртуальное окружение.

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

## 2. CUDA-сборка PyTorch

Сначала установите `torch` и `torchvision` командой из официального PyTorch
selector, выбрав сборку, совместимую с локальным NVIDIA driver. Проект
намеренно не объявляет PyTorch обычной package dependency: иначе `pip` может
заменить рабочий CUDA wheel несовместимой или CPU-сборкой.

Сразу после установки проверьте в том же окружении:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Последнее значение должно быть `True`.

## 3. Проект, JupyterLab и kernel

```bash
python -m pip install -e ".[faiss,jupyter,dev]"
python -m ipykernel install --user --name diatom-dino \
  --display-name "Python (DiatomDINO GPU)"
python -m jupyter lab
```

Если для ОС нет подходящего `faiss-cpu` wheel, установите проект без extra
`faiss`. Для небольшой gallery используется NumPy fallback; альтернативно
FAISS можно поставить через conda.

В JupyterLab обязательно выберите kernel **Python (DiatomDINO GPU)**. Пункт
меню обычно находится в `Kernel -> Change Kernel`.

## 4. Первый preflight

Откройте `notebooks/public/00_environment.ipynb` и выполните ячейки сверху
вниз. Проверяются:

- путь к Python активного kernel;
- версия PyTorch и CUDA runtime;
- видимая NVIDIA GPU, compute capability и VRAM;
- наличие JupyterLab/ipykernel;
- свободное место для `data/`;
- локальные каталоги cache и artifacts.

Эквивалентная read-only команда:

```bash
python -m scripts.check_environment --data-root data \
  --minimum-free-gb 100 --minimum-vram-gb 8 --require-jupyter
```

Она ничего не скачивает и не запускает обучение.
Если `device=cuda`, DINOv2 не переходит на CPU автоматически: неверный kernel
завершит preflight с ошибкой до начала длительного обучения.

## 5. Последовательность notebooks

1. `00_environment.ipynb` - kernel, CUDA, GPU и VRAM.
2. `01_prepare_data.ipynb` - dry-run, загрузка, materialization и split audit.
3. `02_train_detector.ipynb` - YOLO11 train и однократный detector test.
4. `03_train_classifier.ipynb` - DINOv2 training и checkpoint metrics.
5. `04_retrieval_benchmark.ipynb` - Gunduz support gallery и retrieval test.
6. `05_e2e_benchmark.ipynb` - frozen YOLO -> DINOv2 -> FAISS benchmark.

Каждая долгая операция защищена флагом `RUN_* = False`. Сначала выполните
проверочные ячейки, затем измените только требуемый флаг на `True`.

## 6. Почему jobs запускаются как subprocess

Notebook не обучает модели внутри памяти kernel. `core.notebook_runtime`
запускает CLI через `sys.executable` активного kernel и передаёт ему выбранную
GPU. Благодаря этому:

- YOLO и DINO не оставляют CUDA tensors в kernel после завершения;
- повторный этап начинается с чистого GPU context;
- прерывание ячейки корректно останавливает дочерний процесс;
- stdout/stderr отображаются непосредственно под ячейкой;
- CLI и notebook используют одинаковые configs и код.

Torch Hub, Hugging Face и Matplotlib cache сохраняются в `.cache/` проекта, а
не во временном каталоге ОС. Поэтому веса DINOv2 не скачиваются заново после
перезапуска JupyterLab.

## 7. Настройка под VRAM

Начальные ориентиры:

| VRAM | YOLO batch | YOLO imgsz | DINO eval batch | workers |
|---:|---:|---:|---:|---:|
| 8-11 GiB | 2 | 768 | 16 | 2 |
| 12-15 GiB | 4 | 1024 | 32 | 4 |
| 16+ GiB | 8 | 1024 | 64 | 4 |

Это безопасные стартовые профили, а не гарантированные пределы. При OOM
сначала уменьшайте batch, затем eval batch и только после этого `imgsz`. Не
изменяйте состав датасета или frozen benchmark ради экономии VRAM.

На Windows `num_workers` применяется внутри отдельного CLI-процесса, а не в
kernel, поэтому multiprocessing DataLoader не зависит от notebook state. Если
worker завершается с ошибкой, временно задайте `WORKERS = 0` или `2`.

## 8. ClearML

ClearML необязателен и по умолчанию выключен. Установите extra `clearml`,
задайте ключи только через переменные окружения до запуска JupyterLab и
включите `ENABLE_CLEARML = True` в нужном notebook. Значения ключей никогда не
записываются в `.ipynb`.

## 9. Перенос проекта

- `pathlib` поддерживает Windows и Linux.
- YOLO `data.yaml` содержит абсолютные локальные пути, поэтому splits следует
  пересоздать после переноса готового `data/` на другой компьютер.
- В Git не попадают `.cache/`, `.config/`, `data/`, `artifacts/`, outputs и
  checkpoints notebook.
- Для нового полного эксперимента используйте новый output/data root, а не
  перезаписывайте frozen результаты.
