# DiatomDINO

Компактный pet-проект для детекции и retrieval-классификации диатомей:

```text
Gunduz image → YOLO11 Detect → grayscale crop → DINOv2 embedding → FAISS top-k
```

Проект рассчитан на JupyterLab в локальном Windows/Linux окружении с NVIDIA
GPU. Облачные хранилища, DataSphere, S3 и `/home/jupyter` не используются.
Настройка kernel и CUDA описана в [`docs/LOCAL_GPU.md`](docs/LOCAL_GPU.md).

## Что входит в проект

- Детектор обучается только на полных изображениях Gunduz.
- Классификатор обучается только на публичных crop-датасетах UDE,
  Diatom1042 и Siyue Pu.
- Финальный benchmark строится из отложенных изображений Gunduz.
- Данные NII не скачиваются, не материализуются и не участвуют в обучении.
- Класс `Unknown` не обучается и пороги open-set не нужны: основной режим
  является closed-set retrieval.

Такое разделение позволяет проверять переносимость представления:
классификатор не видит фотографии Gunduz при оптимизации, но использует
небольшую Gunduz-gallery при итоговом retrieval-тесте.

## Архитектура

```text
classifier/       DINOv2, projection head, loss, sampler, metrics
detector/         YOLO wrapper, trainer and tester
inference/        FAISS retrieval and end-to-end pipeline
core/             configuration, logging, base lifecycle
data_pipeline/    downloader, parser, materializer and splitter
scripts/          thin command-line entry points
configs/          current configurations
notebooks/        guarded local-GPU workflows (long actions are off by default)
data/             downloaded/runtime data; ignored by Git
artifacts/        checkpoints and reports; ignored by Git
tests/            unit tests
```

Актуальные notebooks находятся в `notebooks/public/`: GPU preflight, подготовка данных,
обучение и тест YOLO, обучение DINOv2, retrieval benchmark и финальная E2E
оценка. Исторические исследовательские notebooks не входят в публичную версию.

- [`00_environment.ipynb`](notebooks/public/00_environment.ipynb)
- [`01_prepare_data.ipynb`](notebooks/public/01_prepare_data.ipynb)
- [`02_train_detector.ipynb`](notebooks/public/02_train_detector.ipynb)
- [`03_train_classifier.ipynb`](notebooks/public/03_train_classifier.ipynb)
- [`04_retrieval_benchmark.ipynb`](notebooks/public/04_retrieval_benchmark.ipynb)
- [`05_e2e_benchmark.ipynb`](notebooks/public/05_e2e_benchmark.ipynb)

Физическая структура данных описана в `data/README.md`.

## Установка

Сначала установите CUDA-сборку `torch`/`torchvision`, соответствующую локальному
NVIDIA driver, через официальный PyTorch selector. Эти пакеты намеренно не
разрешаются проектом через обычный PyPI. Затем установите JupyterLab и проект:

```bash
python -m pip install -e ".[faiss,jupyter,dev]"
python -m ipykernel install --user --name diatom-dino \
  --display-name "Python (DiatomDINO GPU)"
python -m jupyter lab
```

В готовом CUDA-образе можно установить проект без `faiss` extra и использовать
FAISS из conda/образа.

ClearML необязателен: `python -m pip install -e ".[clearml]"`.

В JupyterLab выберите kernel `Python (DiatomDINO GPU)` и начните с
`notebooks/public/00_environment.ipynb`. Долгие этапы выполняются отдельными
дочерними процессами выбранного kernel, поэтому вывод остаётся в notebook, а
CUDA-память освобождается между стадиями.

## Подготовка данных

Все четыре архива скачиваются один раз. Builder не распаковывает исходники
целиком: он читает ZIP и материализует только полные Gunduz-изображения, YOLO
labels и необходимые классификационные crops.

```bash
python -m scripts.prepare_data all --config configs/data.yaml --dry-run
python -m scripts.prepare_data all --config configs/data.yaml
```

Операции атомарны и не перезаписывают готовые `data/datasetDiatom` или
`data/splits`. Для повторной сборки используйте новый root:

```bash
python -m scripts.prepare_data all --set data_root=data-v2
```

## Обучение и оценка

```bash
# 1. Детектор: Gunduz train/val/test
python -m scripts.run_train_detector --config configs/detector.yaml
python -m scripts.run_test_detector --config configs/detector.yaml \
  --weights artifacts/detector/yolo11m/weights/best.pt

# 2. DINOv2: UDE + Diatom1042 + Siyue Pu
python -m scripts.run_train_classifier --config configs/classifier.yaml

# 3. Gunduz support gallery
python -m scripts.run_build_gallery \
  --config configs/classifier_benchmark.yaml \
  --checkpoint artifacts/classifier/dinov2/best.pt

# 4. Known / unseen-species / unseen-genus retrieval
python -m scripts.run_test_classifier \
  --config configs/classifier_benchmark.yaml \
  --checkpoint artifacts/classifier/dinov2/best.pt

# 5. Полный YOLO → DINO → FAISS тест
python -m scripts.run_test_supermodel --config configs/inference.yaml
```

## Что здесь называется zero-shot

Чистая zero-shot классификация нового латинского названия невозможна для
DINOv2+FAISS без текстовой модели: индекс должен иметь хотя бы один подписанный
эталон. Поэтому проект использует термин **unseen-class retrieval**:

- `known` — вид присутствовал в публичном train;
- `unseen_species` — вид не участвовал в обучении, но его род участвовал;
- `unseen_genus` — ни род, ни вид не участвовали в обучении;
- название всегда приходит из независимой Gunduz support-gallery.

Протоколы находятся в `data/splits/benchmark/gunduz/`.

## Проверки

```bash
python -m pytest tests/test_public_data_pipeline.py tests/test_decision_logic.py
```

Исторические NII/open-set эксперименты сохранены локально, но не являются
частью новой публичной архитектуры. Они перечислены в `docs/LEGACY.md`.
