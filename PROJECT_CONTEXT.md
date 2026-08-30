# Контекст проекта DiatomDINO

## Назначение

DiatomDINO — компактный публичный pet-проект для детекции и иерархического
retrieval диатомей. Рабочий каскад:

```text
Gunduz image -> YOLO11 Detect -> grayscale crop -> DINOv2 -> FAISS top-k
```

## Границы публичной версии

- NII не используется ни в train, ни в validation, ни в benchmark.
- YOLO обучается только на полных изображениях Gunduz.
- DINOv2 обучается только на UDE, Diatom1042 и Siyue Pu.
- Независимый benchmark строится из Gunduz test images.
- ResNet artifact gate и обучение класса `Unknown` исключены.
- Inference работает в closed-set режиме: непустой FAISS-индекс всегда
  возвращает наиболее близкий род и вид.
- Старые NII/open-set/stage2.5/stage3 протоколы являются legacy и перечислены в
  `docs/LEGACY.md`.
- Целевая среда — JupyterLab в локальном Windows/Linux окружении с NVIDIA GPU.
  В актуальной архитектуре запрещены S3-пути, `/home/jupyter` и зависимости от
  DataSphere.

## Основные каталоги

- `classifier/` — DINOv2, projection head, SupCon, sampler и метрики.
- `detector/` — YOLO model/trainer/tester.
- `inference/` — FAISS, decision logic и E2E pipeline.
- `core/` — конфигурация, ClearML и базовые lifecycle-классы.
- `data_pipeline/` — загрузка, parsing, materialization и splits.
- `scripts/` — тонкие CLI entry points.
- `configs/` — актуальные YAML.
- `notebooks/public/` — основной JupyterLab workflow с отдельным GPU preflight
  и выключенными по умолчанию флагами долгих операций.
- `core/notebook_runtime.py` — единый root/cache/GPU context и безопасный запуск
  CLI в дочернем процессе активного Jupyter kernel.
- `data/` — runtime data, кроме README не попадает в Git.
- `artifacts/` — checkpoints/reports, не попадает в Git.

## Данные

`python -m scripts.prepare_data all --config configs/data.yaml`:

1. атомарно скачивает четыре immutable ZIP в `data/raw/archives`;
2. читает Gunduz XML и материализует полные PNG, YOLO bbox labels и bbox crops;
3. читает UDE, Diatom1042 и Siyue Pu напрямую из ZIP;
4. извлекает метки штатными source-specific parsers;
5. исключает augmentation/точные дубликаты и повреждённые файлы;
6. сохраняет внешние crops как grayscale RGB PNG;
7. пишет `images.csv`, `objects.csv`, `crops.csv`, `source_audit.csv`;
8. создаёт detector/classifier/benchmark splits и `audit.json`.

Готовые `datasetDiatom` и `splits` никогда не перезаписываются. Для новой версии
задаётся другой `data_root`. NII не является конфигурируемым source и повторно
проверяется валидатором перед фиксацией результата.

## Splits

### Detector

Gunduz images делятся детерминированно на `train=70%`, `val=15%`, `test=15%`.
YOLO получает txt-views с абсолютными путями, поэтому изображения не копируются
по split-каталогам.

### Classifier

UDE, Diatom1042 и Siyue Pu делятся внутри каждого вида на train/validation.
Каждый непустой класс сохраняет хотя бы один train crop; singleton-классы идут
только в train и поддерживаются sampling-with-replacement.

### Gunduz benchmark

Используются crops только из detector test images. Целое исходное изображение
попадает либо в support gallery, либо в query, поэтому фон одного снимка не
пересекает роли. Query получают статус:

- `known` — точная пара genus/species была в classifier train;
- `unseen_species` — genus был, species не было;
- `unseen_genus` — genus отсутствовал.

Это unseen-class retrieval, а не чистая zero-shot классификация: название
нового таксона приходит из подписанного Gunduz support-примера в FAISS.

## Актуальные конфигурации

- `configs/data.yaml` — источники и splits.
- `configs/detector.yaml` — Gunduz YOLO.
- `configs/classifier.yaml` — public-source DINO training.
- `configs/classifier_benchmark.yaml` — Gunduz gallery/query evaluation.
- `configs/inference.yaml` — closed-set E2E test.

ClearML по умолчанию выключен, чтобы clone запускался без секретов; его можно
включить override `--set clearml.enabled=true`.

Перед установкой проекта пользователь отдельно устанавливает CUDA-сборку
PyTorch, совместимую с драйвером компьютера. `torch` и `torchvision` намеренно
исключены из package dependencies, чтобы обычный PyPI не заменил CUDA wheel.
После этого устанавливается extra `jupyter`, регистрируется kernel
`diatom-dino`, а `notebooks/public/00_environment.ipynb` проверяет активный
интерпретатор, GPU и VRAM. Долгие jobs запускаются как subprocess этого kernel.
Запрос `device=cuda` является fail-fast и никогда молча не переключается на CPU.
Инструкция находится в `docs/LOCAL_GPU.md`.

## Closed-set inference

`DecisionLogic(open_set=False)` сохраняет иерархию: сначала выбирается род,
затем вид только среди представителей выбранного рода. Threshold rejection не
используется. Режим `open_set=True` оставлен в библиотечном коде для обратной
совместимости, но не входит в актуальные configs.

E2E tester больше не требует NII и формирует `overall/*` плюс отдельный набор
метрик для каждого реально присутствующего `source_cohort` (в основной версии
это `gunduz/*`).

## Правила изменений

- Сначала валидировать все архивы и IDs.
- Не изменять и не удалять готовые dataset versions автоматически.
- Builder должен быть атомарным и оставлять `.building` при ошибке для аудита.
- После materialization проверять соответствие images, labels, objects и crops.
- Любое изменение схемы данных одновременно отражать здесь и в `data/README.md`.
- Для GitHub не добавлять содержимое `data/`, `artifacts/`, `outputs/`, `work/`.

## Порядок запуска

Основной порядок в JupyterLab:

```text
00_environment -> 01_prepare_data -> 02_train_detector ->
03_train_classifier -> 04_retrieval_benchmark -> 05_e2e_benchmark
```

CLI остаётся эквивалентным интерфейсом:

```bash
python -m scripts.prepare_data all --config configs/data.yaml --dry-run
python -m scripts.prepare_data all --config configs/data.yaml
python -m scripts.run_train_detector --config configs/detector.yaml
python -m scripts.run_train_classifier --config configs/classifier.yaml
python -m scripts.run_build_gallery --config configs/classifier_benchmark.yaml
python -m scripts.run_test_classifier --config configs/classifier_benchmark.yaml
python -m scripts.run_test_supermodel --config configs/inference.yaml
```
