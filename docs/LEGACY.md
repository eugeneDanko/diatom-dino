# Legacy inventory

Новая публичная версия сознательно не включает NII, ResNet artifact gate,
open-set calibration, трёхстадийную NII adaptation, stage 2.5 sweeps и прежние
S3-ориентированные dataset builders. Эти компоненты отвечали исследовательским
вопросам старой супермодели, но не нужны компактному pet-проекту.

Локально они пока сохранены, чтобы не потерять результаты экспериментов. Перед
первой публикацией их можно удалить после проверки нового цикла. К legacy
относятся:

- `dataset_builders/` целиком;
- конфигурации, кроме `data.yaml`, `detector.yaml`, `classifier.yaml`,
  `classifier_benchmark.yaml`, `inference.yaml`;
- notebooks старого curriculum, open-set, ResNet и supermodel-v1;
- migration/report/open-set/stage2.5/stage3 scripts;
- тесты, проверяющие только перечисленные протоколы.

Удалять физические `data/`, старые S3 datasets и checkpoints не требуется: они
исключены из Git. Автоматическое удаление специально не реализовано.
