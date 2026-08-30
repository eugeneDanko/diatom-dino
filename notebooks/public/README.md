# Public notebooks

These notebooks mirror the supported local-GPU workflow from `README.md`.
They contain no outputs, credentials, dataset files, or model checkpoints.
Every download, training, gallery build, and frozen test is guarded by an
explicit `RUN_* = False` flag.

Before opening them, install and select the `Python (DiatomDINO GPU)` kernel.
Run them in order:

1. `00_environment.ipynb`
2. `01_prepare_data.ipynb`
3. `02_train_detector.ipynb`
4. `03_train_classifier.ipynb`
5. `04_retrieval_benchmark.ipynb`
6. `05_e2e_benchmark.ipynb`

`notebooks/00_quickstart.ipynb` remains the compact all-in-one alternative.
Long jobs run as child processes through `core.notebook_runtime`; the kernel
interpreter, selected GPU, cache directories, and working directory therefore
remain consistent across every stage.
