# Public notebooks

These notebooks mirror the supported local-GPU workflow from `README.md`.
They contain no outputs, credentials, dataset files, or model checkpoints.
Every download, training, gallery build, and frozen test is guarded by an
explicit `RUN_* = False` flag.

Run them in order:

1. `01_prepare_data.ipynb`
2. `02_train_detector.ipynb`
3. `03_train_classifier.ipynb`
4. `04_retrieval_benchmark.ipynb`
5. `05_e2e_benchmark.ipynb`

`notebooks/00_quickstart.ipynb` remains the compact all-in-one alternative.
