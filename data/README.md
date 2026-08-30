# Runtime data

This directory is populated by `python -m scripts.prepare_data all` and is not
committed to Git. Only this README is versioned.

```text
data/
├── raw/archives/                 # immutable downloaded ZIP files
├── datasetDiatom/
│   ├── detector/images/          # Gunduz full microscope images
│   ├── detector/labels/          # YOLO bbox labels
│   ├── classifier/crops/
│   │   ├── gunduz/               # benchmark support/query crops only
│   │   ├── ude/                  # classifier optimization pool
│   │   ├── diatom1042/           # classifier optimization pool
│   │   └── siyue_pu/             # classifier optimization pool
│   └── manifests/                # images.csv, objects.csv, crops.csv, audit
└── splits/
    ├── detector/                 # Gunduz train/val/test and data.yaml
    ├── classifier/               # public-source train/val and canonical codec
    └── benchmark/gunduz/         # gallery/query + unseen-class subsets
```

NII data is deliberately unsupported by this public training pipeline.
