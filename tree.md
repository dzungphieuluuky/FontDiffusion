FontDiffusion/
├── configs/
│   ├── inference/
│   │   ├── batch.yaml
│   │   └── fst.yaml
│   └── training/
│       ├── default.yaml
│       └── fst.yaml
├── fonts/
├── my_dataset/
│   ├── handwritten_original/
│   ├── train/
│   └── val/
├── outputs/
│   ├── FontDiffuser/
│   │   └── phase2/
│   └── FontDiffuserFST/
│       └── phase1/
├── src/
│   ├── configs/
│   │   └── fontdiffuser.py
│   ├── dataset/
│   │   ├── collate_fn_fst.py
│   │   ├── font_dataset_fst.py
│   │   └── __init__.py
│   ├── model/
│   │   └── __init__.py
│   ├── tools/
│   │   ├── export_hf_dataset_to_disk_parallel.py
│   │   ├── filename_utils.py
│   │   ├── generate_metadata.py
│   │   ├── utilities.py
│   │   ├── utils.py
│   │   ├── clean_dataset.py
│   │   ├── create_hf_dataset_streaming.py
│   │   ├── create_validation_split.py
│   │   ├── diagnose_dataset.py
│   │   └── upload_models_hybrid.py
│   ├── trainers/
│   │   ├── trainer.py
│   │   ├── trainer_fst.py
│   │   ├── training_config.py
│   │   └── __init__.py
│   └── __init__.py
├── inference/
│   ├── sample_batch.py
│   ├── sample_fst.py
│   └── sample_optimized.py
├── ckpt/
│   ├── phase2/
│   │   └── final/
│   └── fst_model/
├── font_diffusion.ipynb
├── train.py
├── train_fst.py
└── README.md