# FontDiffuser

Diffusion-based font style transfer and generation toolkit with optimized inference, batch processing, and reproducible, hash-named outputs.

## Features
- Font style transfer (content ↔ style)
- Optimized inference: FP16, xformers, channels_last, torch.compile
- Batch & distributed sampling pipelines
- LRU caching for fonts/transforms
- Hash-based filenames and single source of truth (`results_checkpoint.json`)
- Tools for dataset creation, validation, and export

## Quick setup (Windows / Linux / Colab / Kaggle)
1. Clone:
```bash
git clone https://github.com/dzungphieuluuky/FontDiffusion.git
cd FontDiffusion
```
2. Install dependencies (uses `uv` wrapper if available):
```bash
uv pip install -r requirements.txt
```

Colab notebook: https://colab.research.google.com/github/dzungphieuluuky/FontDiffusion/blob/main/font_diffusion.ipynb  
Kaggle notebook: https://www.kaggle.com/code/dzung271828/font-diffusion


## Repo layout
- configs/ — central CLI parser (`configs/fontdiffuser.py`)
- src/ — model implementations and optimized builds
  - src/modules/ — attention, encoders, UNet, SCR modules
- inference/ — sampling pipelines
  - sample_optimized.py (single-GPU), sample_batch.py, sample_distributed.py
- training/ — training orchestration and trainer implementations
  - trainer.py, trainer_fst.py, config.py
- dataset/ — dataset classes and collate logic
- tools/ — dataset creation, validation, filename utilities, exports
- scripts/ — helper shell/batch scripts for common workflows
- ckpt/ — place pretrained checkpoints here
- results_checkpoint.json — canonical metadata for generated datasets

## Usage examples

### Unified inference entrypoint (recommended)
Use run_inference.py to route to optimized, batch, or distributed sampling while reusing the central parser.
- Single-image optimized:
```bash
python run_inference.py --mode sample_optimized --ckpt_dir ckpt/ \
  --content_character "A" --style_image_path path/to/style.png \
  --save_image --save_image_dir results/
```

- Batch generation (single GPU):
```bash
python run_inference.py --mode sample_batch --ckpt_dir ckpt/ \
  --characters chars.txt --style_images styles/ --ttf_path fonts/default.ttf \
  --output_dir results/ --batch_size 8 --fp16 --compile
```

- Distributed multi-GPU generation (Accelerate):
```bash
accelerate launch run_inference.py --mode sample_distributed --ckpt_dir ckpt/ \
  --characters chars.txt --style_images styles/ --ttf_path "fonts/*.ttf" \
  --output_dir my_dataset/ --batch_size 4 --save_interval 10 --use_wandb
```

### Create Hugging Face dataset from generated outputs
- Non-streaming (loads images into memory; simple):
```bash
python tools/create_hf_dataset.py --data-dir ./results/ --repo-id username/fontdiffuser-ds \
  --split train --no-push --local-save ./hf_cache/
```

- Streaming mode (memory-efficient; use on Colab / Kaggle):
```bash
python tools/create_hf_dataset_streaming.py --data-dir ./results/ --repo-id username/fontdiffuser-ds \
  --split train --no-push --local-save ./hf_cache/ --batch-size 200
```

### Export a Hugging Face dataset back to FontDiffusion layout
- Sequential / simple export from local dataset:
```bash
python tools/export_hf_dataset_to_disk.py --output-dir ./exported/ --local-path ./hf_cache/ --split train
```

- High-performance parallel export from Hub or local:
```bash
python tools/export_hf_dataset_to_disk_parallel.py --output-dir ./exported_parallel/ \
  --repo-id username/fontdiffuser-ds --split train --workers 8 --batch-size 1000
```

Notes:
- All commands share CLI flags defined in `configs/fontdiffuser.py`; prefer the unified entrypoint (`run_inference.py`) for inference.
- Use streaming dataset creation when working in low‑RAM environments (Colab / Kaggle).

## Conventions & Notes (must-follow)
- Centralized CLI: always use `configs/fontdiffuser.py` — do not duplicate parsers.
- Single source of truth for dataset metadata: `results_checkpoint.json`.
- Filename hashing: use tools/filename_utils.py for deterministic names.
- Preserve checkpoint formats; migrations require explicit scripts.
- Use utilities in tools/ for IO, hashing, and validation.

## Development & Debugging
- Use `tools/diagnose_dataset.py` to validate dataset integrity before training.
- If training raises FileNotFoundError for images, ensure `results_checkpoint.json` matches files on disk and that create-validation/copy steps did not drop files.
- For dataset image extensions, code accepts both `.png` and `.jpg` where utilities are used.

## Scripts
- scripts/train_phase_1.sh, scripts/train_phase_2.sh — example training runs
- scripts/run_batch_example.sh — example batch inference

## Acknowledgements
```
@misc{FontDiffuser,
  title={FontDiffuser: One-Shot Font Generation via Denoising Diffusion with Multi-Scale Content Aggregation and Style Contrastive Learning},
  author={Zhenhua Yang, Dezhi Peng, Yuxin Kong, Yuyi Zhang, Cong Yao, Lianwen Jin},
  year={2023},
  url={https://github.com/yeungchenwa/FontDiffuser}
}
```

## License
MIT
