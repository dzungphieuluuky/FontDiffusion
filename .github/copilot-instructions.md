# Copilot Instructions — FontDiffuser (concise)

Purpose: help an AI coding agent get productive quickly in this repo by documenting architecture, workflows, conventions, and integration points.

- **Big picture**: The repo implements diffusion-based font style transfer.
  - Model code: `src/` (see `model.py`, `build_optimized.py`).
  - Inference pipelines: `inference/` (`sample_optimized.py`, `sample_batch.py`, `sample_distributed.py`).
  - Training: `train.py`, `train_fst.py` (training orchestration lives at repo root and `training/`).
  - Dataset & ingestion: `dataset/` + `tools/` utilities (creation, validation, export).
  - Checkpoints: `ckpt/` (all scripts expect `--ckpt_dir ckpt`).

- **Core conventions (must-follow)**
  - Centralized CLI: all scripts use the shared parser in `configs/fontdiffuser.py`. Reuse it — do not reimplement argument parsing.
  - Checkpoint compatibility: maintain backward-compatible checkpoint formats; do not rename internal layer keys silently.
  - Filename hashing: generated images and exported artifacts use hash-based names (see `tools/filename_utils.py`) — rely on these for dedup and provenance.
  - Single source of truth for generated datasets: `results_checkpoint.json` (used by dataset tools). Do not infer metadata from folder listings alone.

- **Performance & runtime patterns**
  - Performance flags like FP16, `xformers`, `channels_last`, and `torch.compile` are used and considered safe for determinism in this codebase — follow existing flags in `inference/sample_optimized.py`.
  - Multi-GPU inference uses `sample_distributed.py` / `sample_batch.py`; testing locally should prefer `sample_optimized.py`.

- **Important files & where to look first**
  - Config / CLI: `configs/fontdiffuser.py`
  - Single-GPU inference example: `inference/sample_optimized.py`
  - Training entrypoints: `train.py`, `train_fst.py`
  - Dataset creation: `tools/create_hf_dataset.py`, `tools/create_hf_dataset_streaming.py`
  - Filename helpers: `tools/filename_utils.py`
  - Checkpoints directory: `ckpt/` (contains `unet`, `style_encoder`, `content_encoder` files)

- **Typical developer tasks (examples)**
  - Run inference (single image):
    ```bash
    python inference/sample_optimized.py --ckpt_dir ckpt --content_character "A" --style_image_path style_images/foo.png --save_image --save_image_dir results/
    ```
  - Start training (single-node):
    ```bash
    python train.py --config_path configs/fontdiffuser.py --ckpt_dir ckpt
    ```
  - Create HF dataset (streaming):
    ```bash
    python tools/create_hf_dataset_streaming.py --out_dir my_dataset/ --checkpoint results_checkpoint.json
    ```

- **Patterns an agent should use when modifying code**
  - Minimal, surgical edits: prefer updating behavior via flags/config rather than broad refactors.
  - Preserve public checkpoint and filename formats — include migration steps when changing them.
  - Reuse existing utilities in `tools/` and `dataset/` for IO, hashing, and validation.

- **Dependencies & integration**
  - Key runtime deps: `torch`, `diffusers`, `xformers`, `Pillow`, `fontTools`, `accelerate`, `datasets` — check `requirements.txt`.
  - External integrations: model weights are loaded from `ckpt/`; dataset exports rely on HF dataset tooling in `tools/`.

- **When to ask the human**
  - If a change affects checkpoint naming or weight layout.
  - If a proposed change could alter file-hash output or dataset `results_checkpoint.json` schema.

If anything here is unclear or you'd like a longer, example-driven version, tell me which area to expand (inference, training, dataset tooling, or checkpoint handling).


# Copilot Instructions for FontDiffuser (2026)

## Project Overview
FontDiffuser is a modular toolkit for font style transfer and generation using diffusion models. It supports single/batch character processing, multi-GPU inference, and reproducible, hash-based outputs. The codebase is research-oriented but production-ready, with a focus on extensibility and reproducibility.

## Architecture & Data Flow
- **configs/**: Centralized argument parsing (see fontdiffuser.py). All scripts import this for consistent CLI/API.
- **src/**: Core model architectures (model.py, build_optimized.py, modules/). Model code must remain backward compatible with checkpoint formats.
- **tools/**: Utilities for dataset creation, validation, and export. E.g., create_hf_dataset.py uses results_checkpoint.json as the single source of truth.
- **inference/**: Inference pipelines (sample_optimized.py, sample_batch.py, sample_distributed.py). Use sample_optimized.py for most tasks; sample_distributed.py for multi-GPU.
- **dataset/**: Custom dataset and collate logic. FontDataset and CollateFN handle flexible input and batching.
- **ckpt/**: Pretrained model checkpoints. All inference/training expects weights here, referenced by --ckpt_dir.

## Developer Workflows
- **Inference:**
  - Use sample_optimized.py for most inference. Example:
    ```bash
    python inference/sample_optimized.py --ckpt_dir ckpt --content_character "A" --style_image_path path/to/style.png --save_image --save_image_dir results/
    ```
  - For batch/multi-GPU: use sample_batch.py or sample_distributed.py with appropriate arguments.
- **Training:**
  - Use train.py (or train_fst.py for FST) with arguments from configs/fontdiffuser.py. Distributed/mixed-precision via Accelerate.
- **Dataset Creation/Validation:**
  - Use tools/create_hf_dataset.py and tools/diagnose_dataset.py. Always use results_checkpoint.json as the ground truth for generated data.
- **Interactive Demo:**
  - gradio_app.py provides a web UI for rapid prototyping.

## Project-Specific Patterns & Conventions
- **Argument Parsing:** All scripts use the shared parser from configs/fontdiffuser.py. Never duplicate argument logic.
- **Type Hints:** Use lowercase types (list, dict, torch.Tensor, etc.) everywhere. Prefer torch.from_numpy() for numpy→tensor. Use torch.Tensor to declare tensor types and torch.tensor(...) to create tensors.
- **Caching:** LRU caching is used for font loading, image transforms, and character checks (see tools/utils.py).
- **Data Integrity:** All dataset creation/validation scripts use results_checkpoint.json as the single source of truth. Never trust directory listings alone.
- **Optimizations:** All performance flags (FP16, xformers, channels_last, torch.compile) are safe and do not alter model outputs.
- **Checkpoints:** Place all weights in ckpt/ and reference with --ckpt_dir. Checkpoint names must match model architecture.

## Integration & External Dependencies
- Key dependencies: diffusers, xformers, torch, Pillow, fontTools, gradio, accelerate, datasets. See requirements.txt for full list.
- Data is organized under data/, fonts/, and my_dataset/. Generated results are saved in user-specified directories.

## Examples & Further Guidance
- For new scripts, follow the structure and argument patterns in sample_optimized.py and train.py.
- For batch/multi-GPU, see sample_distributed.py and scripts/export_files.bat.
- For dataset creation/validation, see tools/create_hf_dataset.py and tools/diagnose_dataset.py.
- For interactive testing, use gradio_app.py.
- See README.md for usage, command examples, and troubleshooting.

---
For further details, consult the README or open an issue for project-specific questions.
