# Copilot Instructions for FontDiffuser

## Project Overview
FontDiffuser is a neural network toolkit for font style transfer and generation using diffusion models. It supports both single and batch character processing, with optimizations for efficient inference and memory usage.

## Key Components & Structure
- **configs/**: Argument parsers and configuration logic (see `fontdiffuser.py`).
- **src/**: Core model architectures and pipeline (e.g., `model.py`, `build_optimized.py`).
- **utils.py**: Utility functions for font/image handling, saving, and normalization.
- **sample_optimized.py**: Main entry for optimized inference and batch processing.
- **train.py**: Training entrypoint, using `Accelerate` for distributed training.
- **dataset/**: Custom dataset and collate logic for font data.
- **ckpt/**: Pretrained model checkpoints (required for inference).

## Developer Workflows
- **Optimized Inference:**
  - Use `sample_optimized.py` for most inference tasks.
  - Example: `python sample_optimized.py --ckpt_dir ckpt --content_character "A" --style_image_path path/to/style.png --save_image --save_image_dir results/`
  - For batch: add `--character_input` and `--batch_size`.
- **Training:**
  - Use `train.py` with arguments from `configs/fontdiffuser.py`.
  - Distributed/mixed-precision training via `Accelerate`.
- **Gradio Demo:**
  - `gradio_app.py` provides a web UI for interactive testing.

## Project-Specific Patterns
- **Optimizations:**
  - All optimizations (FP16, xformers, channels_last) are safe and do not alter model weights/outputs.
  - Model building in `src/build_optimized.py` must remain backward compatible with checkpoint architectures.
- **Caching:**
  - LRU caching is used for font loading and image transforms (see `utils.py`).
- **File Naming:**
  - Hash-based file naming for generated images (see `filename_utils.py`).
- **Flexible Input:**
  - Supports both character and image input for content/style.

## Conventions & Integration
- **Argument Parsing:**
  - All scripts use a shared parser from `configs/fontdiffuser.py`.
- **Data:**
  - Font and image data are organized under `data/` and `fonts/`.
- **Checkpoints:**
  - Place pretrained weights in `ckpt/` and specify with `--ckpt_dir`.
- **External Dependencies:**
  - Key: `diffusers`, `xformers`, `torch`, `Pillow`, `fontTools`, `gradio`, `accelerate`.

## Examples
- See `README.md` for usage and command examples.
- For new scripts, follow the structure and argument patterns in `sample_optimized.py` and `train.py`.

---
For further details, consult the README or open an issue for project-specific questions.
