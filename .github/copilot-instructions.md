
# Copilot Instructions for FontDiffuser

## Project Overview
FontDiffuser is a neural network toolkit for font style transfer and generation using diffusion models. It supports both single and batch character processing, with optimizations for efficient inference and memory usage. The toolkit is designed for research and production workflows, with a focus on reproducibility and extensibility.

## Architecture & Data Flow
- **configs/**: Argument parsers and configuration logic (see fontdiffuser.py). All scripts share a common parser for consistency.
- **src/**: Core model architectures and pipeline (model.py, build_optimized.py). Model code must remain backward compatible with checkpoint formats.
- **utils.py**: Utility functions for font/image handling, LRU caching, normalization, and image saving.
- **filename_utils.py**: Hash-based file naming for generated images ensures uniqueness and traceability.
- **sample_optimized.py**: Main entry for optimized inference (single or batch). Handles device, precision, and memory optimizations.
- **train.py**: Training entrypoint, using Accelerate for distributed/mixed-precision training. Arguments are sourced from configs/fontdiffuser.py.
- **dataset/**: Custom dataset and collate logic for font data. Integrates with PyTorch DataLoader.
- **ckpt/**: Pretrained model checkpoints (required for inference). Place all weights here and specify with --ckpt_dir.

## Developer Workflows
- **Optimized Inference:**
  - Use sample_optimized.py for most inference tasks.
  - Example: `python sample_optimized.py --ckpt_dir ckpt --content_character "A" --style_image_path path/to/style.png --save_image --save_image_dir results/`
  - For batch: add `--character_input` and `--batch_size` (see README for details).
- **Training:**
  - Use train.py with arguments from configs/fontdiffuser.py.
  - Distributed/mixed-precision training via Accelerate.
- **Gradio Demo:**
  - gradio_app.py provides a web UI for interactive testing and rapid prototyping.

## Project-Specific Patterns
- **Optimizations:**
  - All optimizations (FP16, xformers, channels_last, torch.compile) are safe and do not alter model weights/outputs.
  - Model building in src/build_optimized.py must remain backward compatible with checkpoint architectures.
- **Caching:**
  - LRU caching is used for font loading, character checks, and image transforms (see utils.py).
- **File Naming:**
  - Generated images use hash-based filenames (see filename_utils.py) for reproducibility and deduplication.
- **Flexible Input:**
  - Supports both character and image input for content/style. Input type is auto-detected by argument presence.

## Conventions & Integration
- **Argument Parsing:**
  - All scripts use a shared parser from configs/fontdiffuser.py for consistency and maintainability.
- **Data Organization:**
  - Font and image data are organized under data/ and fonts/. Generated results are saved in user-specified directories.
- **Checkpoints:**
  - Place pretrained weights in ckpt/ and specify with --ckpt_dir. Checkpoint names must match expected architecture.
- **External Dependencies:**
  - Key: diffusers, xformers, torch, Pillow, fontTools, gradio, accelerate. See requirements.txt for full list.

## Examples & Further Guidance
- See README.md for usage, command examples, and troubleshooting.
- For new scripts, follow the structure and argument patterns in sample_optimized.py and train.py.
- For batch processing, see the batch example in README.md and sample_optimized.py.
- For interactive testing, use gradio_app.py.
- Remember to declare modern type hints with lowercase types (list, tuple, set, dict,...) for functions and classes definitions.
- Use torch.tensor instead of torch.Tensor when declaring types for PyTorch tensors.
- When converting from numpy arrays to PyTorch tensors, use memory efficient methods like torch.from_numpy() instead of torch.tensor().
---
For further details, consult the README or open an issue for project-specific questions.
