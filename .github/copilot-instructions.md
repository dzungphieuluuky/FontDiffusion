

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
- **Type Hints:** Use lowercase types (list, dict, torch.Tensor, etc.) everywhere. Prefer torch.from_numpy() for numpy→tensor.
- **File Naming:** All generated images use hash-based filenames (see tools/filename_utils.py) for deduplication and traceability.
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
