# FontDiffuserWithFST - Usage Guide

Complete guide for training and using the FST-enhanced FontDiffuser model.

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Sampling](#sampling)
- [Model Comparison](#model-comparison)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

FontDiffuserWithFST enhances the original FontDiffuser with:
- **MSSE (Multi-Scale Style Encoder)**: Extracts style features at multiple scales
- **FST (Font Style Transformation)**: Learns transformations between different font styles
- **Enhanced conditioning**: Better style control for font generation

### Architecture Comparison

```
Original FontDiffuser:
Content Encoder → Features ┐
Style Encoder → Features   ├→ U-Net → Generated Font
                          ┘

FontDiffuserWithFST:
Content Encoder → Features      ┐
Style Encoder → Features        │
MSSE → Multi-scale Features     ├→ FST → Enhanced Conditioning → U-Net → Generated Font
                                ┘
```

## 🚀 Installation

```bash
# Install required packages
pip install torch torchvision diffusers accelerate einops

# Clone the repository (if not already)
git clone <your-repo>
cd <your-repo>

# Verify installation
python -c "from model import FontDiffuserWithFST; print('Installation successful!')"
```

## ⚡ Quick Start

### 1. Test Model Components

```bash
# Run shape tests
python simple_shape_test.py

# Run comprehensive tests
python test_fontdiffuser_fst.py
```

### 2. Generate Sample with Pretrained Model

```bash
python sample_fst.py \
    --use_fst \
    --ckpt_dir="ckpt/fst_model/" \
    --character_input \
    --content_character="字" \
    --style_image_path="examples/style.jpg" \
    --save_image \
    --save_image_dir="outputs/" \
    --device="cuda:0"
```

## 🎓 Training

### Phase 1: Initial Training

Train the FST modules from scratch or fine-tune from a pretrained baseline:

```bash
python train_fst.py \
    --use_fst \
    --experience_name="fontdiffuser_fst_phase1" \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=100000 \
    --learning_rate=5e-5 \
    --ckpt_interval=5000 \
    --log_interval=100 \
    --output_dir="outputs/fst_training" \
    --style_source_ratio=0.5 \
    --mixed_precision="fp16" \
    --device="cuda:0"
```

**Key Arguments:**
- `--use_fst`: Enable FST enhancement
- `--style_source_ratio`: Probability of using different source/target style pairs (0.0-1.0)
- `--freeze_original_encoders`: Freeze pretrained encoders, only train FST modules

### Phase 2: Fine-tuning with SCR Loss

Add style contrastive learning for better style consistency:

```bash
python train_fst.py \
    --use_fst \
    --phase_2 \
    --phase_1_ckpt_dir="outputs/fst_training/global_step_100000" \
    --scr_ckpt_path="ckpt/scr.pth" \
    --experience_name="fontdiffuser_fst_phase2" \
    --train_batch_size=4 \
    --max_train_steps=50000 \
    --learning_rate=1e-5 \
    --output_dir="outputs/fst_training_phase2" \
    --freeze_original_encoders
```

### Training from Baseline Model

Start FST training from a pretrained original FontDiffuser:

```bash
python train_fst.py \
    --use_fst \
    --phase_1_ckpt_dir="ckpt/original_fontdiffuser/" \
    --experience_name="baseline_to_fst" \
    --freeze_original_encoders \
    --max_train_steps=50000 \
    --output_dir="outputs/baseline_to_fst"
```

### Training Configuration

**Recommended Settings:**

| Setting | Phase 1 | Phase 2 | Notes |
|---------|---------|---------|-------|
| Batch Size | 4-8 | 4-8 | Adjust based on GPU memory |
| Learning Rate | 5e-5 | 1e-5 | Lower for fine-tuning |
| Max Steps | 100k | 50k | Depends on dataset size |
| Mixed Precision | fp16 | fp16 | Speeds up training |
| Gradient Accumulation | 4 | 4 | Effective batch = batch_size × grad_accum |

## 🎨 Sampling

### Basic Sampling

Generate a single character:

```bash
python sample_fst.py \
    --use_fst \
    --ckpt_dir="ckpt/fst_model/" \
    --character_input \
    --content_character="隆" \
    --style_image_path="examples/style.jpg" \
    --save_image \
    --save_image_dir="outputs/samples/" \
    --num_inference_steps=20
```

### Advanced Sampling with Style Transfer

Use different source and target styles:

```bash
python sample_fst.py \
    --use_fst \
    --ckpt_dir="ckpt/fst_model/" \
    --character_input \
    --content_character="字" \
    --style_image_path="examples/style_target.jpg" \
    --style_source_image_path="examples/style_source.jpg" \
    --save_image \
    --save_image_dir="outputs/style_transfer/" \
    --guidance_scale=7.5 \
    --num_inference_steps=20
```

### Sampling with Content Image

Use an existing glyph as content:

```bash
python sample_fst.py \
    --use_fst \
    --ckpt_dir="ckpt/fst_model/" \
    --content_image_path="examples/content.jpg" \
    --style_image_path="examples/style.jpg" \
    --save_image \
    --save_image_dir="outputs/samples/"
```

### Batch Sampling Script

Create multiple samples efficiently:

```python
# batch_sample.py
import subprocess

characters = ["字", "体", "风", "格"]
style_images = ["style1.jpg", "style2.jpg"]

for char in characters:
    for style in style_images:
        cmd = [
            "python", "sample_fst.py",
            "--use_fst",
            "--ckpt_dir=ckpt/fst_model/",
            "--character_input",
            f"--content_character={char}",
            f"--style_image_path=examples/{style}",
            "--save_image",
            f"--save_image_dir=outputs/batch_{char}_{style.split('.')[0]}/",
        ]
        subprocess.run(cmd)
```

### Sampling Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `--num_inference_steps` | 20 | 10-50 | More steps = higher quality but slower |
| `--guidance_scale` | 7.5 | 1.0-15.0 | Higher = stronger style adherence |
| `--algorithm_type` | dpmsolver++ | - | Sampling algorithm |
| `--method` | multistep | singlestep/multistep | Solver method |

## 📊 Model Comparison

### Compare FST vs Baseline

```python
from fst_utils import ModelComparator

comparator = ModelComparator(device='cuda:0')

# Load both models
fst_model = load_fst_model(...)
baseline_model = load_baseline_model(...)

# Compare outputs
comparison = comparator.compare_outputs(
    fst_model,
    baseline_model,
    content_image,
    style_image
)

print(comparison)
```

### Generate Side-by-Side Comparison

```bash
# Generate with baseline
python sample.py \
    --ckpt_dir="ckpt/baseline/" \
    --character_input \
    --content_character="字" \
    --style_image_path="examples/style.jpg" \
    --save_image_dir="outputs/baseline/"

# Generate with FST
python sample_fst.py \
    --use_fst \
    --ckpt_dir="ckpt/fst_model/" \
    --character_input \
    --content_character="字" \
    --style_image_path="examples/style.jpg" \
    --save_image_dir="outputs/fst/"
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Shape Mismatch Errors

**Problem:** `RuntimeError: shape mismatch in tensor operations`

**Solution:**
```bash
# Run shape tests first
python simple_shape_test.py

# Check that input images are correct size (96x96)
```

#### 2. Out of Memory (OOM)

**Problem:** `CUDA out of memory`

**Solutions:**
- Reduce batch size: `--train_batch_size=2`
- Increase gradient accumulation: `--gradient_accumulation_steps=8`
- Use mixed precision: `--mixed_precision="fp16"`
- Reduce image resolution in config

#### 3. FST Module Not Loading

**Problem:** Missing FST checkpoint files

**Solution:**
```python
# Check if FST checkpoints exist
import os
ckpt_dir = "ckpt/fst_model/"
required_files = ["mss_encoder.pth", "fst_module.pth", "fst_projection.pth"]

for file in required_files:
    path = os.path.join(ckpt_dir, file)
    print(f"{file}: {'✓' if os.path.exists(path) else '✗'}")
```

#### 4. Slow Training

**Solutions:**
- Enable mixed precision: `--mixed_precision="fp16"`
- Use DataLoader with multiple workers
- Reduce logging frequency: `--log_interval=500`
- Profile code to identify bottlenecks

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# In training script
python train_fst.py --use_fst --log_interval=1 ...
```

### Validation

Verify model integrity:

```python
from model import FontDiffuserWithFST

# Load model
model = load_model(...)

# Check parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Test forward pass
with torch.no_grad():
    output = model(...)
    print(f"Output shape: {output['noise_pred'].shape}")
```

## 📈 Performance Tips

### Training Optimization

1. **Use Gradient Checkpointing** (if memory-limited):
```python
model.gradient_checkpointing_enable()
```

2. **Optimize DataLoader**:
```python
train_dataloader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)
```

3. **Learning Rate Scheduling**:
```bash
--lr_scheduler="cosine" \
--lr_warmup_steps=1000
```

### Inference Optimization

1. **Reduce Inference Steps**:
```bash
--num_inference_steps=15  # Instead of 20-50
```

2. **Use Compiled Model** (PyTorch 2.0+):
```python
model = torch.compile(model)
```

3. **Batch Processing**:
Process multiple characters in parallel when possible.

## 📚 Additional Resources

- **Original FontDiffuser Paper**: [Link]
- **FSTDiff Paper**: [Link]
- **Model Architecture Diagrams**: See `docs/architecture.md`
- **Training Logs Analysis**: See `docs/analysis.md`

## 🤝 Contributing

To contribute improvements:

1. Test your changes with `test_fontdiffuser_fst.py`
2. Ensure backward compatibility with baseline model
3. Document new features in this README
4. Submit pull request with clear description

## 📝 License

[Your License Here]

## 🙏 Acknowledgments

- Original FontDiffuser team
- FSTDiff paper authors
- Open-source contributors

---

**Last Updated**: January 2025
**Version**: 1.0.0