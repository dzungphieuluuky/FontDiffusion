# Skeleton-Distance Transform Integration Guide

Complete guide for integrating the Hybrid Skeleton-Distance approach into FontDiffuser to prevent style leakage from content images.

## 📋 Problem Being Solved

**Issue**: Content images (e.g., NomNaTong) have style characteristics (tapering, stroke thickness) that leak into the generation, preventing clean style transfer.

**Solution**: Transform content images into skeleton-distance maps that preserve topology (where to draw) but remove style (how thick to draw).

---

## 🔧 File Modifications

### 1. **font_dataset_fst.py** - Add Skeleton Transform to Dataset

**Location**: `FontDataset.__init__()` and `__getitem__()`

#### Add Import (top of file):
```python
from skeleton_distance_transform import SkeletonDistanceTransform
```

#### Modify `__init__()` (around line 50-80):

```python
def __init__(
    self,
    args,
    phase: str,
    transforms: Optional[list] = None,
    scr: bool = False,
    use_fst: bool = False,
    style_source_same_prob: float = 0.5,
    use_skeleton_transform: bool = False,  # ADD THIS
    skeleton_config: Optional[dict] = None,  # ADD THIS
):
    # ... existing code ...
    
    # ADD SKELETON TRANSFORM CONFIGURATION
    self.use_skeleton_transform = use_skeleton_transform
    
    if self.use_skeleton_transform:
        # Default configuration
        default_config = {
            "method": "medial_axis",
            "distance_method": "hybrid",
            "max_distance": 10.0,
            "sigma": 3.0,
            "output_mode": "dual_channel",
            "normalize": True,
        }
        
        # Update with user config
        if skeleton_config:
            default_config.update(skeleton_config)
        
        # Create transform
        self.skeleton_transform = SkeletonDistanceTransform(**default_config)
        logger.info(f"Skeleton transform enabled: {default_config}")
    else:
        self.skeleton_transform = None
```

#### Modify `__getitem__()` (around line 150-200):

```python
def __getitem__(self, index: int) -> dict:
    # ... existing code to load images ...
    
    # Original content image loading
    content_image = Image.open(content_image_path).convert("RGB")
    
    # Apply transforms
    if self.transforms is not None:
        content_image = self.transforms[0](content_image)  # (C, H, W) tensor
    
    # ADD SKELETON TRANSFORM
    if self.use_skeleton_transform:
        # Apply skeleton-distance transform
        # Input: (C, H, W), Output: (C_out, H, W) where C_out = 1 or 2
        content_image_skeleton = self.skeleton_transform(content_image.unsqueeze(0))
        content_image_skeleton = content_image_skeleton.squeeze(0)
        
        # Store both original and skeleton version
        sample["content_image"] = content_image_skeleton  # Use skeleton for training
        sample["content_image_original"] = content_image  # Keep original for reference
    else:
        sample["content_image"] = content_image
    
    # ... rest of existing code ...
    
    return sample
```

---

### 2. **collate_fn_fst.py** - Handle Variable Channel Content Images

**Location**: `CollateFN.__call__()`

#### Modify `__call__()` (around line 40-75):

```python
def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
    result = {}
    
    # Content images - handle variable channels
    content_images = [item["content_image"] for item in batch]
    
    # Check if all have same number of channels
    num_channels = content_images[0].shape[0]
    
    # Stack content images
    result["content_image"] = torch.stack(content_images)  # (B, C, H, W)
    
    # Store channel info for model
    result["content_num_channels"] = num_channels
    
    # If skeleton transform was used, also batch the original
    if "content_image_original" in batch[0]:
        result["content_image_original"] = torch.stack([
            item["content_image_original"] for item in batch
        ])
    
    # ... rest of existing collation ...
    
    return result
```

---

### 3. **model.py** - Modify Content Encoder Input

**Location**: `FontDiffuserWithFST.__init__()` and `forward()`

#### Add Import (top of file):
```python
from skeleton_distance_transform import DualChannelContentEncoder
```

#### Modify `__init__()` (around line 100-150):

```python
def __init__(
    self,
    unet: UNet,
    content_encoder: ContentEncoder,
    style_encoder: StyleEncoder,
    vae: AutoencoderKL,
    noise_scheduler: DDPMScheduler,
    args,
    use_skeleton_content: bool = False,  # ADD THIS
    skeleton_fusion_method: str = "concat",  # ADD THIS
):
    super().__init__()
    
    # Store skeleton config
    self.use_skeleton_content = use_skeleton_content
    
    # Wrap content encoder if using skeleton transform
    if use_skeleton_content:
        # Wrap the original content encoder to handle 2-channel input
        self.content_encoder = DualChannelContentEncoder(
            original_encoder=content_encoder,
            fusion_method=skeleton_fusion_method,
            learnable_weights=True,  # Make fusion weights learnable
        )
        logger.info(f"Content encoder wrapped for skeleton input (fusion: {skeleton_fusion_method})")
    else:
        self.content_encoder = content_encoder
    
    # ... rest of existing initialization ...
```

#### Modify `forward()` - No changes needed!

The wrapper handles the channel conversion automatically, so the rest of the forward pass remains the same.

---

### 4. **trainer_fst.py** - Add Training Arguments

**Location**: `FontDiffuserFSTTrainer.__init__()` and `_setup_dataset()`

#### Modify `__init__()` (around line 50-100):

```python
def __init__(self, args):
    # ... existing code ...
    
    # ADD SKELETON TRANSFORM PARAMETERS
    self.use_skeleton_content = getattr(args, "use_skeleton_content", False)
    self.skeleton_method = getattr(args, "skeleton_method", "medial_axis")
    self.skeleton_distance_method = getattr(args, "skeleton_distance_method", "hybrid")
    self.skeleton_max_distance = getattr(args, "skeleton_max_distance", 10.0)
    self.skeleton_sigma = getattr(args, "skeleton_sigma", 3.0)
    self.skeleton_output_mode = getattr(args, "skeleton_output_mode", "dual_channel")
    self.skeleton_fusion_method = getattr(args, "skeleton_fusion_method", "concat")
```

#### Modify `_setup_model()` (around line 180-220):

```python
def _setup_model(self):
    # ... load content_encoder, style_encoder, etc. ...
    
    # Create model with skeleton support
    model = FontDiffuserWithFST(
        unet=unet,
        content_encoder=content_encoder,
        style_encoder=style_encoder,
        vae=vae,
        noise_scheduler=noise_scheduler,
        args=self.args,
        use_skeleton_content=self.use_skeleton_content,  # ADD THIS
        skeleton_fusion_method=self.skeleton_fusion_method,  # ADD THIS
    )
    
    return model
```

#### Modify `_setup_dataset()` (around line 220-250):

```python
def _setup_dataset(self, phase: str):
    # ... existing transform setup ...
    
    # ADD SKELETON CONFIG
    skeleton_config = None
    if self.use_skeleton_content:
        skeleton_config = {
            "method": self.skeleton_method,
            "distance_method": self.skeleton_distance_method,
            "max_distance": self.skeleton_max_distance,
            "sigma": self.skeleton_sigma,
            "output_mode": self.skeleton_output_mode,
            "normalize": True,
        }
    
    dataset = FontDatasetFST(
        args=self.args,
        phase=phase,
        transforms=transforms_list,
        scr=self.config.phase_2,
        use_fst=self.use_fst,
        style_source_same_prob=self.style_source_same_prob,
        use_skeleton_transform=self.use_skeleton_content,  # ADD THIS
        skeleton_config=skeleton_config,  # ADD THIS
    )
    
    return dataset
```

#### Modify `_setup_logging()` (around line 572-600):

```python
def _setup_logging(self):
    # ... existing logging config ...
    
    # ADD SKELETON TRANSFORM CONFIG
    if self.use_skeleton_content:
        skeleton_config = {
            "use_skeleton_content": self.use_skeleton_content,
            "skeleton_method": self.skeleton_method,
            "skeleton_distance_method": self.skeleton_distance_method,
            "skeleton_max_distance": self.skeleton_max_distance,
            "skeleton_sigma": self.skeleton_sigma,
            "skeleton_output_mode": self.skeleton_output_mode,
            "skeleton_fusion_method": self.skeleton_fusion_method,
        }
        
        logger.info(f"Skeleton Transform Config: {skeleton_config}")
        
        # Log to wandb/tensorboard if enabled
        if self.accelerator.is_main_process:
            # Add to config
            self.config_dict.update(skeleton_config)
```

---

### 5. **train_fst.py** - Add Command Line Arguments

**Location**: Argument parser setup (around line 50-150)

#### Add arguments:

```python
parser.add_argument(
    "--use_skeleton_content",
    action="store_true",
    help="Use skeleton-distance transform for content images (prevents style leakage)",
)

parser.add_argument(
    "--skeleton_method",
    type=str,
    default="medial_axis",
    choices=["skeletonize", "medial_axis", "zhang_suen"],
    help="Skeletonization algorithm",
)

parser.add_argument(
    "--skeleton_distance_method",
    type=str,
    default="hybrid",
    choices=["edt", "gaussian", "hybrid"],
    help="Distance field generation method",
)

parser.add_argument(
    "--skeleton_max_distance",
    type=float,
    default=10.0,
    help="Maximum influence radius for skeleton distance field",
)

parser.add_argument(
    "--skeleton_sigma",
    type=float,
    default=3.0,
    help="Gaussian sigma for distance field smoothing",
)

parser.add_argument(
    "--skeleton_output_mode",
    type=str,
    default="dual_channel",
    choices=["skeleton_only", "distance_only", "dual_channel"],
    help="What to output from skeleton transform",
)

parser.add_argument(
    "--skeleton_fusion_method",
    type=str,
    default="concat",
    choices=["concat", "add", "weighted"],
    help="How to fuse skeleton and distance channels in content encoder",
)
```

---

## 🚀 Usage Examples

### Example 1: Basic Training with Skeleton Transform

```bash
python train_fst.py \
    --use_fst \
    --use_skeleton_content \
    --skeleton_method medial_axis \
    --skeleton_distance_method hybrid \
    --skeleton_max_distance 10.0 \
    --other_args ...
```

This will:
1. Transform content images to skeleton-distance maps
2. Use dual-channel (skeleton + distance) input
3. Fuse channels via 1×1 convolution
4. Prevent style leakage from content images

### Example 2: Distance-Only Mode

```bash
python train_fst.py \
    --use_fst \
    --use_skeleton_content \
    --skeleton_output_mode distance_only \
    --skeleton_max_distance 8.0 \
    --other_args ...
```

This uses only the distance field (no skeleton channel).

### Example 3: Weighted Fusion

```bash
python train_fst.py \
    --use_fst \
    --use_skeleton_content \
    --skeleton_fusion_method weighted \
    --other_args ...
```

This learns optimal weights for combining skeleton and distance channels.

### Example 4: No Skeleton Transform (Default)

```bash
python train_fst.py \
    --use_fst \
    --other_args ...
```

Traditional content image input (no skeleton transform).

---

## 📊 Expected Behavior

### Before (Without Skeleton Transform)

```
Content Image: NomNaTong character
  ├─ Has tapering strokes (thick → thin)
  ├─ Has decorative flourishes
  └─ Model copies these style elements

Generated Image:
  ├─ Target style: Bold Gothic
  ├─ But still has tapering from NomNaTong ❌
  └─ Style mixing/leakage
```

### After (With Skeleton Transform)

```
Content Image → Skeleton Transform
  ├─ Extract 1-pixel medial axis (no thickness info)
  ├─ Create distance field (smooth influence)
  └─ Feed to content encoder

Generated Image:
  ├─ Target style: Bold Gothic
  ├─ Clean, uniform strokes ✓
  └─ No NomNaTong tapering ✓
```

---

## 🔍 Monitoring During Training

Watch these in your logs:

```
INFO: Skeleton Transform Config: {
    'use_skeleton_content': True,
    'skeleton_method': 'medial_axis',
    'skeleton_distance_method': 'hybrid',
    'skeleton_max_distance': 10.0,
    'skeleton_sigma': 3.0,
    'skeleton_output_mode': 'dual_channel',
    'skeleton_fusion_method': 'concat'
}

INFO: Content encoder wrapped for skeleton input (fusion: concat)

# During training
content_image shape: (B, 2, 96, 96)  ← Dual channel
content_num_channels: 2
```

---

## 🎨 Visualization

To visualize what the skeleton transform is doing:

```python
from skeleton_distance_transform import (
    SkeletonDistanceTransform,
    visualize_skeleton_distance_transform
)

# Create transform
transform = SkeletonDistanceTransform(
    method="medial_axis",
    distance_method="hybrid",
    max_distance=10.0,
    sigma=3.0,
    output_mode="dual_channel",
)

# Load a content image
from PIL import Image
import torchvision.transforms as T

img = Image.open("content_image.png").convert("L")
img_tensor = T.ToTensor()(img).unsqueeze(0)

# Transform
transformed = transform(img_tensor)

# Visualize
import numpy as np

original = img_tensor[0, 0].numpy()
skeleton = transformed[0, 0].numpy()
distance = transformed[0, 1].numpy()

visualize_skeleton_distance_transform(
    original, skeleton, distance,
    save_path="skeleton_visualization.png"
)
```

This will show:
1. Original image (with stroke thickness)
2. Skeleton (1-pixel medial axis)
3. Distance field (smooth influence map)
4. Overlay (skeleton on original)

---

## 🐛 Troubleshooting

### Issue 1: Content Images Look Wrong

**Symptom**: Generated images have no structure

**Solution**: Check skeleton transform parameters
```bash
# Try reducing max_distance
--skeleton_max_distance 5.0

# Or use skeleton_only mode
--skeleton_output_mode skeleton_only
```

### Issue 2: Shape Mismatches

**Symptom**: `RuntimeError: shape mismatch in content encoder`

**Cause**: Content encoder expects 1 channel, getting 2

**Solution**: Ensure `DualChannelContentEncoder` wrapper is active
```python
# In model.py, verify:
if use_skeleton_content:
    self.content_encoder = DualChannelContentEncoder(...)
```

### Issue 3: Still Seeing Style Leakage

**Symptom**: Generated images still copy content image style

**Possible causes**:
1. max_distance too large (captures too much original stroke info)
2. Need to use skeleton_only mode
3. Fusion weights not optimal

**Solutions**:
```bash
# Use smaller max_distance
--skeleton_max_distance 5.0

# Use skeleton only (no distance field)
--skeleton_output_mode skeleton_only

# Use learnable weighted fusion
--skeleton_fusion_method weighted
```

### Issue 4: Training Slower

**Symptom**: Training significantly slower

**Cause**: Skeleton computation is CPU-heavy (scipy operations)

**Solution**: Pre-compute skeletons offline
```python
# Create a script to pre-process all content images
python precompute_skeletons.py \
    --input_dir ./content_images \
    --output_dir ./skeleton_content \
    --config skeleton_config.json

# Then modify dataset to load pre-computed skeletons
```

---

## 📈 Performance Tips

### 1. Pre-compute Skeletons

For faster training, pre-compute skeleton transforms:

```python
# precompute_skeletons.py
from skeleton_distance_transform import SkeletonDistanceTransform
import torch
from pathlib import Path
from tqdm import tqdm

transform = SkeletonDistanceTransform(
    method="medial_axis",
    distance_method="hybrid",
    max_distance=10.0,
    output_mode="dual_channel",
)

input_dir = Path("./content_images")
output_dir = Path("./skeleton_content")
output_dir.mkdir(exist_ok=True)

for img_path in tqdm(list(input_dir.glob("*.png"))):
    # Load
    img = Image.open(img_path).convert("L")
    img_tensor = T.ToTensor()(img).unsqueeze(0)
    
    # Transform
    skeleton = transform(img_tensor)
    
    # Save
    torch.save(skeleton, output_dir / f"{img_path.stem}.pt")
```

Then modify dataset to load pre-computed:
```python
# In font_dataset_fst.py
if self.use_skeleton_transform and self.precomputed_skeleton_dir:
    skeleton_path = self.precomputed_skeleton_dir / f"{content_name}.pt"
    content_image_skeleton = torch.load(skeleton_path)
else:
    # Compute on-the-fly
    content_image_skeleton = self.skeleton_transform(content_image.unsqueeze(0))
```

### 2. Adaptive Max Distance

Use `AdaptiveSkeletonDistanceTransform` for better results:

```python
from skeleton_distance_transform import AdaptiveSkeletonDistanceTransform

transform = AdaptiveSkeletonDistanceTransform(
    min_distance=5.0,
    max_distance=15.0,
    method="medial_axis",
    distance_method="hybrid",
)
```

This automatically adjusts max_distance based on stroke thickness.

---

## 🎯 Summary

The Skeleton-Distance Transform:

1. **Prevents style leakage** - Removes thickness/tapering from content images
2. **Preserves topology** - Keeps the "where to draw" information
3. **Forces style reliance** - Model must use style encoder for thickness
4. **Easy integration** - Just 5 file modifications
5. **Configurable** - Multiple methods and parameters

**Result**: Clean style transfer without NomNaTong tapering or other content-image style artifacts!

---

## 📞 Testing Integration

After integrating, test with:

```bash
# Quick test
python train_fst.py \
    --use_fst \
    --use_skeleton_content \
    --train_batch_size 2 \
    --max_train_steps 100 \
    --output_dir ./test_skeleton

# Check logs for:
# - "Skeleton transform enabled"
# - "Content encoder wrapped for skeleton input"
# - content_image shape: (2, 2, 96, 96) ← Batch size 2, dual channel
```

Happy training! 🚀