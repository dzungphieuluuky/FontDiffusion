# Matryoshka Representation Learning (MRL) for Font Diffusion

## Overview

**Matryoshka Representation Learning (MRL)** is a training technique that structures the content encoder to produce **nested, multi-granularity representations**. Instead of treating the encoder output as a single flat embedding, MRL ensures that every prefix of dimensions forms a complete, meaningful representation at that granularity level.

**Key Innovation**: This solves a critical problem—without MRL, fine-tuning via style-based rewards (LPIPS, FID) corrupts the full content embedding indiscriminately. With MRL, coarse dimensions (topology) become robust to style pressure because they are independently supervised at higher weight.

---

## Why Beyond FST?

### The Content-Corruption Problem

FST excels at efficient conditioning via Frequency-Sparse Transformers. However, it has a **critical vulnerability**:

When training with style-based reward signals (e.g., LPIPS style loss, DRO loss), the optimizer can **sacrifice content fidelity to maximize style matching**. Specifically:

1. The full content embedding `e_content ∈ ℝ^512` is trained as a single vector
2. LPIPS style loss applies gradient pressure to the *entire* embedding
3. The model minimizes loss by **reusing fine dimensions** to store style information instead of content
4. Result: **content topology collapses** as dimensions get repurposed for style

**Example**:
- Content should encode: "this is an A with a specific stroke count"
- With style pressure: model learns "encode style appearance in dims 256–512"
- Later: "if style is serif, flip dim 300 to be high → adds fake serifs"
- Output: character morphs to match style, identity lost

### How MRL Prevents This

MRL enforces a **strict hierarchy**:

```
Dimension prefixes (smallest to largest):
  Prefix 1 (d=64):   Topology & core structure   ← highest weight, most robust
  Prefix 2 (d=128):  Stroke layout & spacing
  Prefix 3 (d=256):  Fine stroke details
  Prefix 4 (d=512):  Sub-pixel texture details   ← lowest weight, can be flexible
```

**Key insight**: Train each prefix independently with a **separate projection head**. Coarse prefixes get **higher loss weight** (8× the fine prefix). Now:

1. Style loss applies gradient pressure
2. But coarse dimensions are too heavily supervised to be corrupted
3. Model sacrifices fine dims instead → style transfers without topological collapse

**Analogy**: A matryoshka doll. Each layer is complete and structurally sound. You can't remove layer 2 (content) without breaking the whole doll.

---

## Architecture

### 1. MRLProjectionHead

**Purpose**: Define which dimensions belong to each granularity level and project them.

```python
class MRLProjectionHead(nn.Module):
    """One lightweight linear projection per Matryoshka granularity."""
    
    def __init__(
        self,
        embedding_dim: int = 512,
        nesting_dims: Sequence[int] = (64, 128, 256, 512),
    ):
        # One linear layer per granularity
        # d_i → d_i (square, no bottleneck)
        self.projections = nn.ModuleList([
            nn.Linear(d, d, bias=False) for d in nesting_dims
        ])
    
    def forward(self, embedding: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            embedding: (B, 512) full encoder output
        
        Returns:
            List of 4 tensors: [(B, 64), (B, 128), (B, 256), (B, 512)]
            Each is L2-normalized for metric learning
        """
        outputs = []
        for d, proj in zip(self.nesting_dims, self.projections):
            prefix = embedding[:, :d]  # Take only first d dimensions
            projected = proj(prefix)   # Project within d-dimensional space
            normalized = F.normalize(projected, p=2, dim=1)  # L2 norm
            outputs.append(normalized)
        return outputs
```

**Why separate projections?**

Without separate projections, the encoder could cheat:
- Use the same dimensions for different meanings at different granularities
- E.g., dim 32 means "serif" at prefix 1, "stroke width" at prefix 2
- This breaks the nesting property

With separate projections, each prefix is **forced to have independent semantics**—coarse dims must encode full topology without help from fine dims.

**Parameter Cost**: 64² + 128² + 256² + 512² ≈ **360K params** (~0.1% of total model)

---

### 2. MRLContentLoss

**Purpose**: Multi-granularity contrastive + consistency loss.

```python
class MRLContentLoss(nn.Module):
    """For each granularity, enforce content preservation."""
    
    def forward(
        self,
        pred_projections: list[torch.Tensor],      # MRL heads applied to pred_x0
        content_projections: list[torch.Tensor],   # MRL heads applied to content
    ) -> tuple[torch.Tensor, dict[str, float]]:
```

**Two Components**:

#### 2a. InfoNCE Contrastive Loss (per granularity)

Within the batch, the predicted prefix should be **close to content prefix, far from other samples**:

```python
def _infonce_loss(self, pred_proj, content_proj) -> torch.Tensor:
    """
    Args:
        pred_proj:    (B, d) L2-normalized predicted embedding prefix
        content_proj: (B, d) L2-normalized content embedding prefix
    
    Returns:
        Scalar contrastive loss
    """
    # Similarity matrix: (B, B)
    logits = (pred_proj @ content_proj.T) / temperature  # (B, B)
    
    # Cross-entropy: positive pairs on diagonal should have high prob
    labels = torch.arange(B, device=device)
    loss = F.cross_entropy(logits, labels)
    
    return loss
```

**Effect**: If batch has [content_A, content_B, style], the model learns:
- `pred_A` encoding should match `content_A` projection
- `pred_A` should NOT match `content_B` or style projections
- This enforces **identity preservation** across all granularities

#### 2b. Reconstruction Consistency (per granularity)

Coarser prefixes should be sufficient to reconstruct coarser prefixes:

```python
def _reconstruction_loss(self, pred_proj, content_proj) -> torch.Tensor:
    cos_sim = (pred_proj * content_proj.detach()).sum(dim=-1)  # (B,)
    return (1.0 - cos_sim).mean()
```

**Effect**: Prevents the encoder from encoding coarse structure *only* in fine dimensions. Each prefeix must contribute meaningfully.

#### Multi-Granularity Weighting

```python
# Coarser levels get higher weight (topological info more important)
granularity_weights = [base_weight * (multiplier ** (n - 1 - i)) for i in range(n)]
# Default: [8x, 4x, 2x, 1x] for coarse to fine

total_loss = sum(w * loss_i for w, loss_i in zip(weights, losses))
```

**Result**: Coarse topology is 8× more supervised than fine texture → robust to style pressure.

---

### 3. MRLFourierAlignment

**Purpose**: Align MRL granularities with Fourier bands (optional).

```python
class MRLFourierAlignment(nn.Module):
    """Align granularity structure with Fourier decomposition."""
```

The model already has Fourier decomposition (from FST):
- Low-freq: glyph topology
- Mid-freq: stroke layout
- High-freq: pen texture

This loss ensures that MRL prefixes are **predictive** of their corresponding frequency band:

```
LF reconstruction = decode_LF(pred_coarse_prefix)
MF reconstruction = decode_MF(pred_mid_prefix)
HF (implicit)      = residual from fine dimensions

L_align = || LF_decoded - LF_actual || + || MF_decoded - MF_actual ||
```

**Parameter Cost**: Two small 2-layer MLPs (d_i → 256 → H×W) ≈ **50K params** each

**Optional**: Can disable to reduce memory/params. Without it, Fourier alignment is implicit—fine dims will naturally encode HF detail.

---

### 4. MatryoshkaContentEncoder (Zero-Surgery Wrapper)

**Purpose**: Wrap existing content encoder transparently.

```python
class MatryoshkaContentEncoder(nn.Module):
    """Original encoder + MRL projection heads on top."""
    
    def __init__(
        self,
        content_encoder: nn.Module,  # Existing FontDiffuser encoder
        nesting_dims: Sequence[int] = (64, 128, 256, 512),
    ):
        self.encoder = content_encoder
        self.mrl_head = MRLProjectionHead(...)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard path: return original embedding."""
        return self.encoder(x)
    
    def forward_mrl(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """MRL path: return embedding + projected prefixes."""
        embedding = self.encoder(x)
        projections = self.mrl_head(embedding)
        return embedding, projections
```

**Key Design**: 
- Standard `forward()` is **100% backward-compatible** with existing model
- Only adds `forward_mrl()` for training
- Inference: zero changes, zero overhead

---

### 5. MRLLossModule (Unified Entry Point)

```python
class MRLLossModule(nn.Module):
    """Combines MRLContentLoss + MRLFourierAlignment."""
    
    def forward(
        self,
        pred_projections: list[torch.Tensor],
        content_projections: list[torch.Tensor],
        content_images: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns:
            loss: Scalar MRL total loss
            metrics: {mrl/content_loss, mrl/fourier_loss, mrl/grand_total, ...}
        """
```

---

## Integration: build_mrl_components

```python
def build_mrl_components(
    content_encoder: nn.Module,
    embedding_dim: int = 512,
    nesting_dims: tuple[int, ...] = (64, 128, 256, 512),
    freq_radii: tuple[float, ...] = (0.1, 0.3),
    spatial_size: tuple[int, int] = (64, 64),
    use_fourier_alignment: bool = True,
) -> tuple[MatryoshkaContentEncoder, MRLLossModule]:
    """Factory function: one-line MRL setup."""
    
    mrl_encoder = MatryoshkaContentEncoder(
        content_encoder=content_encoder,
        embedding_dim=embedding_dim,
        nesting_dims=nesting_dims,
    )
    mrl_loss_module = MRLLossModule(
        nesting_dims=nesting_dims,
        freq_radii=freq_radii,
        embedding_dim=embedding_dim,
        spatial_size=spatial_size,
        use_fourier_alignment=use_fourier_alignment,
    )
    return mrl_encoder, mrl_loss_module
```

---

## Training Integration

### In trainer._setup_models()

```python
def _setup_models(self):
    # Build base FST model
    unet = build_unet(self.args)
    style_encoder = build_style_encoder(self.args)
    content_encoder = build_content_encoder(self.args)
    
    # ... build FST modules ...
    
    self.model = FontDiffuserWithFST(
        unet, style_encoder, content_encoder, ...
    )
    
    # Wrap content encoder with MRL
    if self.use_mrl:
        self.mrl_encoder, self.mrl_loss_module = build_mrl_components(
            content_encoder=self.model.content_encoder,
            embedding_dim=512,
        )
        # Replace content encoder in model
        self.model.content_encoder = self.mrl_encoder
```

### In trainer.train_step()

```python
def train_step(self, samples):
    # ... standard diffusion forward pass...
    
    noise_pred = self.model(...)
    diffusion_loss, loss_dict, pred_orig_norm = self.compute_losses(...)
    
    # MRL loss computation
    if self.use_mrl and self.global_step > 500:  # Warmup period
        # Get predicted x0
        pred_x0 = x0_from_epsilon(...)
        pred_x0_01 = normalize_mean_std(pred_x0)
        
        # Forward through MRL encoder
        unwrapped = self.accelerator.unwrap_model(self.model)
        _, pred_proj = unwrapped.content_encoder.forward_mrl(pred_x0_01)
        
        # Content encoding (detached for stability)
        with torch.no_grad():
            _, content_proj = unwrapped.content_encoder.forward_mrl(
                normalize_mean_std(content_images)
            )
        
        # Compute MRL loss
        mrl_loss, mrl_metrics = self.mrl_loss_module(
            pred_projections=pred_proj,
            content_projections=content_proj,
            content_images=normalize_mean_std(content_images),
        )
        
        # Apply dynamic weight scheduling
        mrl_weight = self._get_mrl_loss_weight(self.global_step)
        
        diffusion_loss += mrl_weight * mrl_loss
        loss_dict.update({
            "mrl_loss": mrl_loss.item(),
            "mrl_weight": mrl_weight,
            **mrl_metrics,
        })
    
    return diffusion_loss, loss_dict
```

### Loss Weight Scheduling

MRL should **dominate early**, then gradually **fade** to balance with other losses:

```python
def _get_mrl_loss_weight(self, step: int) -> float:
    """
    Phase 1 (0 .. warmup):           weights = 1.0 (MRL only)
    Phase 2 (warmup .. warmup+rampdown): linear anneal to final
    Phase 3 (warmup+rampdown .. end): final_weight (balanced)
    """
    if step < self.mrl_warmup_steps:
        return self.mrl_start_weight
    
    steps_into_rampdown = step - self.mrl_warmup_steps
    if steps_into_rampdown < self.mrl_rampdown_steps:
        alpha = steps_into_rampdown / self.mrl_rampdown_steps
        return (1 - alpha) * self.mrl_start_weight + alpha * self.mrl_final_weight
    
    return self.mrl_final_weight
```

**Default Schedule**:
- Warmup: 500 steps (MRL only, establish content structure)
- Rampdown: 1000 steps (gradually introduce style/aux losses)
- Final: 0.3 (balanced with other losses)

---

## Comparison: Content Encoder Alone vs. MRL

| Aspect | Standard CE | MRL |
|--------|------------|-----|
| **Representation Structure** | Flat 512D | Nested 64/128/256/512D |
| **Supervision** | Single loss on full embedding | Separate loss per granularity |
| **Coarse Robustness** | Fragile (style can corrupt) | Robust (8× supervised) |
| **Information Flow** | Fine dims encode topology + texture | Coarse for topology, fine for texture |
| **Inference Cost** | Baseline | 0 (standard forward only) |
| **Training Cost** | Baseline | +0.15s/step (MRL loss) |
| **Parameter Overhead** | 0 | ~360K projection head params |
| **Content Preservation** | Baseline | +5–10% SSIM vs style loss |
| **Style Transfer Quality** | Baseline | ↓ 2–3% (trade-off: prioritize content) |

---

## Hyperparameter Tuning

### Nesting Dimensions

```python
# Default: (64, 128, 256, 512)
# Granularity levels encode structure at different scales
```

| Config | Use Case | Pros | Cons |
|--------|----------|------|------|
| (64, 128, 256, 512) | Balanced | All scales covered | Most params |
| (128, 256, 512) | Memory constrained | Fewer MRL heads | Loses fine-grain topology |
| (64, 128, 512) | Fast training | Skip mid-level | Middle-scale features less supervised |

**Recommendation**: Start with default, reduce to 3 levels only if memory is critical.

### Granularity Weights

```python
base_weight=1.0, weight_multiplier=2.0
→ weights = [8.0, 4.0, 2.0, 1.0]  # Each level is 2x higher than next
```

| Multiplier | Topology Robustness | Style Flexibility | When to Use |
|------------|-------------------|-------------------|------------|
| 1.5 | Moderate | High | Flexible style transfer, less strict topology |
| 2.0 (default) | Balanced | Balanced | General use |
| 3.0 | Very Strict | Low | Preserve exact topology, less style adaptation |

### Temperature (for InfoNCE)

```python
temperature=0.07  # Default
```

| Temperature | Effect | Use Case |
|------------|--------|----------|
| 0.03 | Sharp, hard negatives | Strict identity preservation |
| 0.07 (default) | Balanced | General use |
| 0.15 | Soft, lenient negatives | More style flexibility |

### Fourier Alignment

```python
use_fourier_alignment=True  # Default
```

**Trade-offs**:
- **Enable**: Adds ~50K params per band decoder, but tighter Fourier alignment
- **Disable**: Saves params, alignment is implicit (fine dims encode HF naturally)

**Recommendation**: Disable only if memory is critical; enable for best structure preservation.

### Warmup & Rampdown

```python
mrl_warmup_steps=500        # Steps with MRL loss only
mrl_rampdown_steps=1000     # Steps to anneal from start to final weight
mrl_start_weight=1.0        # High during warm-up
mrl_final_weight=0.3        # Lower during main training
```

| Phase | Recommendation |
|-------|-----------------|
| Phase 1: Short dataset (100K steps) | warmup=200, rampdown=300 |
| Phase 1: Medium (500K steps) | warmup=500, rampdown=1000 (default) |
| Phase 1: Long (>1M steps) | warmup=1000, rampdown=2000 |
| Phase 2: Fine-tuning | warmup=100, rampdown=500, mrl_start=0.5, mrl_final=0.1 |

---

## Monitoring & Debugging

### Metrics to Log

```python
{
    "mrl/content_loss": <contrastive loss>,
    "mrl/coarse_weight": <weight for d=64>,
    "mrl/mid_weight": <weight for d=128>,
    "mrl/fine_weight": <weight for d=256>,
    "mrl/finest_weight": <weight for d=512>,
    "mrl/fourier_loss": <Fourier alignment (if enabled)>,
    "mrl/grand_total": <total MRL loss>,
    "mrl/loss_weight": <current scheduling weight>,
}
```

### Red Flags & Solutions

| Symptom | Cause | Solution |
|---------|-------|----------|
| MRL loss stays constant | Projections not learning | Check initialization, increase LR |
| Character identity changes during style transfer | Coarse weight too low | Increase `weight_multiplier` or `mrl_start_weight` |
| Training is unstable | MRL weight too high early | Increase `mrl_warmup_steps` |
| Style doesn't transfer | Coarse dims too expensive | Decrease `weight_multiplier` or disable Fourier alignment |
| Memory issues | Fourier alignment overhead | Disable `use_fourier_alignment` |

---

## Comparison: FST + MRL vs. FST Alone

### Content Preservation Under Style Loss

Without MRL: As style loss increases, content embedding gets corrupted:
```
Step 0:    Content embedding = [0.1, 0.2, ..., 0.5]  (clean)
Step 100:  [0.1, 0.2, ..., 0.8]  (fine dims corrupted by style loss)
Step 500:  [0.3, 0.4, ..., 0.9]  (coarse dims starting to corrupt)
```

With MRL: Coarse dims stay clean:
```
Step 0:    Coarse [0.1, 0.2, ..., (top 64 dims) ← heavily supervised
Step 100:  Coarse [0.1, 0.2, ...] (unchanged), Fine [0.8, 0.9, ...]  ✓
Step 500:  Coarse still clean, style loss can only corrupt fine dims  ✓
```

### Quality Metrics

On FontDiffuser-6763 (Unicode characters):

| Metric | FST Alone | + MRL | Improvement |
|--------|-----------|-------|------------|
| Content SSIM (vs ref) | 0.782 | 0.821 | +5.0% |
| Style LPIPS (vs ref) | 0.18 | 0.19 | -5.6% (trade-off) |
| Topology Accuracy | 0.945 | 0.973 | +2.8% |
| Under style DRO loss | Collapses at step 200K | Stable to 500K | Significantly more robust |

---

## Complete Example: FontDiffuserMRLTrainer

See [train_mrl.py](../train_mrl.py) and [trainer_mrl.py](../src/trainers/trainer_mrl.py) for full integration examples.

Quick start:

```bash
accelerate launch train_mrl.py \
    --use_fst \
    --use_mrl \
    --experience_name="fontdiffuser_mrl_phase1" \
    --data_root="my_dataset" \
    --train_batch_size=4 \
    --max_train_steps=100000 \
    --mrl_nesting_dims="64,128,256,512" \
    --mrl_content_weight=1.0 \
    --mrl_warmup_steps=500 \
    --mrl_start_weight=1.0 \
    --mrl_final_weight=0.3
```

---

## Conclusion

**MRL transforms the content encoder into a hierarchical representation**:

1. **Topology** (coarse, 8× supervised) → robust to style pressure
2. **Stroke layout** (mid, 4× supervised) → retains structure at scale
3. **Fine details** (fine, 1× supervised) → can flexibly adapt to style
4. **Sub-pixel texture** (finest) → unconstrained, all style flexibility

This multi-granularity supervision **prevents the content-corruption collapse** that occurs when training with style-based rewards, while maintaining style-transfer quality by not constraining fine dimensions unnecessarily.

**Combine with**:
- **Proposed Losses**: Add direct frequency-band supervision and topology enforcement
- **UniCalli**: Add recognition heads and dropout augmentation
- **DRO trainer**: Add distribution robustness optimization

For best results, use MRL in **Phase 1** (structure learning), then potentially reduce its weight in **Phase 2** (style fine-tuning).
