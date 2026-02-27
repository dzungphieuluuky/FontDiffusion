Step 1: Dataset Load
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
content_img from font_dataset_fst.py
├─ Shape: (B, 3, 96, 96)  [RGB from PIL Image.convert('RGB')]
└─ Range: [0, 1] normalized

Step 2: SkeletonDistanceTransform (if enabled in data pipeline)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  (B, 3, 96, 96) RGB
        ↓
        Convert to grayscale (use first channel or average)
        (B, 1, 96, 96)
        ↓
        Binary threshold at 0.5
        (B, 1, 96, 96) binary
        ↓
        ┌─────────────────────────────────┐
        │ Skeletonization (medial_axis)   │
        │ Extract 1-pixel medial axis     │
        └─────────────────────────────────┘
        (B, 1, 96, 96) skeleton
        ↓
        ┌─────────────────────────────────┐
        │ Distance Field (hybrid method)   │
        │ EDT + Gaussian smoothing        │
        └─────────────────────────────────┘
        (B, 1, 96, 96) distance
        ↓
        Stack: [skeleton, distance]
Output: (B, 2, 96, 96)  [dual-channel]
        Channel 0: Skeleton (binary topology)
        Channel 1: Distance (influence map)

Step 3: DualChannelContentEncoder wrapper
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  (B, 2, 96, 96)  skeleton-distance pair
        ↓
        Check: num_channels == 2?  YES
        ↓
        ┌──────────────────────────────────────────────────┐
        │ Fusion Layer (1x1 Conv2d(2, 1, kernel_size=1))  │
        │ Learns optimal blend of skeleton + distance     │
        │ Output: fused = Conv2d(skeleton, distance)      │
        └──────────────────────────────────────────────────┘
        (B, 1, 96, 96)  single fused channel
        ↓
        ┌──────────────────────────────────────────────────┐
        │ RGB Expansion (replicate single channel 3x)     │
        │ fused.repeat(1, 3, 1, 1)                        │
        └──────────────────────────────────────────────────┘
Output: (B, 3, 96, 96)  RGB-replicated for ContentEncoder

Step 4: ContentEncoder forward
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Input:  (B, 3, 96, 96)  [from Step 3]
        ↓
        DBlock 1: (B, 3, 96, 96) → SNConv2d(3→64) → (B, 64, 96, 96)
                                   ↓ AvgPool2d(2) ↓
                                   (B, 64, 48, 48)  [save to residuals]
        ↓
        DBlock 2: (B, 64, 48, 48) → SNConv2d(64→128) → (B, 128, 48, 48)
                                     ↓ AvgPool2d(2) ↓
                                     (B, 128, 24, 24)  [save to residuals]
        ↓
        DBlock 3: (B, 128, 24, 24) → SNConv2d(128→256) → (B, 256, 24, 24)
                                      ↓ AvgPool2d(2) ↓
                                      (B, 256, 12, 12)  [save to residuals]
        ↓
Output: h = (B, 256, 12, 12)  [final feature map]
        residual_features = [(B, 3, 96, 96), (B, 64, 48, 48), (B, 128, 24, 24), (B, 256, 12, 12)]

Step 5: FontDiffuserWithFST.forward usage
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Line 339 in model.py:
content_img_feature, content_residual_features = self.content_encoder(content_img)

Receives: content_img from dataset (could be 2-channel if skeleton-transformed)
Returns:  content_img_feature (B, 256, 12, 12)
          content_residual_features [list of 4 tensors with multi-scale features]

Step 6: U-Net Conditioning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
encoder_hidden_states[1] = content_residual_features

Used for cross-attention at multiple scales in diffusion U-Net:
  ├─ Resolution 12×12: Attends to (B, 256, 12, 12)
  ├─ Resolution 24×24: Attends to (B, 128, 24, 24)
  ├─ Resolution 48×48: Attends to (B, 64, 48, 48)
  └─ Resolution 96×96: Attends to (B, 3, 96, 96)