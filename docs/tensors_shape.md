Input Tensors:
---------------
- noisy_latents:         (B, 4, H, W)         # Noisy latent representations
- timestep:              (B,) or scalar       # Diffusion timestep
- content_img:           (B, 1, 96, 96)       # Source font character image
- style_source_img:      (B, 1, 96, 96)       # Reference char in source font
- style_target_img:      (B, 1, 96, 96)       # Same char in target font

Pipeline:
=========

1. Content Encoding
-------------------
content_img_feature, content_residual_features = ContentEncoder(content_img)
    - content_img_feature:         (B, C1, H1, W1)
    - content_residual_features:   list of (B, Cx, Hx, Wx)
style_content_feature, style_content_res_features = ContentEncoder(style_target_img)
    - style_content_feature:       (B, C1, H1, W1)
    - style_content_res_features:  list of (B, Cx, Hx, Wx)

2. Original Style Encoding
--------------------------
orig_style_feat, orig_style_vec, orig_style_residuals = StyleEncoder(style_target_img)
    - orig_style_feat:             (B, C2, H2, W2)
    - orig_style_vec:              (B, D)           # Style vector
    - orig_style_residuals:        list of (B, Cx, Hx, Wx)

3. Multi-Scale Style Encoding
-----------------------------
source_style_features = MSSE(style_source_img)
    - source_style_features:       (B, N, D)
target_style_features = MSSE(style_target_img)
    - target_style_features:       (B, N, D)

4. Font Style Transformation
----------------------------
transformation_features = FST(source_style_features, target_style_features)
    - transformation_features:     (B, N, D)

5. Prepare U-Net Conditions
---------------------------
fst_condition = FSTProjection(transformation_features)
    - fst_condition:               (B, N, D')
orig_style_projected = OriginalStyleProjection(orig_style_vec).unsqueeze(1)
    - orig_style_projected:        (B, 1, D')
combined_style_condition = torch.cat([fst_condition, orig_style_projected], dim=1)
    - combined_style_condition:    (B, N+1, D')

6. Prepare Encoder Hidden States
--------------------------------
encoder_hidden_states = [
    orig_style_feat,               # (B, C2, H2, W2)
    content_residual_features,     # list of (B, Cx, Hx, Wx)
    combined_style_condition,      # (B, N+1, D')
    style_content_res_features     # list of (B, Cx, Hx, Wx)
]

7. Diffusion U-Net Forward
--------------------------
noise_pred, offset_out_sum = DiffusionUNet(
    noisy_latents,                 # (B, 4, H, W)
    timestep,                      # (B,) or scalar
    encoder_hidden_states,         # see above
    content_encoder_downsample_size
)
    - noise_pred:                  (B, 4, H, W)
    - offset_out_sum:              (B, 4, H, W)

Output:
-------
{
    "noise_pred": noise_pred,                          # (B, 4, H, W)
    "offset_out_sum": offset_out_sum,                  # (B, 4, H, W)
    "content_features": content_img_feature,           # (B, C1, H1, W1)
    "transformation_features": transformation_features,# (B, N, D)
    "fst_condition": fst_condition,                    # (B, N, D')
    "source_style_features": source_style_features,    # (B, N, D)
    "target_style_features": target_style_features,    # (B, N, D)
    "orig_style_feat": orig_style_feat,                # (B, C2, H2, W2)
    "orig_style_vec": orig_style_vec                   # (B, D)
}