import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, repeat


class AdaIN(nn.Module):
    """Adaptive Instance Normalization.

    Aligns the mean and variance of the content features to the style features.
    Supports both spatial (B, C, H, W) and sequential (B, N, C) inputs.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Normalize content features then rescale with style statistics.

        Args:
            content: (B, C, H, W) spatial or (B, N, C) sequential tensor.
            style:   (B, C, H, W) spatial or (B, N, C) sequential tensor.

        Returns:
            Stylized content tensor with same shape as input.
        """
        is_spatial = content.dim() == 4

        if is_spatial:
            content_mean = content.mean(dim=[2, 3], keepdim=True)
            content_std = content.std(dim=[2, 3], keepdim=True) + self.eps
            style_mean = style.mean(dim=[2, 3], keepdim=True)
            style_std = style.std(dim=[2, 3], keepdim=True) + self.eps
        else:
            content_mean = content.mean(dim=1, keepdim=True)
            content_std = content.std(dim=1, keepdim=True) + self.eps
            style_mean = style.mean(dim=1, keepdim=True)
            style_std = style.std(dim=1, keepdim=True) + self.eps

        normalized = (content - content_mean) / content_std
        return normalized * style_std + style_mean


class AdaptivePositionalEncoding(nn.Module):
    """Learnable positional encoding with spatial awareness.

    Uses separate height and width embeddings that are interpolated
    to match the actual spatial dimensions at runtime.
    """

    def __init__(self, channels: int, max_h: int = 48, max_w: int = 48):
        super().__init__()
        self.channels = channels
        self.height_embed = nn.Parameter(torch.randn(max_h, channels // 2))
        self.width_embed = nn.Parameter(torch.randn(max_w, channels // 2))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input feature map.

        Args:
            x: (B, C, H, W) feature map.

        Returns:
            (B, C, H, W) feature map with positional encoding added.
        """
        B, C, H, W = x.shape

        h_embed = (
            F.interpolate(
                self.height_embed.unsqueeze(0).unsqueeze(0),
                size=(H, C // 2),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )  # (H, C//2)

        w_embed = (
            F.interpolate(
                self.width_embed.unsqueeze(0).unsqueeze(0),
                size=(W, C // 2),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )  # (W, C//2)

        h_embed_spatial = h_embed.unsqueeze(0).unsqueeze(2)  # (1, H, 1, C//2)
        w_embed_spatial = w_embed.unsqueeze(0).unsqueeze(1)  # (1, 1, W, C//2)

        pos_embed = torch.cat(
            [
                h_embed_spatial.expand(B, H, W, C // 2),
                w_embed_spatial.expand(B, H, W, C // 2),
            ],
            dim=-1,
        ).permute(0, 3, 1, 2)  # (B, C, H, W)

        return x + self.scale * pos_embed


class CrossAttentionBlock(nn.Module):
    """Memory-efficient cross-attention with optional Flash Attention."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.num_heads: int = num_heads
        self.head_dim: int = query_dim // num_heads
        self.scale: float = self.head_dim**-0.5

        self.to_q: nn.Linear = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k: nn.Linear = nn.Linear(key_dim, query_dim, bias=False)
        self.to_v: nn.Linear = nn.Linear(value_dim, query_dim, bias=False)
        self.proj_out: nn.Linear = nn.Linear(query_dim, query_dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        self.use_flash_attn: bool = use_flash_attn and hasattr(
            F, "scaled_dot_product_attention"
        )

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Apply cross-attention from query to key/value features.

        Args:
            query: (B, N_q, query_dim)
            key:   (B, N_kv, key_dim)
            value: (B, N_kv, value_dim)

        Returns:
            (B, N_q, query_dim) attended output.
        """
        B, N_q, C = query.shape
        _, N_kv, _ = key.shape

        q: torch.Tensor = (
            self.to_q(query)
            .reshape(B, N_q, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k: torch.Tensor = (
            self.to_k(key)
            .reshape(B, N_kv, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v: torch.Tensor = (
            self.to_v(value)
            .reshape(B, N_kv, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        if self.use_flash_attn:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=self.scale,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, N_q, C)
        return self.proj_out(out)


class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention block with feed-forward network."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention and FFN with residual connections.

        Args:
            x: (B, N, dim)

        Returns:
            (B, N, dim)
        """
        norm_x = self.norm(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.ffn(self.norm_ffn(x))
        return x


class FontStyleTransformationModule(nn.Module):
    """Font Style Transformation Module (FST).

    Computes a style transformation representation L_{x→y}^r by:
      1. Applying per-scale positional encodings to source and target features.
      2. Using feature-level AdaIN to align content features to style statistics.
      3. Querying aligned content and pure style with cross-attention.
      4. Using query-level AdaIN to modulate content query outputs toward style.
      5. Combining the AdaIN-stylized result with the structural difference.
      6. Fusing all scales via self-attention and projecting to the final dimension.
      7. Appending a residual from the deepest target scale features.
    """

    def __init__(
        self,
        msse_output_channels: list[int],
        num_queries: int = 220,
        query_dim: int = 128,
        num_cross_attn_blocks: int = 2,
        num_self_attn_blocks: int = 2,
    ) -> None:
        """Initialise the FST module.

        Args:
            msse_output_channels: Channel counts at each encoder scale,
                e.g. [64, 128, 256, 512, 1024].
            num_queries:          Number of learnable style queries (N_L).
            query_dim:            Dimension of each learnable query vector.
            num_cross_attn_blocks: Cross-attention blocks applied per scale.
            num_self_attn_blocks:  Self-attention blocks used during fusion.
        """
        super().__init__()
        self.num_queries = num_queries
        self.query_dim = query_dim
        self.num_scales = len(msse_output_channels)
        self.msse_channels = msse_output_channels

        # Learnable queries (N_L = 220, per paper)
        self.learnable_queries = nn.Parameter(torch.randn(num_queries, query_dim))

        # Per-scale positional encodings
        self.pos_encodings = nn.ModuleList(
            [AdaptivePositionalEncoding(ch) for ch in msse_output_channels]
        )

        # Feature-level AdaIN: aligns content feature distribution to style
        self.feature_adains = nn.ModuleList(
            [AdaIN() for _ in msse_output_channels]
        )

        # Cross-attention blocks per scale
        self.cross_attn_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        CrossAttentionBlock(
                            query_dim=query_dim,
                            key_dim=ch,
                            value_dim=ch,
                            num_heads=8 if query_dim >= 128 else 4,
                        )
                        for _ in range(num_cross_attn_blocks)
                    ]
                )
                for ch in msse_output_channels
            ]
        )

        # Query-level AdaIN: modulates content query outputs toward style statistics
        self.query_adains = nn.ModuleList(
            [AdaIN() for _ in msse_output_channels]
        )

        # Self-attention fusion over concatenated scale features
        fusion_dim = query_dim * self.num_scales
        self.self_attn_blocks = nn.ModuleList(
            [SelfAttentionBlock(dim=fusion_dim) for _ in range(num_self_attn_blocks)]
        )

        # Project fused features to the final channel dimension (= deepest scale)
        final_dim = msse_output_channels[-1]
        self.projection = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, final_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(final_dim * 4, final_dim),
        )

        # Residual projection for the deepest target-scale features
        self.residual_proj = nn.Sequential(
            nn.LayerNorm(final_dim), nn.Linear(final_dim, final_dim)
        )

    def forward(
        self,
        source_features: list[torch.Tensor],
        target_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the font style transformation representation L_{x→y}^r.

        Args:
            source_features: Per-scale content feature maps
                [(B, c_i, H_i, W_i), ...] ordered coarse→fine.
            target_features: Per-scale style feature maps
                [(B, c_i, H_i, W_i), ...] ordered coarse→fine.

        Returns:
            Style transformation tensor of shape
            (B, N_L + H_{n_s} * W_{n_s}, c_{n_s}).
        """
        assert len(source_features) == len(target_features) == self.num_scales, (
            f"Expected {self.num_scales} scales, "
            f"got source={len(source_features)}, target={len(target_features)}"
        )

        batch_size = source_features[0].shape[0]
        queries = repeat(self.learnable_queries, "n d -> b n d", b=batch_size)

        all_transformed: list[torch.Tensor] = []

        for i, (f_src, f_tgt) in enumerate(zip(source_features, target_features)):
            expected_ch = self.msse_channels[i]
            if f_src.shape[1] != expected_ch:
                raise ValueError(
                    f"Scale {i}: expected {expected_ch} channels, "
                    f"got {f_src.shape[1]}. Source: {f_src.shape}, "
                    f"Target: {f_tgt.shape}"
                )

            # A. Positional encoding
            pe = self.pos_encodings[i]
            f_src = pe(f_src)
            f_tgt = pe(f_tgt)

            # B. Feature-level AdaIN — force content to adopt style distribution
            f_src_aligned = self.feature_adains[i](f_src, f_tgt)

            # Flatten spatial dims: (B, C, H, W) -> (B, H*W, C)
            f_src_flat = rearrange(f_src_aligned, "b c h w -> b (h w) c")
            f_tgt_flat = rearrange(f_tgt, "b c h w -> b (h w) c")

            # C. Cross-attention
            # Query aligned content (structure + style statistics)
            L_content = self._apply_cross_attention(i, queries, f_src_flat, f_src_flat)
            # Query pure style reference
            L_style = self._apply_cross_attention(i, queries, f_tgt_flat, f_tgt_flat)

            # D. Query-level AdaIN — modulate content queries toward style statistics
            L_stylized = self.query_adains[i](L_content, L_style)

            # Structural difference residual for explicit style delta signal
            L_diff = L_style - L_content

            all_transformed.append(L_stylized + L_diff)

        # E. Concatenate scales and fuse with self-attention
        L_concat = torch.cat(all_transformed, dim=-1)  # (B, N_L, query_dim * n_s)

        for block in self.self_attn_blocks:
            L_concat = block(L_concat)

        L_transformed = self.projection(L_concat)  # (B, N_L, final_dim)

        # F. Residual: append projected deepest target features
        last_feature_flat = rearrange(target_features[-1], "b c h w -> b (h w) c")
        last_feature_proj = self.residual_proj(last_feature_flat)

        return torch.cat(
            [L_transformed, last_feature_proj], dim=1
        )  # (B, N_L + H*W, final_dim)

    def _apply_cross_attention(
        self,
        scale_idx: int,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the cross-attention block stack for the given scale.

        Args:
            scale_idx: Index into ``self.cross_attn_blocks``.
            Q: Query tensor  (B, N_q, query_dim).
            K: Key tensor    (B, N_kv, key_dim).
            V: Value tensor  (B, N_kv, value_dim).

        Returns:
            (B, N_q, query_dim) output with residual connections applied.
        """
        result = Q
        for block in self.cross_attn_blocks[scale_idx]:
            result = result + block(result, K, V)
        return result