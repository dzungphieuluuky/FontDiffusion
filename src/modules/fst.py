import math
from einops import rearrange, repeat

import torch
import torch.nn as nn


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with projection layers for different dimensions."""

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim**-0.5

        # Projections for queries, keys, and values
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(key_dim, query_dim)  # Project to query_dim
        self.to_v = nn.Linear(value_dim, query_dim)  # Project to query_dim

        self.proj_out = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout)

        # Layer normalization
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_kv = nn.LayerNorm(key_dim)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, query_dim)
            key: (B, N_kv, key_dim)
            value: (B, N_kv, value_dim)
        Returns:
            (B, N_q, query_dim)
        """
        B, N_q, _ = query.shape
        _, N_kv, _ = key.shape

        # Normalize inputs
        query = self.norm_q(query)
        key = self.norm_kv(key)

        # Project to q, k, v
        q = (
            self.to_q(query)
            .reshape(B, N_q, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.to_k(key)
            .reshape(B, N_kv, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.to_v(value)
            .reshape(B, N_kv, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention: (B, num_heads, N_q, head_dim) @ (B, num_heads, head_dim, N_kv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N_q, -1)
        out = self.proj_out(out)

        return out


class SelfAttentionBlock(nn.Module):
    """Self-attention block for feature fusion."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.dropout = nn.Dropout(dropout)

        # Feed-forward network
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, dim)
        Returns:
            (B, N, dim)
        """
        # Self-attention with residual
        x = x + self.dropout(self.attn(self.norm(x)))

        # FFN with residual
        x = x + self.ffn(self.norm_ffn(x))

        return x


class FontStyleTransformationModule(nn.Module):
    def __init__(
        self,
        feature_channels: list[int],  # List of c_i for each scale i
        num_queries: int = 256,
        query_dim: int = 128,
        num_scale_features: int = 5,
        num_cross_attn_blocks: int = 2,
        num_self_attn_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.query_dim = query_dim
        self.num_scale_features = num_scale_features
        self.feature_channels = feature_channels

        # Learnable query vectors L ∈ R^{N_L × d} (Eq. in text before Eq. 3)
        self.learnable_queries = nn.Parameter(torch.randn(num_queries, query_dim))

        # Per-scale learnable positional encodings PE_i (Section 3.2, before Eq. 5)
        # Create positional encodings that match the actual feature channels
        self.pos_encodings = nn.ParameterList()
        for i, ch in enumerate(feature_channels):
            # Create positional encoding with correct channel dimension
            self.pos_encodings.append(nn.Parameter(torch.randn(1, ch, 1, 1)))

        # Cross-attention blocks for each scale (Eq. 5)
        self.cross_attn_blocks = nn.ModuleList()
        for ch in feature_channels:
            blocks = nn.ModuleList()
            for _ in range(num_cross_attn_blocks):
                blocks.append(
                    CrossAttentionBlock(query_dim=query_dim, key_dim=ch, value_dim=ch)
                )
            self.cross_attn_blocks.append(blocks)

        # Self-attention blocks for fusion (Eq. 7)
        self.self_attn_blocks = nn.ModuleList()
        concat_dim = query_dim * num_scale_features
        for _ in range(num_self_attn_blocks):
            self.self_attn_blocks.append(SelfAttentionBlock(dim=concat_dim))

        # MLP to adjust concatenated features to final dimension
        final_dim = feature_channels[-1]  # Use last scale's channel count
        self.mlp_channel_adjust = nn.Sequential(
            nn.Linear(concat_dim, final_dim * 2),
            nn.ReLU(),
            nn.Linear(final_dim * 2, final_dim),
        )

        # Projection for residual connection (Eq. 9)
        self.residual_proj = nn.Linear(final_dim, final_dim)

    def forward(
        self,
        source_features: list[torch.Tensor],  # f_{x_r}^s = [f^{s,1}, ..., f^{s,n_s}]
        target_features: list[torch.Tensor],  # f_{y_r}^s = [f^{s,1}, ..., f^{s,n_s}]
    ) -> torch.Tensor:
        """
        Computes the font style transformation representation L_{x→y}^r.

        Returns:
            Style transformation features of shape (B, N_L + h_{n_s}*w_{n_s}, c_{n_s})
        """
        # Validate input dimensions
        assert len(source_features) == len(target_features) == self.num_scale_features
        assert len(source_features) == len(self.feature_channels)

        batch_size = source_features[0].shape[0]
        queries = repeat(self.learnable_queries, "n d -> b n d", b=batch_size)

        all_transformed = []

        # Process each scale i
        for i, (f_src, f_tgt) in enumerate(zip(source_features, target_features)):
            # Validate feature dimensions match expected channels
            expected_channels = self.feature_channels[i]
            actual_channels = f_src.shape[1]

            if actual_channels != expected_channels:
                raise ValueError(
                    f"Scale {i}: Expected {expected_channels} channels, got {actual_channels}. "
                    f"Source shape: {f_src.shape}, Target shape: {f_tgt.shape}"
                )

            # Add learnable positional encoding (before Eq. 5)
            pe = self.pos_encodings[i]
            f_src = f_src + pe
            f_tgt = f_tgt + pe

            # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
            f_src_flat = rearrange(f_src, "b c h w -> b (h w) c")
            f_tgt_flat = rearrange(f_tgt, "b c h w -> b (h w) c")

            # Project for attention: Q_i = L W_i^Q, K_i = f^{s,i} W_i^K, V_i = f^{s,i} W_i^V
            Q = queries  # (B, N_L, d)
            K_src, V_src = f_src_flat, f_src_flat  # (B, H*W, c_i)
            K_tgt, V_tgt = f_tgt_flat, f_tgt_flat  # (B, H*W, c_i)

            # Cross-attention: L_{x_r}^i = CrossAttn(Q_i, K_{x_r}^i, V_{x_r}^i) (Eq. 5)
            L_src = self._apply_cross_attention(i, Q, K_src, V_src)

            # Cross-attention: L_{y_r}^i = CrossAttn(Q_i, K_{y_r}^i, V_{y_r}^i) (Eq. 6)
            L_tgt = self._apply_cross_attention(i, Q, K_tgt, V_tgt)

            # Compute difference L_{x→y}^i = L_{y_r}^i - L_{x_r}^i (Eq. 7)
            L_diff = L_tgt - L_src  # (B, N_L, d)
            all_transformed.append(L_diff)

        # Concatenate all scales: [L_{x→y}^1; L_{x→y}^2; ...] (Eq. 7)
        L_concat = torch.cat(all_transformed, dim=-1)  # (B, N_L, d * n_s)

        # Self-attention-based fusion (Eq. 7)
        for block in self.self_attn_blocks:
            L_concat = block(L_concat)

        # MLP to adjust channel size to c_{n_s} (1024)
        L_transformed = self.mlp_channel_adjust(L_concat)  # (B, N_L, 1024)

        # Residual connection: concatenate with last-scale target feature (Eq. 9)
        last_feature = target_features[-1]  # f_{y_r}^{s,n_s}
        last_feature_flat = rearrange(last_feature, "b c h w -> b (h w) c")
        last_feature_proj = self.residual_proj(last_feature_flat)  # (B, H*W, 1024)

        # Final output L_{x→y}^r = [L_{x→y}; f_{y_r}^{s,T} W] (Eq. 9)
        output = torch.cat(
            [L_transformed, last_feature_proj], dim=1
        )  # (B, N_L + H*W, 1024)

        return output

    def _apply_cross_attention(self, scale_idx: int, Q, K, V):
        """Apply cross-attention blocks for the specified scale."""
        blocks = self.cross_attn_blocks[scale_idx]

        result = Q
        for block in blocks:
            result = result + block(result, K, V)  # Residual connection
        return result


class TransformerBlock(nn.Module):
    """A single transformer block with optional cross-attention."""

    def __init__(self, dim: int, num_heads: int = 8, is_cross_attention: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.is_cross_attention = is_cross_attention

        if is_cross_attention:
            self.attn = CrossAttention(dim, num_heads)
        else:
            self.attn = SelfAttention(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor = None, value: torch.Tensor = None
    ):
        # Self-attention or cross-attention
        if self.is_cross_attention and context is not None:
            x = x + self.attn(self.norm1(x), context, value)
        else:
            x = x + self.attn(self.norm1(x))

        # FFN
        x = x + self.ffn(self.norm2(x))
        return x


class SelfAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.to_qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Multi-head cross-attention."""

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, value: torch.Tensor = None
    ) -> torch.Tensor:
        if value is None:
            value = context

        B, N, C = x.shape
        _, M, _ = context.shape

        q = (
            self.to_q(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.to_k(context)
            .reshape(B, M, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.to_v(value)
            .reshape(B, M, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)
