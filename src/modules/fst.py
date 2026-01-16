import math
from einops import rearrange, repeat

import torch
import torch.nn as nn


class FontStyleTransformationModule(nn.Module):
    """
    The FST module that learns the transformation between source and target font styles.
    Implements equations (3)-(9) from the FSTDiff paper (Section 3.2).

    Args:
        feature_channels: Number of channels in the input style features (c_i).
        num_queries: Number of learnable queries (N_L, paper uses 256).
        query_dim: Dimension of each query vector (d, paper uses 128).
        num_scale_features: Number of multi-scale features (n_s, paper uses 5).
        num_cross_attn_blocks: Number of transformer blocks for cross-attention (paper uses 2).
        num_self_attn_blocks: Number of transformer blocks for self-attention fusion (paper uses 2).
    """

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

        # Learnable query vectors L ∈ R^{N_L × d} (Eq. in text before Eq. 3)
        self.learnable_queries = nn.Parameter(torch.randn(num_queries, query_dim))

        # Per-scale learnable positional encodings PE_i (Section 3.2, before Eq. 5)
        self.pos_encodings = nn.ParameterList(
            [
                nn.Parameter(torch.randn(1, ch, 1, 1))  # For adding to feature maps
                for ch in feature_channels
            ]
        )

        # Per-scale weight matrices for query, key, value projections
        # W_i^Q ∈ R^{d × d}, W_i^K ∈ R^{c_i × d}, W_i^V ∈ R^{c_i × d} (Eq. 4 & 6)
        self.q_projs = nn.ModuleList(
            [nn.Linear(query_dim, query_dim) for _ in range(num_scale_features)]
        )
        self.k_projs = nn.ModuleList(
            [nn.Linear(ch, query_dim) for ch in feature_channels]
        )
        self.v_projs = nn.ModuleList(
            [nn.Linear(ch, query_dim) for ch in feature_channels]
        )

        # Multi-layer Transformer blocks for cross-attention (paper uses 2)
        self.cross_attn_blocks = nn.ModuleList(
            [
                TransformerBlock(dim=query_dim, num_heads=8, is_cross_attention=True)
                for _ in range(num_cross_attn_blocks)
            ]
        )

        # Multi-layer Transformer blocks for self-attention fusion (paper uses 2)
        self.self_attn_blocks = nn.ModuleList(
            [
                TransformerBlock(dim=query_dim, num_heads=8, is_cross_attention=False)
                for _ in range(num_self_attn_blocks)
            ]
        )

        # MLP to adjust channel size to c_{n_s} (paper uses 1024) after concatenation
        # and weight matrix W for the residual connection (Eq. 9)
        total_concat_dim = (
            query_dim * num_scale_features
        )  # After concatenating all L_{x→y}^i
        self.mlp_channel_adjust = nn.Sequential(
            nn.Linear(total_concat_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
        )
        self.residual_proj = nn.Linear(feature_channels[-1], 1024)  # W in Eq. 9

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
        batch_size = source_features[0].shape[0]
        queries = repeat(self.learnable_queries, "n d -> b n d", b=batch_size)

        all_transformed = []

        # Process each scale i
        for i, (f_src, f_tgt) in enumerate(zip(source_features, target_features)):
            # Add learnable positional encoding (before Eq. 5)
            pe = self.pos_encodings[i]
            f_src = f_src + pe
            f_tgt = f_tgt + pe

            # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
            f_src_flat = rearrange(f_src, "b c h w -> b (h w) c")
            f_tgt_flat = rearrange(f_tgt, "b c h w -> b (h w) c")

            # Project for attention: Q_i = L W_i^Q, K_i = f^{s,i} W_i^K, V_i = f^{s,i} W_i^V
            Q = self.q_projs[i](queries)  # (B, N_L, d)
            K_src = self.k_projs[i](f_src_flat)  # (B, H*W, d)
            V_src = self.v_projs[i](f_src_flat)
            K_tgt = self.k_projs[i](f_tgt_flat)
            V_tgt = self.v_projs[i](f_tgt_flat)

            # Cross-attention blocks to extract style features (Eq. 3 & 5)
            L_src = self._apply_cross_attention(Q, K_src, V_src)  # L_{x_r}^i
            L_tgt = self._apply_cross_attention(Q, K_tgt, V_tgt)  # L_{y_r}^i

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

    def _apply_cross_attention(self, Q, K, V):
        """Apply cross-attention blocks."""
        x = Q
        for block in self.cross_attn_blocks:
            x = block(x, context=K, value=V)
        return x


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
