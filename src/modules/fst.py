import math
from einops import rearrange, repeat

import torch
import torch.nn as nn
from torch.nn import functional as F

class CrossAttentionBlock(nn.Module):
    """Memory-efficient cross-attention with gradient checkpointing."""
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
        self.scale: float = self.head_dim ** -0.5
        
        # Project K/V to query_dim ONCE (not per-head)
        self.to_q: nn.Linear = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k: nn.Linear = nn.Linear(key_dim, query_dim, bias=False)
        self.to_v: nn.Linear = nn.Linear(value_dim, query_dim, bias=False)
        
        self.proj_out: nn.Linear = nn.Linear(query_dim, query_dim)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        
        # Use Flash Attention if available
        self.use_flash_attn: bool = use_flash_attn and hasattr(F, 'scaled_dot_product_attention')
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B, N_q, C = query.shape
        _, N_kv, _ = key.shape
        
        # Project
        q: torch.Tensor = self.to_q(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k: torch.Tensor = self.to_k(key).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v: torch.Tensor = self.to_v(value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.use_flash_attn:
            # Use PyTorch 2.0+ Flash Attention (faster + less memory)
            out = F.scaled_dot_product_attention(
                q, k, v, 
                dropout_p=self.dropout.p if self.training else 0.0,
                scale=self.scale
            )
        else:
            # Standard attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.dropout(attn)
            out = attn @ v
        
        out = out.transpose(1, 2).reshape(B, N_q, C)
        return self.proj_out(out)

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

class AdaptivePositionalEncoding(nn.Module):
    """Learnable positional encoding with spatial awareness."""
    def __init__(self, channels: int, max_h: int = 48, max_w: int = 48):
        super().__init__()
        self.channels = channels
        
        # Learnable embeddings for height and width
        self.height_embed = nn.Parameter(torch.randn(max_h, channels // 2))
        self.width_embed = nn.Parameter(torch.randn(max_w, channels // 2))
        
        # Optional: Add learned scale factor
        self.scale = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W)"""
        B, C, H, W = x.shape

        # Interpolate height embedding: (max_h, C//2) -> (H, C//2)
        h_embed = F.interpolate(
            self.height_embed.unsqueeze(0).unsqueeze(0),  # (1, 1, max_h, C//2)
            size=(H, C // 2),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)  # (H, C//2)

        # Interpolate width embedding: (max_w, C//2) -> (W, C//2)
        w_embed = F.interpolate(
            self.width_embed.unsqueeze(0).unsqueeze(0),  # (1, 1, max_w, C//2)
            size=(W, C // 2),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)  # (W, C//2)

        # Broadcast to spatial grid
        # h_embed: (H, C//2) -> (1, H, 1, C//2)
        # w_embed: (W, C//2) -> (1, 1, W, C//2)
        h_embed_spatial = h_embed.unsqueeze(0).unsqueeze(2)  # (1, H, 1, C//2)
        w_embed_spatial = w_embed.unsqueeze(0).unsqueeze(1)  # (1, 1, W, C//2)
        
        # Concatenate and expand: (1, H, W, C)
        pos_embed = torch.cat(
            [
                h_embed_spatial.expand(B, H, W, C // 2),
                w_embed_spatial.expand(B, H, W, C // 2),
            ],
            dim=-1,
        ).permute(0, 3, 1, 2)  # (B, C, H, W)

        return x + self.scale * pos_embed
    
class FontStyleTransformationModule(nn.Module):
    def __init__(
        self,
        msse_output_channels: list[int],  # Rename for clarity
        num_queries: int = 220,  # 220 + 36 = 256 total (paper spec)
        query_dim: int = 128,
        num_cross_attn_blocks: int = 2,
        num_self_attn_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.num_queries = num_queries
        self.query_dim = query_dim
        self.num_scales = len(msse_output_channels)
        self.msse_channels = msse_output_channels  # [64, 128, 256, 512, 1024]
        
        # Learnable queries (N_L = 220, per paper)
        self.learnable_queries = nn.Parameter(torch.randn(num_queries, query_dim))
        
        # Update FST to use this
        self.pos_encodings = nn.ModuleList([
            AdaptivePositionalEncoding(ch) 
            for ch in msse_output_channels
        ])        
        # Cross-attention blocks per scale
        self.cross_attn_blocks = nn.ModuleList([
            nn.ModuleList([
                CrossAttentionBlock(
                    query_dim=query_dim,
                    key_dim=ch,
                    value_dim=ch,
                    num_heads=8 if query_dim >= 128 else 4
                )
                for _ in range(num_cross_attn_blocks)
            ])
            for ch in msse_output_channels
        ])
        
        # Self-attention for fusion (operates on concatenated features)
        fusion_dim = query_dim * self.num_scales  # 128 * 5 = 640
        self.self_attn_blocks = nn.ModuleList([
            SelfAttentionBlock(dim=fusion_dim)
            for _ in range(num_self_attn_blocks)
        ])
        
        # Project to final dimension (matches last scale: 1024)
        final_dim = msse_output_channels[-1]
        self.projection = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, final_dim * 4),    # Wider
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(final_dim * 4, final_dim * 2), # Additional layer
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(final_dim * 2, final_dim),
        )        
        # Residual connection projection
        self.residual_proj = nn.Sequential(
            nn.LayerNorm(final_dim),
            nn.Linear(final_dim, final_dim)
        )
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
        assert len(source_features) == len(target_features) == self.num_scales
        assert len(source_features) == len(self.msse_channels)

        batch_size = source_features[0].shape[0]
        queries = repeat(self.learnable_queries, "n d -> b n d", b=batch_size)

        all_transformed = []

        # Process each scale i
        for i, (f_src, f_tgt) in enumerate(zip(source_features, target_features)):
            # Validate feature dimensions match expected channels
            expected_channels = self.msse_channels[i]
            actual_channels = f_src.shape[1]

            if actual_channels != expected_channels:
                raise ValueError(
                    f"Scale {i}: Expected {expected_channels} channels, got {actual_channels}. "
                    f"Source shape: {f_src.shape}, Target shape: {f_tgt.shape}"
                )

            # Add learnable positional encoding (before Eq. 5)
            pe = self.pos_encodings[i]
            f_src = f_src + pe(f_src)  # Call pe as a module
            f_tgt = f_tgt + pe(f_tgt)  # Call pe as a module

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
        L_transformed = self.projection(L_concat)  # (B, N_L, 1024)

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
            nn.Linear(dim, dim * 4), 
            nn.GELU(), 
            nn.Linear(dim * 4, dim)
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

    def __init__(self, dim: int, num_heads: int = 8, use_flash_attn: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_flash_attn = use_flash_attn and hasattr(F, 'scaled_dot_product_attention')
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.to_qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, num_heads, N, head_dim)
        
        if self.use_flash_attn:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.0,
                scale=self.scale
            )  # (B, num_heads, N, head_dim)
            out = out.transpose(1, 2).reshape(B, N, C)  # FIX: Add reshape
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        return self.proj(out)

class CrossAttention(nn.Module):
    """Multi-head cross-attention."""

    def __init__(self, dim: int, num_heads: int = 8, use_flash_attn: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.use_flash_attn = use_flash_attn and hasattr(F, 'scaled_dot_product_attention')

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
        )  # (B, num_heads, N, head_dim)
        k = (
            self.to_k(context)
            .reshape(B, M, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )  # (B, num_heads, M, head_dim)
        v = (
            self.to_v(value)
            .reshape(B, M, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )  # (B, num_heads, M, head_dim)
        
        if self.use_flash_attn:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.0,
                scale=self.scale
            )  # (B, num_heads, N, head_dim)
            out = out.transpose(1, 2).reshape(B, N, C)  # FIX: Add reshape
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        return self.proj(out)