"""
Multi-Scale Frequency Decomposition (MSDF) for Content-Style Disentanglement
Decomposes images into frequency bands to physically separate content (low-freq) from style (high-freq).
"""

import torch
import torch.nn as nn
import torch.fft
import numpy as np
from typing import Tuple, Optional, Dict, Literal
import matplotlib.pyplot as plt

class FrequencyDecomposition(nn.Module):
    """
    Decomposes images into low (structure), mid (thickness), and high (texture) frequency bands.
    """
    def __init__(self, image_size: int = 96, low_cutoff: float = 0.10, mid_cutoff: float = 0.40,
                 filter_type: Literal["ideal", "butterworth", "gaussian"] = "gaussian", 
                 normalize_bands: bool = True, return_fft: bool = False):
        super().__init__()
        self.normalize_bands, self.return_fft = normalize_bands, return_fft
        
        # Create frequency grid and filters
        center = image_size // 2
        y, x = np.ogrid[:image_size, :image_size]
        dist = np.sqrt((x - center)**2 + (y - center)**2)
        max_freq = image_size / 2.0
        
        def _get_filter(cutoff, high=False):
            if cutoff == 0: return np.zeros_like(dist)
            if filter_type == "ideal": mask = (dist <= cutoff).astype(float)
            elif filter_type == "butterworth": mask = 1.0 / (1.0 + (dist / cutoff)**4)
            else: mask = np.exp(-(dist**2) / (2 * (cutoff / 2.0)**2)) # Gaussian
            return 1.0 - mask if high else mask

        low_px, mid_px = low_cutoff * max_freq, mid_cutoff * max_freq
        low_f = _get_filter(low_px)
        high_f = _get_filter(mid_px, high=True)
        mid_f = _get_filter(low_px, high=True) - high_f # Bandpass: HP(low) - HP(mid)
        
        for n, f in zip(['low_filter', 'mid_filter', 'high_filter'], [low_f, mid_f, high_f]):
            self.register_buffer(n, torch.from_numpy(f).float())

    def _apply_filter(self, img, fltr):
        fft = torch.fft.fftshift(torch.fft.fft2(img, dim=(-2, -1)), dim=(-2, -1)) * fltr
        out = torch.fft.ifft2(torch.fft.ifftshift(fft, dim=(-2, -1)), dim=(-2, -1)).real
        return out, (torch.abs(fft) if self.return_fft else None)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        low, l_fft = self._apply_filter(x, self.low_filter)
        mid, m_fft = self._apply_filter(x, self.mid_filter)
        high, h_fft = self._apply_filter(x, self.high_filter)
        
        if self.normalize_bands:
            # Vectorized normalization per image in batch
            def norm(b):
                mi, ma = b.amin(dim=(1,2,3), keepdim=True), b.amax(dim=(1,2,3), keepdim=True)
                return (b - mi) / (ma - mi + 1e-8)
            low, mid, high = norm(low), norm(mid), norm(high)
            
        out = {"low_freq": low, "mid_freq": mid, "high_freq": high}
        if self.return_fft: out.update({"low_fft": l_fft, "mid_fft": m_fft, "high_fft": h_fft})
        return out

class MultiScaleFrequencyEncoder(nn.Module):
    """Wrapper that processes content/style through frequency decomposition."""
    def __init__(self, content_encoder: nn.Module, style_encoder: nn.Module, image_size: int = 96,
                 low_cutoff: float = 0.10, mid_cutoff: float = 0.40, use_mid_band: bool = True, 
                 mid_band_target: Literal["content", "style", "both"] = "both"):
        super().__init__()
        self.content_encoder, self.style_encoder = content_encoder, style_encoder
        self.use_mid_band, self.target = use_mid_band, mid_band_target
        self.freq_decomp = FrequencyDecomposition(image_size, low_cutoff, mid_cutoff)
        
        # Fusion layers (1x1 conv to combine bands)
        if use_mid_band:
            if mid_band_target in ["content", "both"]: self.content_fusion = nn.Conv2d(2, 1, 1, bias=False)
            if mid_band_target in ["style", "both"]: self.style_fusion = nn.Conv2d(2, 1, 1, bias=False)

    def forward(self, content_image: torch.Tensor, style_image: Optional[torch.Tensor] = None) -> Tuple:
        c_bands = self.freq_decomp(content_image)
        
        # Prepare content input (low [+ mid])
        c_in = torch.cat([c_bands["low_freq"], c_bands["mid_freq"]], 1) if self.use_mid_band and self.target in ["content", "both"] else c_bands["low_freq"]
        if hasattr(self, 'content_fusion'): c_in = self.content_fusion(c_in)
        c_feat = self.content_encoder(c_in)
        
        if style_image is None: return c_feat
        
        s_bands = self.freq_decomp(style_image)
        # Prepare style input (high [+ mid])
        s_in = torch.cat([s_bands["mid_freq"], s_bands["high_freq"]], 1) if self.use_mid_band and self.target in ["style", "both"] else s_bands["high_freq"]
        if hasattr(self, 'style_fusion'): s_in = self.style_fusion(s_in)
        return c_feat, self.style_encoder(s_in)

def visualize_frequency_decomposition(orig, low, mid, high, save_path=None):
    """Visualize spatial and frequency domains."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    imgs, titles = [orig, low, mid, high], ['Original', 'Low (Structure)', 'Mid (Thickness)', 'High (Details)']
    
    for i, (im, t) in enumerate(zip(imgs, titles)):
        axes[0, i].imshow(im, cmap='gray'); axes[0, i].set_title(t); axes[0, i].axis('off')
        axes[1, i].imshow(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(im)))), cmap='hot')
        axes[1, i].set_title(f'{t} FFT'); axes[1, i].axis('off')
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else: plt.show()
    plt.close()

def batch_visualize_frequency_decomposition(images: torch.Tensor, decomp_module: FrequencyDecomposition, num_samples=4, save_dir=None):
    from pathlib import Path
    if save_dir: Path(save_dir).mkdir(parents=True, exist_ok=True)
    bands = decomp_module(images)
    for i in range(min(num_samples, images.shape[0])):
        visualize_frequency_decomposition(
            images[i,0].cpu().numpy(), bands["low_freq"][i,0].cpu().numpy(),
            bands["mid_freq"][i,0].cpu().numpy(), bands["high_freq"][i,0].cpu().numpy(),
            f"{save_dir}/freq_decomp_{i}.png" if save_dir else None)

def example_usage():
    # Setup
    freq_decomp = FrequencyDecomposition(image_size=96)
    dummy_image = torch.zeros(1, 1, 96, 96)
    
    # Create synthetic character
    dummy_image[0, 0, 20:76, 40:56] = 1.0 # Stroke
    for y in range(20, 76): # Tapering
        t = 16 - int((y-20)*0.2)
        dummy_image[0, 0, y, 48-t//2:48+t//2] = 1.0
    dummy_image[0, 0, 25:30, 42:44] = 0.5 # Detail
    
    # Run
    bands = freq_decomp(dummy_image)
    visualize_frequency_decomposition(dummy_image[0,0].numpy(), bands["low_freq"][0,0].numpy(), 
                                      bands["mid_freq"][0,0].numpy(), bands["high_freq"][0,0].numpy())
    print("Visualization complete.")

if __name__ == "__main__":
    example_usage()