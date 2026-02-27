"""
Adversarial Content-Style Discriminator (ACSD)

Forces content encoder to learn style-invariant representations through
adversarial training. A style discriminator tries to predict the source style
from content features, while the content encoder tries to fool it.

Key Concept:
    If the discriminator cannot tell which style a content image came from,
    then the content encoder has successfully removed all style information.

Inspired by:
- Domain-Adversarial Neural Networks (DANN)
- Style-agnostic feature learning
- Adversarial disentanglement

Usage:
    1. Add StyleDiscriminator to model
    2. Compute adversarial loss during training
    3. Content encoder learns style-invariant features
    4. Result: Zero style leakage from content images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training.
    
    Forward: Identity (pass through)
    Backward: Reverses gradient (multiplies by -lambda)
    
    This allows the content encoder to minimize discriminator loss
    (fool the discriminator) while discriminator maximizes it
    (correctly predict style).
    """
    
    @staticmethod
    def forward(ctx, x, lambda_value):
        ctx.lambda_value = lambda_value
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Reverse gradient
        return grad_output.neg() * ctx.lambda_value, None


class GradientReversal(nn.Module):
    """Gradient reversal layer wrapper."""
    
    def __init__(self, lambda_value: float = 1.0):
        """
        Args:
            lambda_value: Gradient reversal strength
        """
        super().__init__()
        self.lambda_value = lambda_value
    
    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_value)
    
    def set_lambda(self, lambda_value: float):
        """Update gradient reversal strength (for schedule)."""
        self.lambda_value = lambda_value


class StyleDiscriminator(nn.Module):
    """
    Style discriminator that predicts source style from content features.
    
    Architecture:
        Content Features → Global Pool → MLP → Style Logits
    
    The discriminator tries to answer: "Which style family does this
    content image belong to?" (e.g., NomNaTong, Gothic, Ming, etc.)
    
    The content encoder tries to make features indistinguishable across styles.
    """
    
    def __init__(
        self,
        input_channels: int = 256,  # Content encoder output channels
        num_styles: int = 10,  # Number of style families
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        use_spectral_norm: bool = True,
    ):
        """
        Args:
            input_channels: Number of channels from content encoder
            num_styles: Number of distinct style families to discriminate
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            use_spectral_norm: Whether to use spectral normalization (stabilizes training)
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.num_styles = num_styles
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Build MLP discriminator
        layers = []
        
        in_dim = input_channels
        for hidden_dim in hidden_dims:
            # Linear layer (with optional spectral norm)
            linear = nn.Linear(in_dim, hidden_dim)
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            
            layers.extend([
                linear,
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ])
            
            in_dim = hidden_dim
        
        # Final classification layer
        final_linear = nn.Linear(in_dim, num_styles)
        if use_spectral_norm:
            final_linear = nn.utils.spectral_norm(final_linear)
        
        layers.append(final_linear)
        
        self.discriminator = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, content_features: torch.Tensor) -> torch.Tensor:
        """
        Predict style from content features.
        
        Args:
            content_features: (B, C, H, W) - Features from content encoder
            
        Returns:
            style_logits: (B, num_styles) - Style predictions
        """
        # Global pooling
        pooled = self.global_pool(content_features)  # (B, C, 1, 1)
        pooled = pooled.flatten(1)  # (B, C)
        
        # Discriminate
        logits = self.discriminator(pooled)  # (B, num_styles)
        
        return logits


class MultiScaleStyleDiscriminator(nn.Module):
    """
    Multi-scale style discriminator for stronger disentanglement.
    
    Operates on multiple feature levels from content encoder to ensure
    style information is removed at all scales.
    """
    
    def __init__(
        self,
        input_channels_list: List[int],  # Channels at each scale
        num_styles: int = 10,
        shared_classifier: bool = True,
    ):
        """
        Args:
            input_channels_list: List of channel counts at each scale
                                 e.g., [64, 128, 256] for 3 scales
            num_styles: Number of style families
            shared_classifier: Whether to share classifier across scales
        """
        super().__init__()
        
        self.num_scales = len(input_channels_list)
        self.shared_classifier = shared_classifier
        
        # Create discriminator for each scale
        if shared_classifier:
            # Project all scales to same dimension, then shared classifier
            self.projections = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Conv2d(channels, 256, 1),
                    nn.LeakyReLU(0.2),
                )
                for channels in input_channels_list
            ])
            
            self.shared_disc = StyleDiscriminator(
                input_channels=256,
                num_styles=num_styles,
                hidden_dims=[256, 128],
            )
        
        else:
            # Separate discriminator per scale
            self.discriminators = nn.ModuleList([
                StyleDiscriminator(
                    input_channels=channels,
                    num_styles=num_styles,
                    hidden_dims=[512, 256, 128],
                )
                for channels in input_channels_list
            ])
    
    def forward(self, multi_scale_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Predict style at multiple scales.
        
        Args:
            multi_scale_features: List of (B, C_i, H_i, W_i) features
            
        Returns:
            List of (B, num_styles) logits per scale
        """
        logits_list = []
        
        if self.shared_classifier:
            for i, features in enumerate(multi_scale_features):
                projected = self.projections[i](features)
                logits = self.shared_disc(projected)
                logits_list.append(logits)
        else:
            for i, features in enumerate(multi_scale_features):
                logits = self.discriminators[i](features)
                logits_list.append(logits)
        
        return logits_list


class AdversarialContentStyleLoss(nn.Module):
    """
    Complete adversarial loss for content-style disentanglement.
    
    Components:
    1. Discriminator Loss: Correctly classify styles
    2. Adversarial Loss: Fool discriminator (make features style-invariant)
    3. Entropy Regularization: Ensure discriminator remains confident
    """
    
    def __init__(
        self,
        num_styles: int,
        adversarial_weight: float = 1.0,
        entropy_weight: float = 0.1,
        gradient_reversal_lambda: float = 1.0,
    ):
        """
        Args:
            num_styles: Number of style families
            adversarial_weight: Weight for adversarial loss
            entropy_weight: Weight for entropy regularization
            gradient_reversal_lambda: Strength of gradient reversal
        """
        super().__init__()
        
        self.num_styles = num_styles
        self.adversarial_weight = adversarial_weight
        self.entropy_weight = entropy_weight
        
        # Gradient reversal layer
        self.gradient_reversal = GradientReversal(gradient_reversal_lambda)
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
    
    def discriminator_loss(
        self,
        style_logits: torch.Tensor,
        style_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss for training discriminator to correctly classify styles.
        
        Args:
            style_logits: (B, num_styles) - Predicted style logits
            style_labels: (B,) - Ground truth style labels
            
        Returns:
            Discriminator loss (scalar)
        """
        return self.ce_loss(style_logits, style_labels)
    
    def adversarial_loss(
        self,
        style_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss for content encoder to fool discriminator.
        
        Encourages uniform distribution over styles (maximum confusion).
        
        Args:
            style_logits: (B, num_styles) - Predicted style logits
            
        Returns:
            Adversarial loss (scalar)
        """
        # Target: uniform distribution (maximum confusion)
        batch_size = style_logits.shape[0]
        uniform_target = torch.ones_like(style_logits) / self.num_styles
        
        # KL divergence from uniform
        log_probs = F.log_softmax(style_logits, dim=-1)
        kl_div = F.kl_div(
            log_probs,
            uniform_target,
            reduction='batchmean',
        )
        
        return kl_div
    
    def entropy_regularization(
        self,
        style_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Entropy regularization to prevent discriminator collapse.
        
        Encourages discriminator to make confident predictions.
        
        Args:
            style_logits: (B, num_styles) - Predicted style logits
            
        Returns:
            Entropy loss (scalar)
        """
        probs = F.softmax(style_logits, dim=-1)
        log_probs = F.log_softmax(style_logits, dim=-1)
        
        # Entropy: -sum(p * log(p))
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # We want to minimize entropy (confident predictions)
        return entropy
    
    def forward(
        self,
        content_features: torch.Tensor,
        style_labels: torch.Tensor,
        discriminator: nn.Module,
        train_discriminator: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute adversarial losses.
        
        Args:
            content_features: (B, C, H, W) - Features from content encoder
            style_labels: (B,) - Ground truth style labels
            discriminator: StyleDiscriminator module
            train_discriminator: Whether to train discriminator or encoder
            
        Returns:
            Dictionary with:
                - disc_loss: Discriminator classification loss
                - adv_loss: Adversarial loss for content encoder
                - entropy_loss: Entropy regularization
                - total_loss: Combined loss
                - accuracy: Discriminator accuracy (for monitoring)
        """
        if train_discriminator:
            # Train discriminator (no gradient reversal)
            style_logits = discriminator(content_features.detach())
            
            disc_loss = self.discriminator_loss(style_logits, style_labels)
            entropy_loss = self.entropy_regularization(style_logits)
            
            total_loss = disc_loss + self.entropy_weight * entropy_loss
            
            # Calculate accuracy
            predictions = style_logits.argmax(dim=-1)
            accuracy = (predictions == style_labels).float().mean()
            
            return {
                "disc_loss": disc_loss,
                "entropy_loss": entropy_loss,
                "total_loss": total_loss,
                "accuracy": accuracy.item(),
            }
        
        else:
            # Train content encoder (with gradient reversal)
            reversed_features = self.gradient_reversal(content_features)
            style_logits = discriminator(reversed_features)
            
            adv_loss = self.adversarial_loss(style_logits)
            
            total_loss = self.adversarial_weight * adv_loss
            
            # Calculate confusion (how uniform the predictions are)
            probs = F.softmax(style_logits, dim=-1)
            max_probs = probs.max(dim=-1)[0]
            confusion = 1.0 - max_probs.mean()  # High = good (uniform distribution)
            
            return {
                "adv_loss": adv_loss,
                "total_loss": total_loss,
                "confusion": confusion.item(),
            }


class StyleLabelExtractor:
    """
    Helper to extract style labels from image filenames.
    
    Assumes filename format: style+character.png (e.g., "gothic+A.png")
    """
    
    def __init__(self, style_to_idx: Optional[Dict[str, int]] = None):
        """
        Args:
            style_to_idx: Mapping from style name to index
                         If None, will be built dynamically
        """
        self.style_to_idx = style_to_idx or {}
        self.idx_to_style = {v: k for k, v in self.style_to_idx.items()}
        self.next_idx = len(self.style_to_idx)
    
    def extract_style(self, filename: str) -> str:
        """
        Extract style name from filename.
        
        Args:
            filename: e.g., "gothic+A.png"
            
        Returns:
            Style name: e.g., "gothic"
        """
        try:
            # Try format: style+char.png
            if '+' in filename:
                style = filename.split('+')[0]
                return style
            
            # Try format: style_char.png
            if '_' in filename:
                parts = filename.split('_')
                # Assume first part is style
                return parts[0]
            
            # Fallback: use full filename without extension
            return filename.split('.')[0]
        
        except:
            return "unknown"
    
    def get_or_create_label(self, style_name: str) -> int:
        """
        Get label index for style name, creating if necessary.
        
        Args:
            style_name: Style name string
            
        Returns:
            Label index
        """
        if style_name not in self.style_to_idx:
            self.style_to_idx[style_name] = self.next_idx
            self.idx_to_style[self.next_idx] = style_name
            self.next_idx += 1
        
        return self.style_to_idx[style_name]
    
    def batch_extract_labels(self, filenames: List[str]) -> torch.Tensor:
        """
        Extract labels for a batch of filenames.
        
        Args:
            filenames: List of filenames
            
        Returns:
            Tensor of label indices
        """
        labels = []
        for filename in filenames:
            style = self.extract_style(filename)
            label = self.get_or_create_label(style)
            labels.append(label)
        
        return torch.tensor(labels, dtype=torch.long)
    
    @property
    def num_styles(self) -> int:
        """Number of distinct styles seen."""
        return len(self.style_to_idx)


def create_adversarial_content_encoder(
    original_content_encoder: nn.Module,
    num_styles: int,
    adversarial_weight: float = 0.5,
    use_multi_scale: bool = False,
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """
    Factory function to create adversarial training setup.
    
    Args:
        original_content_encoder: Original ContentEncoder module
        num_styles: Number of style families in dataset
        adversarial_weight: Weight for adversarial loss
        use_multi_scale: Whether to use multi-scale discriminator
        
    Returns:
        (content_encoder, discriminator, loss_module)
    """
    # Content encoder (no modification needed)
    content_encoder = original_content_encoder
    
    # Get output channels from content encoder
    # Assume last residual feature has the channels we need
    # This needs to be configured based on your ContentEncoder architecture
    output_channels = 256  # Default, adjust based on your architecture
    
    # Create discriminator
    if use_multi_scale:
        discriminator = MultiScaleStyleDiscriminator(
            input_channels_list=[64, 128, 256],  # Adjust to your architecture
            num_styles=num_styles,
        )
    else:
        discriminator = StyleDiscriminator(
            input_channels=output_channels,
            num_styles=num_styles,
        )
    
    # Create loss module
    loss_module = AdversarialContentStyleLoss(
        num_styles=num_styles,
        adversarial_weight=adversarial_weight,
    )
    
    return content_encoder, discriminator, loss_module


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example showing how to use adversarial content-style discriminator."""
    
    # Simulate content encoder output
    batch_size = 4
    content_features = torch.randn(batch_size, 256, 12, 12)
    
    # Style labels (which style family each image belongs to)
    style_labels = torch.tensor([0, 1, 0, 2])  # 3 different styles
    num_styles = 3
    
    # Create discriminator
    discriminator = StyleDiscriminator(
        input_channels=256,
        num_styles=num_styles,
    )
    
    # Create loss module
    loss_module = AdversarialContentStyleLoss(
        num_styles=num_styles,
        adversarial_weight=0.5,
    )
    
    # Training step for discriminator
    disc_losses = loss_module(
        content_features,
        style_labels,
        discriminator,
        train_discriminator=True,
    )
    
    print("Discriminator Training:")
    print(f"  Loss: {disc_losses['total_loss'].item():.4f}")
    print(f"  Accuracy: {disc_losses['accuracy']:.2%}")
    
    # Training step for content encoder
    adv_losses = loss_module(
        content_features,
        style_labels,
        discriminator,
        train_discriminator=False,
    )
    
    print("\nContent Encoder Training:")
    print(f"  Adversarial Loss: {adv_losses['adv_loss'].item():.4f}")
    print(f"  Confusion: {adv_losses['confusion']:.2%}")


if __name__ == "__main__":
    example_usage()