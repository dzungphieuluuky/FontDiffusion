"""
Utility functions for FST model management and comparison.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import json
from pathlib import Path


class ModelCheckpointManager:
    """Manage checkpoints for FST and baseline models."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_fst_checkpoint(
        self,
        model,
        global_step: int,
        optimizer=None,
        lr_scheduler=None,
        metadata: Optional[Dict] = None,
    ):
        """Save FST model checkpoint with all components."""
        save_dir = self.base_dir / f"global_step_{global_step}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save core components
        torch.save(model.diffusion_unet.state_dict(), save_dir / "unet.pth")
        torch.save(model.style_encoder.state_dict(), save_dir / "style_encoder.pth")
        torch.save(model.content_encoder.state_dict(), save_dir / "content_encoder.pth")

        # Save FST-specific components
        torch.save(model.mss_encoder.state_dict(), save_dir / "mss_encoder.pth")
        torch.save(model.fst_module.state_dict(), save_dir / "fst_module.pth")
        torch.save(model.fst_projection.state_dict(), save_dir / "fst_projection.pth")
        torch.save(
            model.original_style_projection.state_dict(),
            save_dir / "original_style_projection.pth",
        )

        # Save full model
        torch.save(model.state_dict(), save_dir / "complete_model.pth")

        # Save optimizer and scheduler if provided
        if optimizer is not None:
            torch.save(optimizer.state_dict(), save_dir / "optimizer.pth")
        if lr_scheduler is not None:
            torch.save(lr_scheduler.state_dict(), save_dir / "lr_scheduler.pth")

        # Save metadata
        if metadata is None:
            metadata = {}
        metadata.update(
            {
                "global_step": global_step,
                "model_type": "FontDiffuserWithFST",
            }
        )

        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved FST checkpoint at step {global_step}")
        return save_dir

    def load_fst_checkpoint(
        self,
        model,
        checkpoint_dir: str,
        load_optimizer: bool = False,
        load_scheduler: bool = False,
    ) -> tuple[nn.Module, Optional[torch.optim.Optimizer], Optional[object]]:
        """Load FST model checkpoint."""
        ckpt_dir = Path(checkpoint_dir)

        # Load core components
        if (ckpt_dir / "unet.pth").exists():
            model.diffusion_unet.load_state_dict(torch.load(ckpt_dir / "unet.pth"))
        if (ckpt_dir / "style_encoder.pth").exists():
            model.style_encoder.load_state_dict(
                torch.load(ckpt_dir / "style_encoder.pth")
            )
        if (ckpt_dir / "content_encoder.pth").exists():
            model.content_encoder.load_state_dict(
                torch.load(ckpt_dir / "content_encoder.pth")
            )

        # Load FST components
        if (ckpt_dir / "mss_encoder.pth").exists():
            model.mss_encoder.load_state_dict(torch.load(ckpt_dir / "mss_encoder.pth"))
        if (ckpt_dir / "fst_module.pth").exists():
            model.fst_module.load_state_dict(torch.load(ckpt_dir / "fst_module.pth"))
        if (ckpt_dir / "fst_projection.pth").exists():
            model.fst_projection.load_state_dict(
                torch.load(ckpt_dir / "fst_projection.pth")
            )
        if (ckpt_dir / "original_style_projection.pth").exists():
            model.original_style_projection.load_state_dict(
                torch.load(ckpt_dir / "original_style_projection.pth")
            )

        optimizer = None
        lr_scheduler = None

        if load_optimizer and (ckpt_dir / "optimizer.pth").exists():
            optimizer = torch.optim.AdamW(model.parameters())  # Placeholder
            optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pth"))

        if load_scheduler and (ckpt_dir / "lr_scheduler.pth").exists():
            # This needs to be created based on your scheduler config
            pass

        print(f"Loaded FST checkpoint from {checkpoint_dir}")
        return model, optimizer, lr_scheduler

    def convert_baseline_to_fst(self, baseline_ckpt_dir: str, output_dir: str):
        """Convert baseline checkpoint to FST format (initializes FST modules randomly)."""
        from src.models.fst_model import FontDiffuserWithFST
        from src import (
            FontDiffuserModel,
            build_unet,
            build_style_encoder,
            build_content_encoder,
        )

        baseline_dir = Path(baseline_ckpt_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load baseline components
        # Note: You'll need to pass proper args here
        print("Loading baseline model...")
        # This is a simplified version - adjust based on your actual model architecture

        print("Converting to FST format...")
        # Copy baseline weights
        for file in ["unet.pth", "style_encoder.pth", "content_encoder.pth"]:
            if (baseline_dir / file).exists():
                import shutil

                shutil.copy(baseline_dir / file, output_path / file)

        print(f"Converted checkpoint saved to {output_dir}")
        print("Note: FST modules are randomly initialized. Train them before use.")


class ModelComparator:
    """Compare outputs from FST and baseline models."""

    def __init__(self, device="cuda:0"):
        self.device = device
        self.metrics = {}

    def compare_outputs(
        self,
        fst_model,
        baseline_model,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
        style_source_image: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Compare outputs from both models."""
        if style_source_image is None:
            style_source_image = style_image

        # Generate with FST model
        with torch.no_grad():
            fst_outputs = fst_model(
                noisy_latents=torch.randn(1, 4, 24, 24).to(self.device),
                timestep=torch.tensor([500]).to(self.device),
                content_img=content_image,
                style_source_img=style_source_image,
                style_target_img=style_image,
                return_dict=True,
            )

        # Generate with baseline model
        with torch.no_grad():
            baseline_outputs = baseline_model(
                x_t=torch.randn(1, 4, 24, 24).to(self.device),
                timesteps=torch.tensor([500]).to(self.device),
                style_images=style_image,
                content_images=content_image,
                content_encoder_downsample_size=4,
            )

        # Compare results
        comparison = {
            "fst_noise_pred_shape": tuple(fst_outputs["noise_pred"].shape),
            "baseline_noise_pred_shape": tuple(baseline_outputs[0].shape),
            "fst_transformation_features_shape": tuple(
                fst_outputs["transformation_features"].shape
            ),
            "fst_additional_outputs": len(fst_outputs)
            - 2,  # Excluding noise_pred and offset
        }

        return comparison

    def compute_feature_statistics(
        self, model, dataloader, num_batches: int = 10, is_fst: bool = True
    ) -> Dict:
        """Compute statistics of model features."""
        stats = {
            "content_feature_norms": [],
            "style_feature_norms": [],
        }

        if is_fst:
            stats["transformation_feature_norms"] = []

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                content_img = batch["content_image"].to(self.device)
                style_img = batch["style_image"].to(self.device)

                if is_fst:
                    outputs = model(
                        noisy_latents=torch.randn(content_img.shape[0], 4, 24, 24).to(
                            self.device
                        ),
                        timestep=torch.randint(0, 1000, (content_img.shape[0],)).to(
                            self.device
                        ),
                        content_img=content_img,
                        style_source_img=style_img,
                        style_target_img=style_img,
                        return_dict=True,
                    )

                    stats["content_feature_norms"].append(
                        outputs["content_features"].norm().item()
                    )
                    stats["transformation_feature_norms"].append(
                        outputs["transformation_features"].norm().item()
                    )

        # Compute mean and std
        for key in stats:
            if stats[key]:
                stats[key] = {
                    "mean": sum(stats[key]) / len(stats[key]),
                    "std": torch.tensor(stats[key]).std().item(),
                    "min": min(stats[key]),
                    "max": max(stats[key]),
                }

        return stats


def print_model_summary(model, model_name="Model"):
    """Print detailed model summary."""
    print(f"\n{'=' * 80}")
    print(f"{model_name} Summary")
    print(f"{'=' * 80}")

    total_params = 0
    trainable_params = 0

    print(f"\n{'Component':<40} {'Parameters':<15} {'Trainable':<10}")
    print("-" * 80)

    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)

        total_params += num_params
        trainable_params += num_trainable

        print(f"{name:<40} {num_params:>14,} {num_trainable:>9,}")

    print("-" * 80)
    print(f"{'Total':<40} {total_params:>14,} {trainable_params:>9,}")
    print(f"\nTrainable percentage: {100 * trainable_params / total_params:.2f}%")
    print("=" * 80 + "\n")


def visualize_fst_features(
    transformation_features: torch.Tensor, save_path: Optional[str] = None
):
    """Visualize FST transformation features."""
    import matplotlib.pyplot as plt
    import numpy as np

    # transformation_features: (B, N_L + H*W, 1024)
    features = transformation_features[0].cpu().numpy()  # Take first batch

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Feature magnitude heatmap
    ax = axes[0, 0]
    im = ax.imshow(features.T, aspect="auto", cmap="viridis")
    ax.set_title("Feature Magnitude Heatmap")
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Feature Dimension")
    plt.colorbar(im, ax=ax)

    # Plot 2: Mean feature per token
    ax = axes[0, 1]
    mean_features = features.mean(axis=1)
    ax.plot(mean_features)
    ax.set_title("Mean Feature Magnitude per Token")
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Mean Magnitude")
    ax.axvline(x=256, color="r", linestyle="--", label="Query/Spatial boundary")
    ax.legend()

    # Plot 3: Feature distribution
    ax = axes[1, 0]
    ax.hist(features.flatten(), bins=50, alpha=0.7)
    ax.set_title("Feature Value Distribution")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Frequency")

    # Plot 4: Top feature dimensions
    ax = axes[1, 1]
    feature_importance = np.abs(features).mean(axis=0)
    top_indices = np.argsort(feature_importance)[-20:]
    ax.barh(range(20), feature_importance[top_indices])
    ax.set_title("Top 20 Feature Dimensions by Magnitude")
    ax.set_xlabel("Mean Absolute Value")
    ax.set_ylabel("Feature Dimension")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


# Example usage
if __name__ == "__main__":
    # Example: Print model summary
    from src.models.fst_model import FontDiffuserWithFST
    from src import (
        FontDiffuserModel,
        build_unet,
        build_style_encoder,
        build_content_encoder,
    )

    # Create dummy args
    class Args:
        # Add minimal required args
        pass

    print("This module provides utilities for FST model management.")
    print("Import and use the classes/functions as needed.")
