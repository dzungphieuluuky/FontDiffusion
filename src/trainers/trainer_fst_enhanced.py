"""
Enhanced Trainer class for FontDiffuserWithFST using proposed auxiliary losses.

This trainer extends FontDiffuserFSTTrainer with:
  - FreqBandContentStyleLoss: Enforces content/style separation in frequency space
  - StrokeTopologyLoss: Ensures stroke presence/absence consistency
  - FreqWeightedDiffusionLoss: Spatially weights diffusion loss emphasizing strokes

All three losses are zero-trainable-weight and add minimal computational cost.
"""

import argparse
import logging
import math
import os
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from torchvision import transforms
import onnx
import onnxruntime

from src.dataset.font_dataset_fst import FontDataset as FontDatasetFST
from src.dataset.collate_fn_fst import CollateFN as CollateFNFST
from src.modules import UNet, ContentEncoder, StyleEncoder, SCR
from src.modules.frequency_decomposition import FrequencyDecomposition
from src.modules.proposed_losses import (
    FontDiffuserAuxLosses,
    compute_fft2,
)
from src import (
    ContentPerceptualLoss,
    FontDiffuserModel,
    build_content_encoder,
    build_ddpm_scheduler,
    build_scr,
    build_style_encoder,
    build_unet,
    build_fst,
    build_mss_encoder,
    build_fst_projection,
    build_original_style_projection,
    get_unet_cross_attention_dim,
    build_identity_loss_module,
    build_skeleton_transform,
    build_dual_channel_content_encoder,
    build_frequency_decomposition,
)
from src.model import FontDiffuserWithFST
from src.tools.utilities import (
    find_checkpoint,
    HFTqdm,
    load_model_checkpoint,
    save_model_checkpoint,
)
from src.tools.utils import (
    normalize_mean_std,
    reNormalize_img,
    save_args_to_yaml,
    x0_from_epsilon,
)
from src.trainers.training_config import TrainingConfig
from src.trainers.trainer_fst import FontDiffuserFSTTrainer
from src.modules.skeleton_distance_transform import SkeletonDistanceTransform

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class FontDiffuserFSTTrainerEnhanced(FontDiffuserFSTTrainer):
    """Enhanced FST trainer with proposed auxiliary losses for improved training."""

    def __init__(self, args: argparse.Namespace):
        """Initialize enhanced trainer with auxiliary loss configuration.

        Additional CLI arguments expected:
            use_aux_losses: Enable auxiliary loss functions (default: False)
            aux_freq_band: Enable FreqBandContentStyleLoss (default: True)
            aux_stroke_topo: Enable StrokeTopologyLoss (default: True)
            aux_freq_diff: Enable FreqWeightedDiffusionLoss (default: True)
            aux_freq_weight: Weight for freq band loss (default: 0.5)
            aux_topo_weight: Weight for topology loss (default: 0.3)
            aux_lf_radius: Low-freq radius for freq band loss (default: 0.1)
            aux_hf_radius: High-freq radius for freq band loss (default: 0.4)
            aux_lf_weight: Weight on low-freq component (default: 1.0)
            aux_hf_weight: Weight on high-freq component (default: 0.5)
            aux_topo_threshold: Stroke binarization threshold (default: 0.5)
            aux_topo_temperature: Sigmoid temperature for soft binarization (default: 0.05)
            aux_topo_topology_weight: Weight on topology BCE (default: 1.0)
            aux_topo_density_weight: Weight on density consistency (default: 0.3)
            aux_dark_ink: Whether ink is dark (default: True)
            aux_fw_lf_radius: Low-freq radius for diffusion weighting (default: 0.15)
            aux_fw_max_weight: Max weight for stroke pixels (default: 3.0)
            aux_fw_normalize_weights: Normalize weight map (default: True)
            aux_anneal_temperature: Anneal topology temperature during training (default: False)
            aux_temperature_schedule: Temperature schedule type (default: 'linear')
        """
        # Initialize auxiliary loss configuration
        self.use_aux_losses = getattr(args, "use_aux_losses", False)
        self.aux_anneal_temperature = getattr(args, "aux_anneal_temperature", False)
        self.aux_temperature_schedule = getattr(args, "aux_temperature_schedule", "linear")

        # Initialize parent class first
        super().__init__(args)

        # Initialize auxiliary losses if enabled
        if self.use_aux_losses:
            self._init_aux_losses()

    def _init_aux_losses(self):
        """Initialize auxiliary loss module with configuration from args."""
        logger.info("Initializing enhanced auxiliary losses...")

        # Extract loss configuration from args with defaults
        aux_config = {
            "use_freq_band": getattr(self.args, "aux_freq_band", True),
            "use_stroke_topo": getattr(self.args, "aux_stroke_topo", True),
            "use_freq_diff": getattr(self.args, "aux_freq_diff", True),
            "freq_weight": getattr(self.args, "aux_freq_weight", 0.5),
            "topo_weight": getattr(self.args, "aux_topo_weight", 0.3),
            # FreqBandContentStyleLoss
            "lf_radius": getattr(self.args, "aux_lf_radius", 0.1),
            "hf_radius": getattr(self.args, "aux_hf_radius", 0.4),
            "lf_weight": getattr(self.args, "aux_lf_weight", 1.0),
            "hf_weight": getattr(self.args, "aux_hf_weight", 0.5),
            # StrokeTopologyLoss
            "threshold": getattr(self.args, "aux_topo_threshold", 0.5),
            "temperature": getattr(self.args, "aux_topo_temperature", 0.05),
            "topology_weight": getattr(self.args, "aux_topo_topology_weight", 1.0),
            "density_weight": getattr(self.args, "aux_topo_density_weight", 0.3),
            "dark_ink": getattr(self.args, "aux_dark_ink", True),
            # FreqWeightedDiffusionLoss
            "fw_lf_radius": getattr(self.args, "aux_fw_lf_radius", 0.15),
            "max_weight": getattr(self.args, "aux_fw_max_weight", 3.0),
            "normalize_weights": getattr(self.args, "aux_fw_normalize_weights", True),
        }

        self.aux_losses = FontDiffuserAuxLosses(**aux_config)
        self.aux_config = aux_config

        # Store initial temperature for annealing
        self.aux_initial_temperature = aux_config["temperature"]

        logger.info(f"✓ Auxiliary losses initialized with config:")
        logger.info(f"  freq_band: {aux_config['use_freq_band']}")
        logger.info(f"  stroke_topo: {aux_config['use_stroke_topo']}")
        logger.info(f"  freq_diff: {aux_config['use_freq_diff']}")

    def _wrap_components(self):
        """Wrap components for distributed training, including auxiliary losses."""
        super()._wrap_components()

        # Wrap auxiliary losses if enabled
        if self.use_aux_losses and hasattr(self, "aux_losses"):
            self.aux_losses = self.accelerator.prepare(self.aux_losses)
            logger.info("✓ Auxiliary losses wrapped for distributed training")

    def _setup_logging(self):
        """Setup logging with auxiliary loss configuration."""
        super()._setup_logging()

        if not self.accelerator.is_main_process:
            return

        if self.use_aux_losses:
            aux_cfg = {
                "use_aux_losses": self.use_aux_losses,
                "use_freq_band": self.aux_config.get("use_freq_band", True),
                "use_stroke_topo": self.aux_config.get("use_stroke_topo", True),
                "use_freq_diff": self.aux_config.get("use_freq_diff", True),
                "freq_weight": self.aux_config.get("freq_weight", 0.5),
                "topo_weight": self.aux_config.get("topo_weight", 0.3),
                "anneal_temperature": self.aux_anneal_temperature,
                "temperature_schedule": self.aux_temperature_schedule,
            }
            self.accelerator.log({"aux_loss_config": aux_cfg})

    def _compute_aux_losses(
        self,
        noise_pred: torch.Tensor,
        noise: torch.Tensor,
        pred_x0: torch.Tensor,
        content_images: torch.Tensor,
        style_images: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """Compute auxiliary losses.

        Args:
            noise_pred: Model noise prediction (B, C, H, W)
            noise: Ground truth noise (B, C, H, W)
            pred_x0: Predicted clean x0 (B, C, H, W)
            content_images: Content image (B, C, H, W)
            style_images: Style image (B, C, H, W)

        Returns:
            aux_loss: Sum of auxiliary losses (scalar)
            diffusion_loss: Optional frequency-weighted diffusion loss (scalar)
            metrics: Dictionary of loss components and diagnostics
        """
        # Pre-compute FFTs to avoid redundant transforms
        with torch.no_grad():
            fft_pred = compute_fft2(pred_x0)
            fft_content = compute_fft2(content_images)
            fft_style = compute_fft2(style_images)

        # Call auxiliary losses with pre-computed FFTs
        aux_total, diffusion_loss, metrics = self.aux_losses(
            pred_x0=pred_x0,
            content=content_images,
            style=style_images,
            noise_pred=noise_pred,
            noise_target=noise,
            fft_pred=fft_pred,
            fft_content=fft_content,
            fft_style=fft_style,
        )

        return aux_total, diffusion_loss, metrics

    def _anneal_temperature(self, progress: float) -> None:
        """Anneal stroke topology loss temperature based on training progress.

        Args:
            progress: Training progress as fraction in [0, 1]
        """
        if not self.aux_anneal_temperature or not hasattr(self, "aux_losses"):
            return

        # Get unwrapped module from distributed wrapper
        aux_losses = self.accelerator.unwrap_model(self.aux_losses)
        if not hasattr(aux_losses, "stroke_topo_loss"):
            return

        stroke_topo = aux_losses.stroke_topo_loss
        if self.aux_temperature_schedule == "linear":
            # Linear schedule: T_final = 0.01
            new_temp = self.aux_initial_temperature * (1.0 - 0.9 * progress)
        elif self.aux_temperature_schedule == "exponential":
            # Exponential schedule
            new_temp = self.aux_initial_temperature * math.exp(-5.0 * progress)
        elif self.aux_temperature_schedule == "cosine":
            # Cosine annealing
            new_temp = (
                self.aux_initial_temperature
                * (1.0 + math.cos(math.pi * progress)) / 2.0
            )
        else:
            return

        stroke_topo.temperature = new_temp

        # Log temperature if logging on main process
        if self.accelerator.is_main_process and self.global_step % 100 == 0:
            logger.debug(f"Annealed topology temperature to {new_temp:.6f}")

    def train_step(self, samples):
        """Training step with auxiliary loss integration.

        Computes:
        1. Standard diffusion loss (MSE)
        2. Auxiliary losses (frequency band, stroke topology, frequency-weighted diffusion)
        3. Phase 2 SCR loss (if enabled)
        4. FST consistency and identity losses (if enabled)

        Returns:
            total_loss: Combined scalar loss
            loss_dict: Dictionary of all loss components for logging
        """
        self.model.train()
        content_images, style_images, target_images, nonorm_target = (
            samples["content_image"],
            samples["style_image"],
            samples["target_image"],
            samples["nonorm_target_image"],
        )
        noise = torch.randn_like(target_images)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (target_images.shape[0],),
            device=target_images.device,
        ).long()
        noisy_targets = self.noise_scheduler.add_noise(target_images, noise, timesteps)

        content_images, style_images = self.apply_classifier_free_guidance(
            content_images, style_images, self.config.drop_prob, samples
        )

        # Forward pass
        if self.use_fst:
            out = self.model(
                noisy_targets,
                timesteps,
                content_images,
                samples.get("style_source_image"),
                style_images,
                self.args.content_encoder_downsample_size,
            )
            noise_pred, offset_out_sum = out["noise_pred"], out["offset_out_sum"]
        else:
            noise_pred, offset_out_sum = self.model(
                noisy_targets,
                timesteps,
                style_images,
                content_images,
                self.args.content_encoder_downsample_size,
            )

        # Compute standard base losses
        total_loss, loss_dict, pred_orig_norm = self.compute_losses(
            noise_pred, noise, offset_out_sum, noisy_targets, nonorm_target, timesteps
        )

        # Add auxiliary losses if enabled
        if self.use_aux_losses:
            # Reconstruct pred_x0 for auxiliary losses
            pred_x0 = x0_from_epsilon(
                self.noise_scheduler, noise_pred, noisy_targets, timesteps
            )
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            pred_x0_norm = (pred_x0 + 1) / 2  # Map to [0, 1]

            # Normalize images for loss computation
            content_norm = (content_images + 1) / 2
            style_norm = (style_images + 1) / 2

            aux_loss, diffusion_loss, aux_metrics = self._compute_aux_losses(
                noise_pred,
                noise,
                pred_x0_norm,
                content_norm,
                style_norm,
            )

            # Add auxiliary losses to total
            total_loss = total_loss + aux_loss
            loss_dict.update(aux_metrics)

            # Option: replace standard MSE with frequency-weighted diffusion loss
            # Uncomment to use frequency-weighted diffusion instead of standard MSE
            # if diffusion_loss is not None:
            #     total_loss = total_loss + self.config.aux_diffusion_coefficient * diffusion_loss

        # Phase 2 SCR loss
        if self.config.phase_2 and self.scr and samples.get("neg_images") is not None:
            sc_loss = self.compute_phase2_loss(
                pred_orig_norm, target_images, samples["neg_images"]
            )
            total_loss += self.config.sc_coefficient * sc_loss
            loss_dict["sc_loss"] = sc_loss.item()

        # FST consistency loss
        if (
            self.use_fst
            and self.num_consistency_pairs > 0
            and samples.get("consistency_source_images") is not None
        ):
            if samples["consistency_source_images"].numel() > 0:
                model = self.accelerator.unwrap_model(self.model)
                c_loss = model.compute_consistency_loss(
                    samples["consistency_source_images"],
                    samples["consistency_target_images"],
                )
                total_loss += self.consistency_loss_weight * c_loss
                loss_dict["consistency_loss"] = c_loss.item()

        # FST identity loss
        if (
            self.use_fst
            and self.num_identity_pairs > 0
            and samples.get("num_identity_pairs_total", 0) > 0
        ):
            model = self.accelerator.unwrap_model(self.model)
            id_loss, id_metrics = model.compute_identity_loss(
                samples["identity_pair_sources"],
                samples["identity_pair_targets"],
                self.fst_num_queries,
            )
            total_loss += self.identity_loss_weight * id_loss
            loss_dict.update(
                {
                    "identity_loss": id_loss.item(),
                    **{f"identity_{k}": v for k, v in id_metrics.items()},
                }
            )

        return total_loss, loss_dict

    def train(self):
        """Main training loop with temperature annealing."""
        num_update_steps = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        num_epochs = math.ceil(self.config.max_train_steps / num_update_steps)
        if getattr(self.args, "resume_from_checkpoint", None):
            self.load_checkpoint(self.args.resume_from_checkpoint)

        progress_bar = HFTqdm(
            range(self.config.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        loss_accum, count_accum = 0, 0

        for epoch in range(self.current_epoch, num_epochs):
            for step, samples in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    total_loss, loss_dict = self.train_step(samples)
                    loss_accum += total_loss.detach().item()
                    count_accum += 1

                    self.accelerator.backward(total_loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    progress_bar.update(1)
                    self.global_step += 1

                    if self.accelerator.is_main_process:
                        loss_avg = loss_accum / count_accum
                        loss_accum, count_accum = 0, 0

                        if self.global_step % self.config.log_interval == 0:
                            logs = {
                                "loss/avg_train_loss": loss_avg,
                                "train/lr": self.lr_scheduler.get_last_lr()[0],
                                "train/epoch": epoch + step / len(self.train_dataloader),
                                "train/grad_norm": grad_norm.item(),
                                **{f"loss/{k}": v for k, v in loss_dict.items()},
                            }
                            progress_bar.set_postfix(logs)
                            self.accelerator.log(logs, step=self.global_step)

                        # Anneal topology temperature
                        if self.use_aux_losses:
                            progress = self.global_step / self.config.max_train_steps
                            self._anneal_temperature(progress)

                        if self.global_step % self.config.ckpt_interval == 0:
                            self.save_checkpoint()

                    if self.global_step >= self.config.max_train_steps:
                        break

            self.current_epoch += 1

        progress_bar.close()
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.save_checkpoint(is_final=True)
        self.accelerator.end_training()

    def save_checkpoint(self, is_final: bool = False):
        """Save checkpoint including auxiliary loss state using safetensors format."""
        unwrapped = self.accelerator.unwrap_model(self.model)
        if not self.accelerator.is_main_process:
            return

        save_dir = Path(self.args.output_dir) / (
            "final" if is_final else f"checkpoint_step_{self.global_step}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.use_fst:
            components = {
                "unet": unwrapped.diffusion_unet,
                "style_encoder": unwrapped.style_encoder,
                "content_encoder": unwrapped.content_encoder,
                "mss_encoder": unwrapped.mss_encoder,
                "fst_module": unwrapped.fst_module,
                "fst_projection": unwrapped.fst_projection,
                "original_style_projection": unwrapped.original_style_projection,
            }
            for name, mod in components.items():
                save_model_checkpoint(
                    mod.state_dict(), save_dir / f"{name}.safetensors"
                )
            if self.identity_loss_module:
                save_model_checkpoint(
                    self.identity_loss_module.state_dict(),
                    save_dir / "identity_loss_module.safetensors",
                )
            if unwrapped.skeleton_transform is not None:
                save_model_checkpoint(
                    unwrapped.skeleton_transform.state_dict(),
                    save_dir / "skeleton_transform.safetensors",
                )
        else:
            save_model_checkpoint(
                unwrapped.state_dict(), save_dir / "model.safetensors"
            )

        if self.config.phase_2 and self.scr:
            save_model_checkpoint(
                self.scr.state_dict(), save_dir / "scr.safetensors"
            )

        # Save auxiliary loss configuration
        if self.use_aux_losses:
            import json

            aux_config_path = save_dir / "aux_loss_config.json"
            with open(aux_config_path, "w") as f:
                json.dump(self.aux_config, f, indent=2)
            logger.info(f"✓ Auxiliary loss config saved to {aux_config_path}")

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.current_epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "config": asdict(self.config),
            "fst_config": {
                k: getattr(self, k)
                for k in [
                    "use_fst",
                    "style_source_same_prob",
                    "fst_num_queries",
                    "fst_query_dim",
                    "num_consistency_pairs",
                    "num_identity_pairs",
                ]
            },
            **(
                {
                    "skeleton_config": {
                        k: getattr(self, k)
                        for k in ["use_skeleton_content", "skeleton_method"]
                    }
                }
                if self.use_skeleton_content
                else {}
            ),
            **(
                {
                    "frequency_config": {
                        k: getattr(self, k)
                        for k in ["use_frequency_decomp", "frequency_low_cutoff"]
                    }
                }
                if self.use_frequency_decomp
                else {}
            ),
        }
        if self.use_aux_losses:
            state["aux_loss_config"] = self.aux_config
        torch.save(state, save_dir / "training_state.pt")

        save_args_to_yaml(self.args, save_dir / "args.yaml")
        logger.info(f"✓ Checkpoint saved to {save_dir}")

    def load_checkpoint(self, path):
        """Load checkpoint including auxiliary loss configuration.
        
        Calls parent's load_checkpoint and additionally restores
        aux_loss_config if auxiliary losses are enabled.
        """
        # Call parent's load_checkpoint to handle all FST checkpoint loading
        success = super().load_checkpoint(path)
        if not success:
            return False

        # Load auxiliary loss configuration if present
        if self.use_aux_losses:
            ckpt_dir = Path(path)
            state_file = next(
                (
                    ckpt_dir / f
                    for f in ["training_state.pt", "training_state.pth"]
                    if (ckpt_dir / f).exists()
                ),
                None,
            )
            if state_file:
                try:
                    state = torch.load(state_file, map_location="cpu")
                    if "aux_loss_config" in state:
                        # Update auxiliary loss configuration from checkpoint
                        self.aux_config = state["aux_loss_config"]
                        logger.info(f"✓ Loaded auxiliary loss config from checkpoint")
                except Exception as e:
                    logger.warning(f"Failed to load aux_loss_config: {e}")
                    # Continue anyway - aux losses will use default config

        return True
