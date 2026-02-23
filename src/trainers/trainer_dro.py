"""
Direct Reward Optimization (DRO) trainer.

Wraps FontDiffuserFSTTrainer and augments each training step with a
differentiable reward signal computed from evaluate.py-style metrics:
  - SSIM between prediction and content image  (content fidelity)
  - LPIPS between prediction and style image   (style similarity)

The reward loss is added on top of the existing FST diffusion losses, allowing
the model to generalise beyond the synthetic FontDiffuser baseline dataset.
"""

import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler

from src.trainers.trainer_fst import FontDiffuserFSTTrainer
from src.trainers.dro_rewards import DRORewardModule
from src.tools.utilities import HFTqdm
from src.tools.utils import x0_from_epsilon

logger = logging.getLogger(__name__)


class FontDiffuserDROTrainer(FontDiffuserFSTTrainer):
    """DRO wrapper around FontDiffuserFSTTrainer.

    Adds a reward-weighted loss term to each training step:

        total_loss = fst_loss - dro_weight * reward(pred_x0, content, style)

    The reward is computed by decoding a one-step x0 estimate from the current
    noise prediction then evaluating SSIM and LPIPS against the content and
    style references respectively.

    New args recognised (add to configs/fontdiffuser.py):
        --use_dro           : bool  — enable DRO (default False)
        --dro_weight        : float — overall DRO loss weight (default 0.1)
        --dro_ssim_weight   : float — SSIM component weight  (default 1.0)
        --dro_lpips_weight  : float — LPIPS component weight (default 1.0)
        --dro_reward_scale  : float — reward scale factor    (default 1.0)
        --dro_warmup_steps  : int   — steps before DRO is activated (default 0)
    """

    def __init__(self, args) -> None:
        """Initialise DRO trainer.

        Args:
            args: Parsed argument namespace.  Must contain all args expected by
                  FontDiffuserFSTTrainer plus the DRO-specific args above.
        """
        # Parse DRO-specific args before calling super
        self.use_dro: bool = getattr(args, "use_dro", False)
        self.dro_weight: float = getattr(args, "dro_weight", 0.1)
        self.dro_ssim_weight: float = getattr(args, "dro_ssim_weight", 1.0)
        self.dro_lpips_weight: float = getattr(args, "dro_lpips_weight", 1.0)
        self.dro_reward_scale: float = getattr(args, "dro_reward_scale", 1.0)
        self.dro_warmup_steps: int = getattr(args, "dro_warmup_steps", 0)

        super().__init__(args)

        if self.use_dro:
            logger.info("=" * 80)
            logger.info("DRO — Direct Reward Optimization enabled")
            logger.info(f"  dro_weight      : {self.dro_weight}")
            logger.info(f"  dro_ssim_weight : {self.dro_ssim_weight}")
            logger.info(f"  dro_lpips_weight: {self.dro_lpips_weight}")
            logger.info(f"  dro_reward_scale: {self.dro_reward_scale}")
            logger.info(f"  dro_warmup_steps: {self.dro_warmup_steps}")
            logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Override _setup_models to also build the reward module
    # ------------------------------------------------------------------

    def _setup_models(self) -> None:
        """Build FST model components then attach the DRO reward module."""
        super()._setup_models()

        if self.use_dro:
            self.reward_module = DRORewardModule(
                ssim_weight=self.dro_ssim_weight,
                lpips_weight=self.dro_lpips_weight,
                reward_scale=self.dro_reward_scale,
            )
            logger.info("✓ DRORewardModule created")
        else:
            self.reward_module = None

    # ------------------------------------------------------------------
    # Override _wrap_components to prepare the reward module
    # ------------------------------------------------------------------

    def _wrap_components(self) -> None:
        """Wrap FST components then move reward module to the correct device."""
        super()._wrap_components()

        if self.use_dro and self.reward_module is not None:
            # Move VGG to the same device as the model (not wrapped with
            # accelerator — it has no trainable params)
            device = self.accelerator.device
            self.reward_module = self.reward_module.to(device)
            logger.info(f"✓ DRORewardModule moved to {device}")

    # ------------------------------------------------------------------
    # Denormalise helper
    # ------------------------------------------------------------------

    @staticmethod
    def _denorm(tensor: torch.Tensor) -> torch.Tensor:
        """Convert [-1, 1] normalised tensor to [0, 1].

        Args:
            tensor: Any-shape float tensor normalised to [-1, 1].

        Returns:
            Tensor clamped to [0, 1].
        """
        return ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # DRO reward computation
    # ------------------------------------------------------------------

    def _compute_dro_reward_loss(
        self,
        noise_pred: torch.Tensor,
        noise: torch.Tensor,
        noisy_target_images: torch.Tensor,
        timesteps: torch.Tensor,
        content_images: torch.Tensor,
        style_images: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Decode x0 estimate and compute the DRO reward loss.

        Steps:
        1. Reconstruct x0 from the noise prediction using the DDPM scheduler.
        2. Denormalise pred_x0, content, and style to [0, 1].
        3. Compute composite reward (SSIM + LPIPS).
        4. Return negative reward as a minimisable loss.

        Args:
            noise_pred: (B, C, H, W) predicted noise from the model.
            noise: (B, C, H, W) ground-truth noise added to target.
            noisy_target_images: (B, C, H, W) noisy target at timestep t.
            timesteps: (B,) diffusion timesteps.
            content_images: (B, C, H, W) content reference in [-1, 1].
            style_images: (B, C, H, W) style reference in [-1, 1].

        Returns:
            Tuple of (reward_loss scalar tensor, metrics dict).
        """
        # Reconstruct x0 estimate
        pred_x0 = x0_from_epsilon(
            scheduler=self.noise_scheduler,
            noise_pred=noise_pred,
            x_t=noisy_target_images,
            timesteps=timesteps,
        )

        # Denormalise to [0, 1]
        pred_x0_01 = self._denorm(pred_x0)
        content_01 = self._denorm(content_images)
        style_01 = self._denorm(style_images)

        # Resize to match spatial dims if needed (content may differ in size)
        if content_01.shape[-2:] != pred_x0_01.shape[-2:]:
            content_01 = F.interpolate(
                content_01, size=pred_x0_01.shape[-2:], mode="bilinear", align_corners=False
            )
        if style_01.shape[-2:] != pred_x0_01.shape[-2:]:
            style_01 = F.interpolate(
                style_01, size=pred_x0_01.shape[-2:], mode="bilinear", align_corners=False
            )

        reward, metrics = self.reward_module(
            pred_images=pred_x0_01,
            content_images=content_01,
            style_images=style_01,
        )

        # Minimise negative reward
        reward_loss = -reward
        return reward_loss, metrics

    # ------------------------------------------------------------------
    # Override train_step to inject DRO reward
    # ------------------------------------------------------------------

    def train_step(
        self,
        samples: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """FST training step augmented with DRO reward loss.

        The DRO reward loss is only added after ``dro_warmup_steps`` to allow
        the diffusion objective to stabilise first.

        Args:
            samples: Batch dict from FontDatasetFST.

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        # Run standard FST training step
        total_loss, loss_dict = super().train_step(samples)

        # Skip DRO if disabled or still in warmup
        if not self.use_dro or self.reward_module is None:
            return total_loss, loss_dict

        if self.global_step < self.dro_warmup_steps:
            loss_dict["dro/skipped"] = 1.0
            return total_loss, loss_dict

        # Re-run forward pass to get noise_pred with grad for reward
        # (super().train_step already called backward — we need a fresh forward
        #  under no_grad-free context for the reward branch)
        content_images = samples["content_image"]
        style_images = samples["style_image"]
        target_images = samples["target_image"]

        noise = torch.randn_like(target_images)
        bsz = target_images.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=target_images.device,
        ).long()
        noisy_target_images = self.noise_scheduler.add_noise(target_images, noise, timesteps)

        content_images_cfg, style_images_cfg = self.apply_classifier_free_guidance(
            content_images.clone(),
            style_images.clone(),
            self.config.drop_prob,
            samples=None,
        )

        self.model.train()
        if self.use_fst:
            style_source_images = samples.get("style_source_image")
            model_output = self.model(
                noisy_latents=noisy_target_images,
                timestep=timesteps,
                content_images=content_images_cfg,
                style_source_images=style_source_images,
                style_target_images=style_images_cfg,
                content_encoder_downsample_size=self.args.content_encoder_downsample_size,
            )
            noise_pred = model_output["noise_pred"]
        else:
            noise_pred, _ = self.model(
                x_t=noisy_target_images,
                timesteps=timesteps,
                style_images=style_images_cfg,
                content_images=content_images_cfg,
                content_encoder_downsample_size=self.args.content_encoder_downsample_size,
            )

        reward_loss, reward_metrics = self._compute_dro_reward_loss(
            noise_pred=noise_pred,
            noise=noise,
            noisy_target_images=noisy_target_images,
            timesteps=timesteps,
            content_images=content_images,
            style_images=style_images,
        )

        # Scale and accumulate reward loss
        scaled_reward_loss = self.dro_weight * reward_loss
        total_loss = total_loss + scaled_reward_loss

        loss_dict.update(reward_metrics)
        loss_dict["dro/reward_loss"] = reward_loss.item()
        loss_dict["dro/scaled_reward_loss"] = scaled_reward_loss.item()

        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # Override save/load checkpoint to persist DRO config
    # ------------------------------------------------------------------

    def save_checkpoint(self, is_final: bool = False) -> None:
        """Save FST checkpoint extended with DRO configuration.

        Args:
            is_final: If True, saves to ``final/`` subdirectory.
        """
        super().save_checkpoint(is_final=is_final)

        # Append DRO config to training_state.pt
        if self.accelerator.is_main_process:
            save_dir = (
                Path(self.args.output_dir) / "final"
                if is_final
                else Path(self.args.output_dir) / f"checkpoint_step_{self.global_step}"
            )
            state_path = save_dir / "training_state.pt"
            if state_path.exists():
                training_state = torch.load(state_path, map_location="cpu")
                training_state["dro_config"] = {
                    "use_dro": self.use_dro,
                    "dro_weight": self.dro_weight,
                    "dro_ssim_weight": self.dro_ssim_weight,
                    "dro_lpips_weight": self.dro_lpips_weight,
                    "dro_reward_scale": self.dro_reward_scale,
                    "dro_warmup_steps": self.dro_warmup_steps,
                }
                torch.save(training_state, state_path)
                logger.info("✓ DRO config appended to training_state.pt")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load FST checkpoint and validate DRO configuration.

        Args:
            checkpoint_path: Path to checkpoint directory.

        Returns:
            True if loaded successfully, False otherwise.
        """
        success = super().load_checkpoint(checkpoint_path)
        if not success:
            return False

        state_path = Path(checkpoint_path) / "training_state.pt"
        if state_path.exists():
            training_state = torch.load(state_path, map_location="cpu")
            dro_cfg = training_state.get("dro_config", {})
            if dro_cfg:
                logger.info(f"DRO config from checkpoint: {dro_cfg}")
                if dro_cfg.get("use_dro") != self.use_dro:
                    logger.warning(
                        f"⚠️ DRO mode mismatch: checkpoint={dro_cfg.get('use_dro')}, "
                        f"current={self.use_dro}"
                    )

        return True