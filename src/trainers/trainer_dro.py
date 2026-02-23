"""
Direct Reward Optimization (DRO) trainer.

Wraps FontDiffuserFSTTrainer and augments each training step with a
differentiable reward signal computed from evaluate.py-style metrics:
  - SSIM between prediction and content image  (content fidelity)
  - LPIPS between prediction and style image   (style similarity)

The reward is added to the FST diffusion loss in a single combined backward
pass — there is exactly one accelerator.backward() call per step.
"""

import logging
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from src.trainers.trainer_fst import FontDiffuserFSTTrainer
from src.trainers.dro_rewards import DRORewardModule
from src.tools.utilities import HFTqdm

logger = logging.getLogger(__name__)


class FontDiffuserDROTrainer(FontDiffuserFSTTrainer):
    """DRO wrapper around FontDiffuserFSTTrainer.

    Overrides train_step to compute FST losses and the DRO reward in a single
    forward pass, then calls accelerator.backward exactly once on the combined
    loss.  This avoids the double-backward bug that arises when calling
    super().train_step (which already runs its own backward) and then adding a
    second reward loss on top.

    New args (add via add_dro_args in configs/fontdiffuser.py):
        --use_dro                : bool  — enable DRO (default: False)
        --dro_weight             : float — overall reward loss weight (default: 0.1)
        --dro_ssim_weight        : float — SSIM component weight (default: 1.0)
        --dro_lpips_weight       : float — LPIPS component weight (default: 1.0)
        --dro_reward_scale       : float — reward scale factor (default: 1.0)
        --dro_warmup_steps       : int   — steps before reward is added (default: 0)
        --dro_max_timestep_frac  : float — only evaluate reward below this fraction
                                           of num_train_timesteps (default: 0.3)
        --dro_sharp_weight       : float — sharpness reward weight (default: 0.0)
        --dro_div_weight         : float — diversity penalty weight (default: 0.0)
        --dro_normalise_reward   : bool  — normalise reward to unit variance (default: False)
    """

    def __init__(self, args) -> None:
        """Initialise DRO trainer.

        Args:
            args: Parsed argument namespace.  Must contain all args expected by
                  FontDiffuserFSTTrainer plus the DRO-specific args above.
        """
        self.use_dro: bool = getattr(args, "use_dro", False)
        self.dro_weight: float = getattr(args, "dro_weight", 0.1)
        self.dro_ssim_weight: float = getattr(args, "dro_ssim_weight", 1.0)
        self.dro_lpips_weight: float = getattr(args, "dro_lpips_weight", 1.0)
        self.dro_reward_scale: float = getattr(args, "dro_reward_scale", 1.0)
        self.dro_warmup_steps: int = getattr(args, "dro_warmup_steps", 0)
        self.dro_max_timestep_frac: float = getattr(args, "dro_max_timestep_frac", 0.3)
        self.dro_sharp_weight: float = getattr(args, "dro_sharp_weight", 0.0)
        self.dro_div_weight: float = getattr(args, "dro_div_weight", 0.0)
        self.dro_normalise_reward: bool = getattr(args, "dro_normalise_reward", False)

        super().__init__(args)

        if self.use_dro:
            logger.info("=" * 60)
            logger.info("DRO - Direct Reward Optimization enabled")
            logger.info(f"  dro_weight             : {self.dro_weight}")
            logger.info(f"  dro_ssim_weight        : {self.dro_ssim_weight}")
            logger.info(f"  dro_lpips_weight       : {self.dro_lpips_weight}")
            logger.info(f"  dro_reward_scale       : {self.dro_reward_scale}")
            logger.info(f"  dro_warmup_steps       : {self.dro_warmup_steps}")
            logger.info(f"  dro_max_timestep_frac  : {self.dro_max_timestep_frac}")
            logger.info(f"  dro_sharp_weight       : {self.dro_sharp_weight}")
            logger.info(f"  dro_div_weight         : {self.dro_div_weight}")
            logger.info(f"  dro_normalise_reward   : {self.dro_normalise_reward}")
            logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Override _setup_models to build the reward module
    # ------------------------------------------------------------------

    def _setup_models(self) -> None:
        """Build FST model components then attach the DRO reward module."""
        super()._setup_models()

        if self.use_dro:
            self.reward_module = DRORewardModule(
                ssim_weight=self.dro_ssim_weight,
                lpips_weight=self.dro_lpips_weight,
                reward_scale=self.dro_reward_scale,
                sharp_weight=self.dro_sharp_weight,
                div_weight=self.dro_div_weight,
                normalise=self.dro_normalise_reward,
            )
            logger.info("[OK] DRORewardModule created")
        else:
            self.reward_module = None

    # ------------------------------------------------------------------
    # Override _wrap_components to prepare the reward module
    # ------------------------------------------------------------------

    def _wrap_components(self) -> None:
        """Wrap FST components then prepare reward module via accelerator."""
        super()._wrap_components()

        if self.use_dro and self.reward_module is not None:
            # VGG has no trainable params so accelerator.prepare is a no-op for
            # the optimizer, but it ensures correct device/dtype placement under
            # FSDP, DeepSpeed, and AMP without bypassing the runtime.
            self.reward_module = self.accelerator.prepare(self.reward_module)
            logger.info("[OK] DRORewardModule prepared via accelerator")

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _denorm(tensor: torch.Tensor) -> torch.Tensor:
        """Convert [-1, 1] normalised tensor to [0, 1].

        Args:
            tensor: Float tensor normalised to [-1, 1].

        Returns:
            Tensor clamped to [0, 1].
        """
        return ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)

    @staticmethod
    def _match_spatial(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Bilinearly resize source to match target spatial dimensions if needed.

        Args:
            source: (B, C, H_s, W_s) float tensor.
            target: (B, C, H_t, W_t) float tensor used as size reference.

        Returns:
            source resized to (B, C, H_t, W_t) or unchanged if already matching.
        """
        if source.shape[-2:] == target.shape[-2:]:
            return source
        return F.interpolate(
            source,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    # ------------------------------------------------------------------
    # DRO reward branch — called inside the single combined forward pass
    # ------------------------------------------------------------------

    def _compute_dro_reward_loss(
        self,
        noise_pred: torch.Tensor,
        noisy_target_images: torch.Tensor,
        timesteps: torch.Tensor,
        content_images: torch.Tensor,
        style_images: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Reconstruct pred_x0 and compute the DRO reward loss.

        Uses the same noise_pred, noisy_target_images, and timesteps that were
        produced by the FST forward pass so the reward is coherent with the
        diffusion objective.

        Args:
            noise_pred: (B, C, H, W) predicted noise from the model.
            noisy_target_images: (B, C, H, W) x_t used in the same forward pass.
            timesteps: (B,) timesteps used in the same forward pass.
            content_images: (B, C, H, W) content reference in [-1, 1].
            style_images: (B, C, H, W) style reference in [-1, 1].

        Returns:
            Tuple of (reward_loss scalar tensor, metrics dict).
        """
        from src.tools.utils import x0_from_epsilon

        pred_x0 = x0_from_epsilon(
            scheduler=self.noise_scheduler,
            noise_pred=noise_pred,
            x_t=noisy_target_images,
            timesteps=timesteps,
        )

        pred_x0_01 = self._denorm(pred_x0)
        content_01 = self._match_spatial(self._denorm(content_images), pred_x0_01)
        style_01 = self._match_spatial(self._denorm(style_images), pred_x0_01)

        reward, metrics = self.reward_module(
            pred_images=pred_x0_01,
            content_images=content_01,
            style_images=style_01,
        )

        reward_loss = -reward
        return reward_loss, metrics

    # ------------------------------------------------------------------
    # Full train_step override — single forward pass, single backward
    # ------------------------------------------------------------------

    def train_step(
        self,
        samples: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Combined FST + DRO training step.

        Runs one forward pass, computes all losses (diffusion, consistency,
        identity, phase-2 SCR, and DRO reward), sums them, and returns the
        combined loss for the caller (train()) to call accelerator.backward on.

        DRO is only added after dro_warmup_steps and only for timesteps below
        dro_max_timestep_frac * num_train_timesteps so that pred_x0 is reliable.

        Args:
            samples: Batch dict from FontDatasetFST.

        Returns:
            Tuple of (combined_loss tensor, loss_dict).
        """
        self.model.train()

        content_images = samples["content_image"]
        style_images = samples["style_image"]
        target_images = samples["target_image"]
        nonorm_target = samples["nonorm_target_image"]

        noise = torch.randn_like(target_images)
        bsz = target_images.shape[0]

        # Decide timestep range: if DRO is active this step, restrict to low
        # timesteps so that pred_x0 is reliable for reward computation.
        global_step = getattr(self, "global_step", 0)
        dro_active = (
            self.use_dro
            and self.reward_module is not None
            and global_step >= self.dro_warmup_steps
        )

        num_train_ts = self.noise_scheduler.config.num_train_timesteps
        if dro_active:
            max_ts = max(1, int(num_train_ts * self.dro_max_timestep_frac))
        else:
            max_ts = num_train_ts

        timesteps = torch.randint(
            0,
            max_ts,
            (bsz,),
            device=target_images.device,
        ).long()

        noisy_targets = self.noise_scheduler.add_noise(target_images, noise, timesteps)

        # Classifier-free guidance dropout — pass full samples so style_source
        # conditioning is also dropped correctly (fixes samples=None bug).
        content_cfg, style_cfg = self.apply_classifier_free_guidance(
            content_images.clone(),
            style_images.clone(),
            self.config.drop_prob,
            samples=samples,
        )

        # ---- Forward pass ------------------------------------------------
        if self.use_fst:
            out = self.model(
                noisy_targets,
                timesteps,
                content_cfg,
                samples.get("style_source_image"),
                style_cfg,
                self.args.content_encoder_downsample_size,
            )
            noise_pred = out["noise_pred"]
            offset_out_sum = out["offset_out_sum"]
        else:
            noise_pred, offset_out_sum = self.model(
                noisy_targets,
                timesteps,
                style_cfg,
                content_cfg,
                self.args.content_encoder_downsample_size,
            )

        # ---- Diffusion losses --------------------------------------------
        total_loss, loss_dict, pred_orig_norm = self.compute_losses(
            noise_pred, noise, offset_out_sum, noisy_targets, nonorm_target, timesteps
        )

        # ---- Phase-2 SCR loss --------------------------------------------
        if self.config.phase_2 and self.scr and samples.get("neg_images") is not None:
            sc_loss = self.compute_phase2_loss(
                pred_orig_norm, target_images, samples["neg_images"]
            )
            total_loss = total_loss + self.config.sc_coefficient * sc_loss
            loss_dict["sc_loss"] = sc_loss.item()

        # ---- Consistency loss --------------------------------------------
        if (
            self.use_fst
            and self.num_consistency_pairs > 0
            and samples.get("consistency_source_images") is not None
            and samples["consistency_source_images"].numel() > 0
        ):
            unwrapped = self.accelerator.unwrap_model(self.model)
            c_loss = unwrapped.compute_consistency_loss(
                samples["consistency_source_images"],
                samples["consistency_target_images"],
            )
            total_loss = total_loss + self.consistency_loss_weight * c_loss
            loss_dict["consistency_loss"] = c_loss.item()

        # ---- Identity loss -----------------------------------------------
        if (
            self.use_fst
            and self.num_identity_pairs > 0
            and samples.get("num_identity_pairs_total", 0) > 0
        ):
            unwrapped = self.accelerator.unwrap_model(self.model)
            id_loss, id_metrics = unwrapped.compute_identity_loss(
                samples["identity_pair_sources"],
                samples["identity_pair_targets"],
                self.fst_num_queries,
            )
            total_loss = total_loss + self.identity_loss_weight * id_loss
            loss_dict["identity_loss"] = id_loss.item()
            loss_dict.update({f"identity_{k}": v for k, v in id_metrics.items()})

        # ---- DRO reward loss (same forward, same timesteps) --------------
        if dro_active:
            reward_loss, reward_metrics = self._compute_dro_reward_loss(
                noise_pred=noise_pred,
                noisy_target_images=noisy_targets,
                timesteps=timesteps,
                content_images=content_images,
                style_images=style_images,
            )
            scaled_reward_loss = self.dro_weight * reward_loss
            total_loss = total_loss + scaled_reward_loss
            loss_dict.update(reward_metrics)
            loss_dict["dro/reward_loss"] = reward_loss.item()
            loss_dict["dro/scaled_reward_loss"] = scaled_reward_loss.item()
        elif self.use_dro:
            loss_dict["dro/warmup_active"] = 1.0

        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # Checkpoint — persist and restore DRO config
    # ------------------------------------------------------------------

    def save_checkpoint(self, is_final: bool = False) -> None:
        """Save FST checkpoint extended with DRO configuration.

        Args:
            is_final: If True, saves to the ``final/`` subdirectory.
        """
        super().save_checkpoint(is_final=is_final)

        if not self.accelerator.is_main_process:
            return

        save_dir = (
            Path(self.args.output_dir) / "final"
            if is_final
            else Path(self.args.output_dir) / f"checkpoint_step_{self.global_step}"
        )
        state_path = save_dir / "training_state.pt"

        if not state_path.exists():
            logger.warning(
                f"[DRO] training_state.pt not found at {state_path} — "
                "DRO config was NOT saved."
            )
            return

        training_state = torch.load(
            state_path, map_location="cpu", weights_only=True
        )
        training_state["dro_config"] = {
            "use_dro": self.use_dro,
            "dro_weight": self.dro_weight,
            "dro_ssim_weight": self.dro_ssim_weight,
            "dro_lpips_weight": self.dro_lpips_weight,
            "dro_reward_scale": self.dro_reward_scale,
            "dro_warmup_steps": self.dro_warmup_steps,
            "dro_max_timestep_frac": self.dro_max_timestep_frac,
            "dro_sharp_weight": self.dro_sharp_weight,
            "dro_div_weight": self.dro_div_weight,
            "dro_normalise_reward": self.dro_normalise_reward,
        }
        torch.save(training_state, state_path)
        logger.info(f"[OK] DRO config appended to {state_path}")

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
        if not state_path.exists():
            return True

        training_state = torch.load(
            state_path, map_location="cpu", weights_only=True
        )
        dro_cfg = training_state.get("dro_config", {})
        if not dro_cfg:
            return True

        logger.info(f"DRO config from checkpoint: {dro_cfg}")

        if dro_cfg.get("use_dro") != self.use_dro:
            logger.warning(
                f"DRO mode mismatch: checkpoint={dro_cfg.get('use_dro')}, "
                f"current={self.use_dro}"
            )

        return True