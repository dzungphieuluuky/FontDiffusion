"""
Direct Reward Optimization (DRO) trainer.

Wraps FontDiffuserFSTTrainer and augments each training step with a
differentiable reward signal computed from evaluate.py-style metrics:
  - SSIM between prediction and content image  (content fidelity)
  - LPIPS between prediction and style image   (style similarity)

The reward is added to the FST diffusion loss in a single combined backward
pass -- there is exactly one accelerator.backward() call per step.

Inherited from FontDiffuserFSTTrainer (no override needed):
  - _setup_data        : FontDatasetFST + CollateFNFST dataloader
  - _setup_optimizer   : AdamW + lr scheduler
  - _wrap_components   : accelerator.prepare for model/optimizer/dataloader
                         (extended here for reward_module)
  - train              : epoch/step loop with gradient accumulation
  - export_to_onnx     : ONNX export for FST model

Overridden here:
  - _setup_models      : adds DRORewardModule after FST model setup
  - _setup_logging     : logs DRO config on top of FST config
  - _wrap_components   : prepares reward_module via accelerator
  - train_step         : single forward pass combining FST + DRO losses
  - save_checkpoint    : appends dro_config to training_state.pt
  - load_checkpoint    : validates dro_config on resume
"""

import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from src.trainers.trainer_fst import FontDiffuserFSTTrainer
from src.trainers.dro_rewards import DRORewardModule
from src.tools.utils import x0_from_epsilon

logger = logging.getLogger(__name__)


class FontDiffuserDROTrainer(FontDiffuserFSTTrainer):
    """DRO wrapper around FontDiffuserFSTTrainer.

    All model construction, data loading, optimizer setup, and the training
    loop are inherited unchanged from FontDiffuserFSTTrainer.  Only train_step
    is replaced so that FST losses and the DRO reward are summed before the
    single accelerator.backward() call in train().

    Args:
        args: Parsed argument namespace containing all FontDiffuserFSTTrainer
              args plus the DRO-specific args below.

    DRO-specific args (register in configs/fontdiffuser.py):
        --use_dro                : bool  -- enable DRO (default: False)
        --dro_weight             : float -- overall reward loss weight (default: 0.1)
        --dro_ssim_weight        : float -- SSIM component weight (default: 1.0)
        --dro_lpips_weight       : float -- LPIPS component weight (default: 1.0)
        --dro_reward_scale       : float -- reward scale factor (default: 1.0)
        --dro_warmup_steps       : int   -- steps before reward is added (default: 0)
        --dro_max_timestep_frac  : float -- reward only at t < frac*T (default: 0.3)
        --dro_sharp_weight       : float -- sharpness reward weight (default: 0.0)
        --dro_div_weight         : float -- diversity penalty weight (default: 0.0)
        --dro_normalise_reward   : bool  -- normalise reward to unit var (default: False)
    """

    def __init__(self, args) -> None:
        # Parse DRO args before super().__init__ so they are available when
        # _setup_models and _setup_logging are called by the parent __init__.
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

        # Calls FontDiffuserFSTTrainer.__init__ which internally calls:
        #   _setup_models, _setup_data, _setup_optimizer,
        #   _setup_logging, _wrap_components
        super().__init__(args)

    # ------------------------------------------------------------------
    # _setup_models — inherited FST setup + DRORewardModule construction
    # ------------------------------------------------------------------

    def _setup_models(self) -> None:
        """Build all FST model components then attach DRORewardModule.

        Delegates the full FST model construction (unet, style_encoder,
        content_encoder, mss_encoder, fst_module, fst_projection,
        original_style_projection, skeleton_transform, frequency_decomp,
        identity_loss_module, perceptual_loss, scr) to the parent, then
        appends the reward module if DRO is enabled.
        """
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
    # _setup_logging — inherited FST logging + DRO config
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        """Log FST config then append DRO config to the experiment tracker.

        The parent logs fst_config, skeleton_config, frequency_config, and
        model parameter counts.  This override appends dro_config so the full
        training configuration is recorded in one place.
        """
        super()._setup_logging()

        if not self.accelerator.is_main_process:
            return

        if not self.use_dro:
            return

        dro_cfg = {
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
        self.accelerator.log({"dro_config": dro_cfg})

        logger.info("=" * 60)
        logger.info("DRO - Direct Reward Optimization enabled")
        for k, v in dro_cfg.items():
            logger.info(f"  {k:<28}: {v}")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # _wrap_components — inherited FST wrapping + reward module placement
    # ------------------------------------------------------------------

    def _wrap_components(self) -> None:
        """Prepare FST components then place reward module via accelerator.

        The parent handles accelerator.prepare for model, optimizer,
        lr_scheduler, train_dataloader, and identity_loss_module, and also
        loads deferred FST module states and optionally torch.compiles the
        model.  This override adds reward_module placement.

        VGG has no trainable params so accelerator.prepare is a no-op for the
        optimizer, but it ensures correct device/dtype placement under FSDP,
        DeepSpeed, and AMP without raw .to(device) calls.
        """
        super()._wrap_components()

        if self.use_dro and self.reward_module is not None:
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
            Float tensor clamped to [0, 1].
        """
        return ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)

    @staticmethod
    def _match_spatial(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Bilinearly resize source to match target spatial dimensions if needed.

        Args:
            source: (B, C, H_s, W_s) float tensor.
            target: (B, C, H_t, W_t) float tensor used only as a size reference.

        Returns:
            source resized to (B, C, H_t, W_t), or source unchanged if sizes match.
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
    # DRO reward computation
    # ------------------------------------------------------------------

    def _compute_dro_reward_loss(
        self,
        noise_pred: torch.Tensor,
        noisy_target_images: torch.Tensor,
        timesteps: torch.Tensor,
        content_images: torch.Tensor,
        style_images: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Reconstruct pred_x0 from this step's tensors and compute reward loss.

        Reuses the noise_pred, noisy_target_images, and timesteps already
        produced by the forward pass so the reward is coherent with the
        diffusion objective and no second forward pass is needed.

        Args:
            noise_pred: (B, C, H, W) predicted noise from the model.
            noisy_target_images: (B, C, H, W) x_t from the same forward pass.
            timesteps: (B,) timesteps from the same forward pass.
            content_images: (B, C, H, W) content reference in [-1, 1].
            style_images: (B, C, H, W) style reference in [-1, 1].

        Returns:
            Tuple of (reward_loss scalar tensor, metrics dict).
        """
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

        # Minimise negative reward
        reward_loss = -reward
        return reward_loss, metrics

    # ------------------------------------------------------------------
    # train_step — single forward pass, single backward
    # ------------------------------------------------------------------

    def train_step(
        self,
        samples: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Combined FST + DRO training step.

        Mirrors FontDiffuserFSTTrainer.train_step exactly for the diffusion,
        SCR, consistency, and identity losses, then appends the DRO reward
        loss before returning.  The caller (train()) calls accelerator.backward
        exactly once on the returned combined loss.

        DRO reward is only included after dro_warmup_steps and only when
        timesteps are below dro_max_timestep_frac * num_train_timesteps so
        that the one-step pred_x0 estimate is reliable.

        Args:
            samples: Batch dict from FontDatasetFST containing keys:
                content_image, style_image, target_image, nonorm_target_image,
                and optionally style_source_image, neg_images,
                consistency_source_images, consistency_target_images,
                identity_pair_sources, identity_pair_targets,
                num_identity_pairs_total.

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

        # Determine whether DRO reward should be added this step.
        # Warmup check uses getattr guard in case global_step is not yet set.
        global_step = getattr(self, "global_step", 0)
        dro_active = (
            self.use_dro
            and self.reward_module is not None
            and global_step >= self.dro_warmup_steps
        )

        # Restrict timestep range when DRO is active so pred_x0 is reliable.
        # At large t the one-step x0 estimate is too noisy to carry a useful
        # reward signal.
        num_train_ts = self.noise_scheduler.config.num_train_timesteps
        max_ts = (
            max(1, int(num_train_ts * self.dro_max_timestep_frac))
            if dro_active
            else num_train_ts
        )

        timesteps = torch.randint(
            0,
            max_ts,
            (bsz,),
            device=target_images.device,
        ).long()

        noisy_targets = self.noise_scheduler.add_noise(target_images, noise, timesteps)

        # Pass full samples so style_source_image conditioning is dropped
        # correctly alongside content and style during CFG dropout.
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

        # ---- Diffusion losses (noise prediction + offset) ----------------
        total_loss, loss_dict, pred_orig_norm = self.compute_losses(
            noise_pred, noise, offset_out_sum, noisy_targets, nonorm_target, timesteps
        )

        # ---- Phase-2 SCR contrastive loss --------------------------------
        if self.config.phase_2 and self.scr and samples.get("neg_images") is not None:
            sc_loss = self.compute_phase2_loss(
                pred_orig_norm, target_images, samples["neg_images"]
            )
            total_loss = total_loss + self.config.sc_coefficient * sc_loss
            loss_dict["sc_loss"] = sc_loss.item()

        # ---- FST consistency loss ----------------------------------------
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

        # ---- FST identity mapping loss -----------------------------------
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

        # ---- DRO reward loss (reuses same noise_pred and timesteps) ------
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
            # Warmup still in progress — log a marker without adding loss.
            loss_dict["dro/warmup_active"] = 1.0

        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # save_checkpoint — FST checkpoint + dro_config block
    # ------------------------------------------------------------------

    def save_checkpoint(self, is_final: bool = False) -> None:
        """Save FST checkpoint then append dro_config to training_state.pt.

        The parent saves all model components (unet, style_encoder,
        content_encoder, mss_encoder, fst_module, fst_projection,
        original_style_projection, identity_loss_module, scr) and writes
        training_state.pt with optimizer/scheduler state and fst_config.
        This override re-opens that file to append the dro_config block so
        the full training configuration is preserved in one file.

        Args:
            is_final: If True saves to <output_dir>/final/, otherwise to
                      <output_dir>/checkpoint_step_<global_step>/.
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
                f"[DRO] training_state.pt not found at {state_path} -- "
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

    # ------------------------------------------------------------------
    # load_checkpoint — FST checkpoint loading + dro_config validation
    # ------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load FST checkpoint then validate the stored dro_config.

        The parent restores global_step, current_epoch, optimizer state,
        lr_scheduler state, and all model component weights.  This override
        reads back dro_config and warns if the DRO mode does not match the
        current run configuration so the user is not silently misled.

        Args:
            checkpoint_path: Path to the checkpoint directory.

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

        logger.info(f"DRO config restored from checkpoint: {dro_cfg}")

        if dro_cfg.get("use_dro") != self.use_dro:
            logger.warning(
                f"DRO mode mismatch: checkpoint={dro_cfg.get('use_dro')}, "
                f"current={self.use_dro}. Continuing with current settings."
            )

        return True