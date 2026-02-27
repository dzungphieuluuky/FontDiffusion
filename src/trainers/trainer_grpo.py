"""
GRPO trainer for FontDiffuser.

Group Relative Policy Optimization -- no critic, no value network.
For each (content, style) pair, generate G rollouts, score each with
the reward module, normalize within the group, then compute the clipped
importance-weighted policy gradient loss.

Reference: DeepSeek-R1 (Shao et al., 2024).

New args (add to configs/fontdiffuser.py --grpo group):
    --use_grpo              : bool  -- enable GRPO (default: False)
    --grpo_group_size       : int   -- rollouts per prompt G (default: 4)
    --grpo_clip_eps         : float -- PPO-style clip epsilon (default: 0.2)
    --grpo_pg_weight        : float -- weight of GRPO loss (default: 0.01)
    --grpo_sample_steps     : int   -- denoising steps per rollout (default: 5)
    --grpo_warmup_steps     : int   -- steps before GRPO activates (default: 1000)
    --grpo_kl_coeff         : float -- KL penalty coefficient (default: 0.01)
    --grpo_reward_clip      : float -- reward clip value (default: 5.0)
"""
import logging
from pathlib import Path

import torch
import torch.nn.functional as F

from src.trainers.trainer_dro import FontDiffuserDROTrainer

logger = logging.getLogger(__name__)


class FontDiffuserGRPOTrainer(FontDiffuserDROTrainer):
    """GRPO trainer wrapping FontDiffuserDROTrainer.

    For each training batch:
      1. DRO supervised losses (inherited, single forward pass).
      2. G independent rollouts per sample using the current policy (no grad).
      3. Score all G*B samples with the reward module.
      4. Normalize rewards within each group of G (zero-mean, unit std).
      5. Compute clipped importance-weighted PG loss using old vs new log-probs.
      6. Optionally add KL penalty against the reference (frozen) policy.
      7. Sum all losses, single accelerator.backward().

    Args:
        args: Namespace with all DRO args plus GRPO-specific args.
    """

    def __init__(self, args) -> None:
        self.use_grpo: bool = getattr(args, "use_grpo", False)
        self.grpo_group_size: int = getattr(args, "grpo_group_size", 4)
        self.grpo_clip_eps: float = getattr(args, "grpo_clip_eps", 0.2)
        self.grpo_pg_weight: float = getattr(args, "grpo_pg_weight", 0.01)
        self.grpo_sample_steps: int = getattr(args, "grpo_sample_steps", 5)
        self.grpo_warmup_steps: int = getattr(args, "grpo_warmup_steps", 1000)
        self.grpo_kl_coeff: float = getattr(args, "grpo_kl_coeff", 0.01)
        self.grpo_reward_clip: float = getattr(args, "grpo_reward_clip", 5.0)

        # Reference policy (frozen copy for KL) -- set after super().__init__
        self._ref_model = None

        super().__init__(args)

    # ------------------------------------------------------------------
    # _setup_models -- attach frozen reference policy for KL penalty
    # ------------------------------------------------------------------

    def _setup_models(self) -> None:
        """Build all DRO model components then freeze a reference copy."""
        super()._setup_models()

        if self.use_grpo and self.grpo_kl_coeff > 0.0:
            import copy
            self._ref_model = copy.deepcopy(self.model)
            for p in self._ref_model.parameters():
                p.requires_grad_(False)
            self._ref_model.eval()
            logger.info("[GRPO] Reference policy frozen for KL penalty.")

    # ------------------------------------------------------------------
    # Rollout: sample pred_x0 AND collect log-probs along trajectory
    # ------------------------------------------------------------------

    def _grpo_rollout(
        self,
        content_images: torch.Tensor,
        style_images: torch.Tensor,
        style_source: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a short denoising rollout, collecting step log-probs.

        Args:
            content_images: (B, C, H, W) content conditioning.
            style_images: (B, C, H, W) style conditioning.
            style_source: (B, C, H, W) or None.

        Returns:
            pred_x0: (B, C, H, W) final denoised sample.
            rollout_log_prob: (B,) sum of step log-probs along trajectory.
        """
        bsz, C, H, W = content_images.shape
        device = content_images.device

        x = torch.randn(bsz, C, H, W, device=device)
        num_train_ts = self.noise_scheduler.config.num_train_timesteps
        step_size = num_train_ts // self.grpo_sample_steps
        ts_schedule = list(range(num_train_ts - 1, 0, -step_size))[: self.grpo_sample_steps]

        log_probs = torch.zeros(bsz, device=device)
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.eval()

        with torch.inference_mode():
            for t_val in ts_schedule:
                t = torch.full((bsz,), t_val, device=device, dtype=torch.long)

                if self.use_fst:
                    out = unwrapped(
                        x, t, content_images, style_source, style_images,
                        self.args.content_encoder_downsample_size,
                    )
                    noise_pred = out["noise_pred"]
                else:
                    noise_pred, _ = unwrapped(
                        x, t, style_images, content_images,
                        self.args.content_encoder_downsample_size,
                    )

                # Gaussian log p: -0.5 * ||noise_pred - noise||^2 per sample
                noise_sample = (x - self.noise_scheduler.alphas_cumprod[t_val] ** 0.5 * x) / (
                    (1 - self.noise_scheduler.alphas_cumprod[t_val]) ** 0.5 + 1e-8
                )
                step_lp = -0.5 * (
                    (noise_pred - noise_sample) ** 2
                ).mean(dim=(1, 2, 3))
                log_probs = log_probs + step_lp

                x = self.noise_scheduler.step(noise_pred, t[0], x).prev_sample

        unwrapped.train()
        return x, log_probs

    # ------------------------------------------------------------------
    # Recompute log-probs under current (updated) policy -- WITH grad
    # ------------------------------------------------------------------

    def _recompute_log_probs(
        self,
        pred_x0_detached: torch.Tensor,
        content_images: torch.Tensor,
        style_images: torch.Tensor,
        style_source: torch.Tensor | None,
        model=None,
    ) -> torch.Tensor:
        """Recompute rollout log-probs under a given policy (with grad).

        Single-step surrogate: re-noise pred_x0 to a random t, then
        measure log p_theta(noise | noisy_x0, conditioning).

        Args:
            pred_x0_detached: (B, C, H, W) detached rollout sample.
            content_images: (B, C, H, W).
            style_images: (B, C, H, W).
            style_source: (B, C, H, W) or None.
            model: policy to evaluate (defaults to self.model).

        Returns:
            log_probs: (B,) per-sample log-probability.
        """
        if model is None:
            model = self.model

        bsz = pred_x0_detached.shape[0]
        device = pred_x0_detached.device
        num_train_ts = self.noise_scheduler.config.num_train_timesteps
        max_ts = max(1, int(num_train_ts * self.dro_max_timestep_frac))

        noise = torch.randn_like(pred_x0_detached)
        t = torch.randint(0, max_ts, (bsz,), device=device).long()
        noisy = self.noise_scheduler.add_noise(pred_x0_detached, noise, t)

        if self.use_fst:
            out = model(
                noisy, t, content_images, style_source, style_images,
                self.args.content_encoder_downsample_size,
            )
            noise_pred = out["noise_pred"]
        else:
            noise_pred, _ = model(
                noisy, t, style_images, content_images,
                self.args.content_encoder_downsample_size,
            )

        log_probs = -0.5 * (
            (noise_pred - noise) ** 2
        ).mean(dim=(1, 2, 3))
        return log_probs

    # ------------------------------------------------------------------
    # GRPO loss
    # ------------------------------------------------------------------

    def _compute_grpo_loss(
        self,
        content_images: torch.Tensor,
        style_images: torch.Tensor,
        style_source: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Core GRPO loss computation.

        Generates G rollouts, scores them, normalizes within group,
        then clips the importance ratio and computes PG loss.

        Args:
            content_images: (B, C, H, W).
            style_images: (B, C, H, W).
            style_source: (B, C, H, W) or None.

        Returns:
            grpo_loss: scalar tensor.
            metrics: dict of float metrics for logging.
        """
        G = self.grpo_group_size
        bsz = content_images.shape[0]
        device = content_images.device

        # -- G rollouts, each (B, C, H, W) and (B,) log-probs ------------
        all_x0 = []
        all_old_lp = []
        for _ in range(G):
            x0, lp = self._grpo_rollout(content_images, style_images, style_source)
            all_x0.append(x0)
            all_old_lp.append(lp)

        # Stack: (G, B, C, H, W) and (G, B)
        x0_group = torch.stack(all_x0, dim=0)
        old_lp_group = torch.stack(all_old_lp, dim=0)

        # -- Score all G*B samples with reward module ---------------------
        if self.reward_module is None:
            return torch.tensor(0.0, device=device), {}

        rewards = torch.zeros(G, bsz, device=device)
        with torch.no_grad():
            for g in range(G):
                pred_01 = self._denorm(x0_group[g].detach())
                c_01 = self._match_spatial(self._denorm(content_images), pred_01)
                s_01 = self._match_spatial(self._denorm(style_images), pred_01)
                r, _ = self.reward_module(
                    pred_images=pred_01,
                    content_images=c_01,
                    style_images=s_01,
                )
                rewards[g] = r.clamp(-self.grpo_reward_clip, self.grpo_reward_clip)

        # -- Group-relative normalization (GRPO baseline) -----------------
        r_mean = rewards.mean(dim=0, keepdim=True)       # (1, B)
        r_std = rewards.std(dim=0, keepdim=True) + 1e-8  # (1, B)
        advantages = (rewards - r_mean) / r_std           # (G, B)

        # -- Clipped importance-weighted PG loss (PPO-style clip) ---------
        grpo_losses = []
        kl_penalties = []
        for g in range(G):
            # New log-probs under current policy (WITH grad)
            new_lp = self._recompute_log_probs(
                x0_group[g].detach(), content_images, style_images, style_source
            )
            old_lp = old_lp_group[g].detach()

            ratio = torch.exp(new_lp - old_lp)         # (B,)
            adv = advantages[g]                         # (B,)

            clipped = torch.clamp(
                ratio, 1.0 - self.grpo_clip_eps, 1.0 + self.grpo_clip_eps
            )
            pg = -torch.min(ratio * adv, clipped * adv).mean()
            grpo_losses.append(pg)

            # Optional KL penalty against frozen reference policy
            if self._ref_model is not None and self.grpo_kl_coeff > 0.0:
                ref_lp = self._recompute_log_probs(
                    x0_group[g].detach(),
                    content_images, style_images, style_source,
                    model=self._ref_model,
                )
                kl = (old_lp.detach() - ref_lp.detach()).mean()
                kl_penalties.append(kl)

        grpo_loss = torch.stack(grpo_losses).mean()

        metrics: dict[str, float] = {
            "grpo/loss": grpo_loss.item(),
            "grpo/reward_mean": rewards.mean().item(),
            "grpo/reward_std": rewards.std().item(),
            "grpo/advantage_mean": advantages.mean().item(),
        }

        if kl_penalties:
            kl_mean = torch.stack(kl_penalties).mean()
            grpo_loss = grpo_loss + self.grpo_kl_coeff * kl_mean
            metrics["grpo/kl"] = kl_mean.item()

        return grpo_loss, metrics

    # ------------------------------------------------------------------
    # train_step
    # ------------------------------------------------------------------

    def train_step(
        self,
        samples: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """DRO losses + GRPO policy gradient loss."""
        total_loss, loss_dict = super().train_step(samples)

        if not self.use_grpo:
            return total_loss, loss_dict

        global_step = getattr(self, "global_step", 0)
        if global_step < self.grpo_warmup_steps:
            loss_dict["grpo/warmup_active"] = 1.0
            return total_loss, loss_dict

        content_images = samples["content_image"]
        style_images = samples["style_image"]
        style_source = samples.get("style_source_image")

        grpo_loss, grpo_metrics = self._compute_grpo_loss(
            content_images, style_images, style_source
        )
        scaled = self.grpo_pg_weight * grpo_loss
        total_loss = total_loss + scaled

        loss_dict.update(grpo_metrics)
        loss_dict["grpo/scaled_loss"] = scaled.item()

        return total_loss, loss_dict

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(self, is_final: bool = False) -> None:
        """Appends grpo_config to training_state.pt."""
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
            return

        training_state = torch.load(
            state_path, map_location="cpu", weights_only=True
        )
        training_state["grpo_config"] = {
            "use_grpo": self.use_grpo,
            "grpo_group_size": self.grpo_group_size,
            "grpo_clip_eps": self.grpo_clip_eps,
            "grpo_pg_weight": self.grpo_pg_weight,
            "grpo_sample_steps": self.grpo_sample_steps,
            "grpo_kl_coeff": self.grpo_kl_coeff,
            "grpo_reward_clip": self.grpo_reward_clip,
        }
        torch.save(training_state, state_path)
        logger.info(f"[OK] GRPO config saved to {state_path}")