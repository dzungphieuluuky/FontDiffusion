import argparse
import logging
import torch

from src.trainers.trainer_fst import FontDiffuserFSTTrainer
from src.modules.unicalli_improvements import (
    AsymmetricNoisingScheduler,
    ConditionalDropoutAugmentation,
)

logger = logging.getLogger(__name__)


class FontDiffuserUnicalliTrainer(FontDiffuserFSTTrainer):
    """Trainer for FontDiffuser with UniCalli improvements."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        # UniCalli specific arguments
        self.style_noise_fraction = getattr(args, "style_noise_fraction", 0.0)
        self.p_drop_content = getattr(args, "p_drop_content", 0.1)
        self.p_drop_style = getattr(args, "p_drop_style", 0.05)
        self.use_hard_negative = getattr(args, "use_hard_negative", True)
        self.curriculum_steps = getattr(args, "curriculum_steps", 1000)

        # Initialize UniCalli modules
        self.conditional_dropout = ConditionalDropoutAugmentation(
            p_drop_content=self.p_drop_content,
            p_drop_style=self.p_drop_style,
            use_hard_negative=self.use_hard_negative,
            curriculum_steps=self.curriculum_steps,
        )

    def _setup_models(self):
        super()._setup_models()
        self.asym_scheduler = AsymmetricNoisingScheduler(
            self.noise_scheduler,
            style_noise_fraction=self.style_noise_fraction,
            content_noise_fraction=0.0,
        )
        logger.info("✓ Initialized UniCalli AsymmetricNoisingScheduler")

    def _setup_logging(self):
        super()._setup_logging()
        if not self.accelerator.is_main_process:
            return

        unicalli_cfg = {
            "style_noise_fraction": self.style_noise_fraction,
            "p_drop_content": self.p_drop_content,
            "p_drop_style": self.p_drop_style,
            "use_hard_negative": self.use_hard_negative,
            "curriculum_steps": self.curriculum_steps,
        }
        self.accelerator.log({"unicalli_config": unicalli_cfg})

    def train_step(self, samples):
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

        # --- UNICALLI IMPROVEMENT 1: Asymmetric Noising Scheduler ---
        noisy_targets = self.asym_scheduler.add_target_noise(
            target_images, noise, timesteps
        )
        style_images, _ = self.asym_scheduler.add_style_noise(style_images)

        # --- UNICALLI IMPROVEMENT 2: Conditional Dropout Augmentation ---
        content_images, style_images, _ = self.conditional_dropout.apply(
            content_images, style_images, self.global_step
        )

        content_images, style_images = self.apply_classifier_free_guidance(
            content_images, style_images, self.config.drop_prob, samples
        )

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

        total_loss, loss_dict, pred_orig_norm = self.compute_losses(
            noise_pred, noise, offset_out_sum, noisy_targets, nonorm_target, timesteps
        )

        if self.config.phase_2 and self.scr and samples.get("neg_images") is not None:
            sc_loss = self.compute_phase2_loss(
                pred_orig_norm, target_images, samples["neg_images"]
            )
            total_loss += self.config.sc_coefficient * sc_loss
            loss_dict["sc_loss"] = sc_loss.item()

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
