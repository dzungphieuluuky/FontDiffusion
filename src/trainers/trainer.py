import logging
import math
import os
from dataclasses import asdict
from pathlib import Path
import traceback

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from torchvision import transforms
from tqdm.auto import tqdm


from src.configs.fontdiffuser import get_parser
from src.dataset import FontDataset, CollateFN
from src import (
    ContentPerceptualLoss,
    FontDiffuserModel,
    build_content_encoder,
    build_ddpm_scheduler,
    build_scr,
    build_style_encoder,
    build_unet,
)
from src.tools.utils import (
    normalize_mean_std,
    reNormalize_img,
    save_args_to_yaml,
    x0_from_epsilon,
)

from src.tools.utilities import (
    find_checkpoint,
    load_model_checkpoint,
    save_model_checkpoint,
    HFTqdm,
)

from src.trainers.training_config import TrainingConfig

logger = logging.getLogger(__name__)


class FontDiffuserTrainer:
    """Main trainer class for FontDiffuser."""

    def __init__(self, args):
        self.args = args
        self.config = self._create_config(args)
        self.config.validate()

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            project_dir=f"{self.args.output_dir}/{self.args.logging_dir}",
        )

        self.global_step = 0
        self.current_epoch = 0

        # Will be initialized in setup()
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.train_dataloader = None
        self.noise_scheduler = None
        self.perceptual_loss = None
        self.scr = None

    def _create_config(self, args) -> TrainingConfig:
        """Create TrainingConfig from parsed args."""
        return TrainingConfig(
            learning_rate=args.learning_rate,
            train_batch_size=args.train_batch_size,
            max_train_steps=args.max_train_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            lr_scheduler=args.lr_scheduler,
            lr_warmup_steps=args.lr_warmup_steps,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_weight_decay=args.adam_weight_decay,
            adam_epsilon=args.adam_epsilon,
            max_grad_norm=args.max_grad_norm,
            perceptual_coefficient=args.perceptual_coefficient,
            offset_coefficient=args.offset_coefficient,
            sc_coefficient=(
                args.sc_coefficient if hasattr(args, "sc_coefficient") else 1.0
            ),
            style_transform_coefficient=getattr(
                args, "style_transform_coefficient", 0.1
            ),
            phase_1=args.phase_1,
            phase_2=args.phase_2,
            drop_prob=getattr(args, "drop_prob", 0.1),
            enable_style_transform=getattr(args, "enable_style_transform", False),
        )

    def setup(self):
        """Setup all components for training."""
        if self.accelerator.is_main_process:
            Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)

        if self.args.seed is not None:
            set_seed(self.args.seed)

        self._setup_models()
        self._setup_data()
        self._setup_optimizer()
        self._wrap_components()

        if self.accelerator.is_main_process:
            self._setup_logging()

    def _setup_models(self):
        """Initialize all model components."""
        # Build core components
        unet = build_unet(args=self.args)
        style_encoder = build_style_encoder(args=self.args)
        content_encoder = build_content_encoder(args=self.args)
        self.noise_scheduler = build_ddpm_scheduler(self.args)
        # Load phase 1 checkpoints if specified
        if self.args.phase_1_ckpt_dir is not None:
            self._load_phase1_checkpoints(
                unet=unet,
                style_encoder=style_encoder,
                content_encoder=content_encoder,
                ckpt_dir=self.args.phase_1_ckpt_dir,
            )

        # Create main model
        self.model = FontDiffuserModel(
            unet=unet,
            style_encoder=style_encoder,
            content_encoder=content_encoder,
        )

        # Perceptual loss (always used)
        self.perceptual_loss = ContentPerceptualLoss()

        # SCR for phase 2 (optional)
        self.scr = None
        if self.config.phase_2:
            self.scr = build_scr(args=self.args)
            if hasattr(self.args, "scr_ckpt_path") and self.args.scr_ckpt_path:
                self._load_scr_checkpoint(self.args.scr_ckpt_path)
            self.scr.requires_grad_(False)

    def _load_phase1_checkpoints(
        self, unet, style_encoder, content_encoder, ckpt_dir: str
    ):
        """Load phase 1 checkpoints with error handling."""
        logger.info("Loading Phase 1 checkpoints...")
        components = {
            "unet": unet,
            "style_encoder": style_encoder,
            "content_encoder": content_encoder,
        }

        for name, component in components.items():
            try:
                ckpt_path = find_checkpoint(ckpt_dir, name)
                if not ckpt_path.exists():
                    logger.warning(f"Checkpoint for {name} not found at {ckpt_path}")
                    continue

                state_dict = load_model_checkpoint(ckpt_path)
                component.load_state_dict(state_dict)
                logger.info(f"Loaded {name} from {ckpt_path}")

            except Exception as e:
                logger.error(f"Failed to load {name} from {ckpt_dir}: {e}")
                logger.debug(traceback.format_exc())

    def _load_scr_checkpoint(self, ckpt_path: str):
        """Load SCR checkpoint with error handling."""
        try:
            state_dict = load_model_checkpoint(ckpt_path)
            self.scr.load_state_dict(state_dict)
            logger.info(f"Loaded SCR from {ckpt_path}")
        except FileNotFoundError:
            logger.warning(f"SCR checkpoint not found at {ckpt_path}")
        except OSError as e:
            logger.error(f"OS error loading SCR checkpoint: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading SCR checkpoint: {e}")
            logger.debug(traceback.format_exc())

    def _setup_data(self):
        """Setup data transforms and dataloaders."""
        content_transforms = transforms.Compose(
            [
                transforms.Resize(
                    self.args.content_image_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        style_transforms = transforms.Compose(
            [
                transforms.Resize(
                    self.args.style_image_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        target_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (self.args.resolution, self.args.resolution),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        train_dataset = FontDataset(
            args=self.args,
            phase="train",
            transforms=[content_transforms, style_transforms, target_transforms],
            scr=self.config.phase_2,
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.config.train_batch_size,
            collate_fn=CollateFN(),
            num_workers=getattr(self.args, "num_workers", 4),
            pin_memory=True,
            persistent_workers=True,
        )

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Scale learning rate if requested
        learning_rate = self.config.learning_rate
        if self.args.scale_lr:
            learning_rate *= (
                self.config.gradient_accumulation_steps
                * self.config.train_batch_size
                * self.accelerator.num_processes
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.adam_weight_decay,
            eps=self.config.adam_epsilon,
        )

        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_warmup_steps
            * self.config.gradient_accumulation_steps,
            num_training_steps=self.config.max_train_steps
            * self.config.gradient_accumulation_steps,
        )

    def _wrap_components(self):
        """Wrap components with accelerator."""
        # Prepare trainable components
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        )

        # Move non-trainable components to device
        if self.scr is not None:
            self.scr = self.scr.to(self.accelerator.device)

    def _setup_logging(self):
        """Setup logging and tracking."""
        self.accelerator.init_trackers(self.args.experience_name)

        # Save configuration
        save_args_to_yaml(
            args=self.args,
            output_file=f"{self.args.output_dir}/{self.args.experience_name}_config.yaml",
        )

        # Log configuration
        config_dict = {
            "training_config": asdict(self.config),
        }
        self.accelerator.log(config_dict)

    def apply_classifier_free_guidance(
        self,
        content_images: torch.Tensor,
        style_images: torch.Tensor,
        drop_prob: float,
        samples: dict[str, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply classifier-free guidance by masking some samples.

        Returns clones of inputs to avoid in-place modifications affecting gradients.
        """
        # Clone inputs to avoid in-place modifications
        content_images = content_images.clone()
        style_images = style_images.clone()

        bsz = content_images.shape[0]
        context_mask = torch.bernoulli(
            torch.zeros(bsz, device=content_images.device) + drop_prob
        )

        # Mask content and style images
        for i, mask_value in enumerate(context_mask):
            if mask_value == 1:
                content_images[i, :, :, :] = 1.0
                style_images[i, :, :, :] = 1.0

        # Mask source style images if provided
        if samples is not None and "source_style_image" in samples:
            source_style_images = samples["source_style_image"].clone()
            for i, mask_value in enumerate(context_mask):
                if mask_value == 1:
                    source_style_images[i, :, :, :] = 1.0
            samples["source_style_image"] = source_style_images

        return content_images, style_images

    def compute_losses(
        self,
        noise_pred: torch.Tensor,
        noise: torch.Tensor,
        offset_out_sum: torch.Tensor,
        noisy_target_images: torch.Tensor,
        nonorm_target_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        """Compute all losses for the training step."""
        # Diffusion loss
        diff_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        # Offset loss (divided by 2 as per original implementation)
        offset_loss = offset_out_sum / 2.0

        # Perceptual loss
        pred_original_sample_norm = x0_from_epsilon(
            scheduler=self.noise_scheduler,
            noise_pred=noise_pred,
            x_t=noisy_target_images,
            timesteps=timesteps,
        )
        pred_original_sample = reNormalize_img(pred_original_sample_norm)
        norm_pred_ori = normalize_mean_std(pred_original_sample)
        norm_target_ori = normalize_mean_std(nonorm_target_images)

        percep_loss = self.perceptual_loss.calculate_loss(
            generated_images=norm_pred_ori,
            target_images=norm_target_ori,
            device=self.accelerator.device,
        )

        # Total loss
        total_loss = (
            diff_loss
            + self.config.perceptual_coefficient * percep_loss
            + self.config.offset_coefficient * offset_loss
        )

        loss_dict = {
            "diff_loss": diff_loss.item(),
            "percep_loss": percep_loss.item(),
            "offset_loss": offset_loss.item(),
            "train_loss": total_loss.item(),
        }

        return total_loss, loss_dict, pred_original_sample_norm

    def compute_phase2_loss(
        self,
        pred_original_sample_norm: torch.Tensor,
        target_images: torch.Tensor,
        neg_images: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SCR loss for phase 2 training.

        Args:
            pred_original_sample_norm: [B, 3, resolution, resolution]
            target_images: [B, 3, resolution, resolution]
            neg_images: [B, num_neg, 3, resolution, resolution] ← 5D, DO NOT RESHAPE

        Returns:
            sc_loss: scalar tensor
        """
        # Validate input shapes
        assert pred_original_sample_norm.dim() == 4, (
            f"Expected pred shape [B, C, H, W], got {pred_original_sample_norm.shape}"
        )
        assert target_images.dim() == 4, (
            f"Expected target shape [B, C, H, W], got {target_images.shape}"
        )
        assert neg_images.dim() == 5, (
            f"Expected neg_images shape [B, num_neg, C, H, W], got {neg_images.shape}"
        )

        sample_emb, pos_emb, neg_emb = self.scr(
            pred_original_sample_norm,  # [B, 3, resolution, resolution]
            target_images,  # [B, 3, resolution, resolution]
            neg_images,  # [B, num_neg, 3, resolution, resolution]
            nce_layers=self.args.nce_layers,
        )

        sc_loss = self.scr.calculate_nce_loss(
            sample_s=sample_emb,
            pos_s=pos_emb,
            neg_s=neg_emb,
        )

        return sc_loss

    def train_step(
        self,
        samples: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Perform a single training step."""
        self.model.train()

        # Extract and prepare inputs
        content_images = samples["content_image"]
        style_images = samples["style_image"]
        target_images = samples["target_image"]
        nonorm_target_images = samples["nonorm_target_image"]

        # Generate noise and timesteps
        noise = torch.randn_like(target_images)
        bsz = target_images.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=target_images.device,
        ).long()

        # Add noise to targets
        noisy_target_images = self.noise_scheduler.add_noise(
            target_images, noise, timesteps
        )

        # Apply classifier-free guidance
        content_images, style_images = self.apply_classifier_free_guidance(
            content_images,
            style_images,
            self.config.drop_prob,
            samples=samples,
        )

        # Forward pass
        source_style_images = None
        if self.config.enable_style_transform and "source_style_image" in samples:
            source_style_images = samples["source_style_image"]

        noise_pred, offset_out_sum = self.model(
            x_t=noisy_target_images,
            timesteps=timesteps,
            style_images=style_images,
            content_images=content_images,
            content_encoder_downsample_size=self.args.content_encoder_downsample_size,
        )

        # Compute losses
        total_loss, loss_dict, pred_original_sample_norm = self.compute_losses(
            noise_pred=noise_pred,
            noise=noise,
            offset_out_sum=offset_out_sum,
            noisy_target_images=noisy_target_images,
            nonorm_target_images=nonorm_target_images,
            timesteps=timesteps,
        )

        # SCR loss for phase 2
        if self.config.phase_2 and self.scr is not None:
            neg_images = samples.get("neg_images")
            if neg_images is not None:
                sc_loss = self.compute_phase2_loss(
                    pred_original_sample_norm=pred_original_sample_norm,
                    target_images=target_images,
                    neg_images=neg_images,
                )
                total_loss += self.config.sc_coefficient * sc_loss
                loss_dict["sc_loss"] = sc_loss.item()

        return total_loss, loss_dict

    def save_checkpoint(self, is_final: bool = False):
        """Save training checkpoint."""
        if not self.accelerator.is_main_process:
            return

        # Determine checkpoint name
        if is_final:
            save_dir = Path(self.args.output_dir) / "final"
        else:
            save_dir = (
                Path(self.args.output_dir) / f"checkpoint_step_{self.global_step}"
            )

        save_dir.mkdir(parents=True, exist_ok=True)

        # Unwrap model for saving
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Save individual components
        save_model_checkpoint(
            unwrapped_model.config.unet.state_dict(), save_dir / "unet.safetensors"
        )
        save_model_checkpoint(
            unwrapped_model.config.style_encoder.state_dict(),
            save_dir / "style_encoder.safetensors",
        )
        save_model_checkpoint(
            unwrapped_model.config.content_encoder.state_dict(),
            save_dir / "content_encoder.safetensors",
        )

        # Save full model if requested
        if getattr(self.args, "save_full_model", False):
            save_model_checkpoint(
                unwrapped_model.state_dict(), save_dir / "full_model.safetensors"
            )

        # Save SCR for phase 2
        if self.config.phase_2 and self.scr is not None:
            save_model_checkpoint(self.scr.state_dict(), save_dir / "scr.safetensors")

        logger.info(f"Saved checkpoint to {save_dir}")
        self.accelerator.log(
            {"checkpoint_saved": True, "checkpoint_step": self.global_step}
        )

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training checkpoint.

        Returns:
            bool: True if checkpoint was loaded successfully
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Load model state
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(checkpoint["model_state_dict"])

            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            logger.info(
                f"Resuming from step {self.global_step}, epoch {self.current_epoch}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return False

    def train(self):
        """Main training loop."""
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.config.gradient_accumulation_steps
        )
        num_train_epochs = math.ceil(
            self.config.max_train_steps / num_update_steps_per_epoch
        )

        # Resume from checkpoint if specified
        if (
            hasattr(self.args, "resume_from_checkpoint")
            and self.args.resume_from_checkpoint
        ):
            if not self.load_checkpoint(self.args.resume_from_checkpoint):
                logger.warning("Starting training from scratch")

        # Setup progress bar
        progress_bar = HFTqdm(
            range(self.config.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
            desc="Training",
        )

        # Initialize tracking variables
        train_loss_accum = 0.0
        loss_accum_count = 0

        # Training loop
        for epoch in range(self.current_epoch, num_train_epochs):
            self.current_epoch = epoch

            for step, samples in enumerate(self.train_dataloader):
                # Skip steps if resuming
                if self.global_step >= self.config.max_train_steps:
                    break

                with self.accelerator.accumulate(self.model):
                    # Forward pass and loss computation
                    loss, loss_dict = self.train_step(samples)

                    # Backward pass
                    self.accelerator.backward(loss)

                    # Gradient clipping
                    if self.accelerator.sync_gradients:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )

                    # Optimization step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    # Update tracking variables
                    train_loss_accum += loss.detach().item()
                    loss_accum_count += 1

                    # Sync and log
                    if self.accelerator.sync_gradients:
                        # Update progress bar
                        progress_bar.update(1)
                        self.global_step += 1

                        # Compute average loss
                        avg_train_loss = train_loss_accum / loss_accum_count

                        # Prepare log dictionary
                        log_dict = {
                            "train_loss": avg_train_loss,
                            "learning_rate": self.lr_scheduler.get_last_lr()[0],
                            "epoch": epoch + step / len(self.train_dataloader),
                            "global_step": self.global_step,
                            "grad_norm": (
                                grad_norm.item()
                                if self.accelerator.sync_gradients
                                else 0.0
                            ),
                        }

                        # Add individual losses
                        for loss_name, loss_val in loss_dict.items():
                            log_dict[f"train/{loss_name}"] = loss_val

                        # Log to tracker
                        self.accelerator.log(log_dict, step=self.global_step)

                        # Reset accumulators
                        train_loss_accum = 0.0
                        loss_accum_count = 0

                        # Log to console
                        if self.global_step % self.args.log_interval == 0:
                            logger.info(
                                f"Step {self.global_step}: "
                                f"loss={avg_train_loss:.4f}, "
                                f"lr={self.lr_scheduler.get_last_lr()[0]:.6f}, "
                                f"grad_norm={grad_norm.item():.4f}"
                            )

                        # Save checkpoint
                        if (
                            self.global_step % self.args.ckpt_interval == 0
                            and self.accelerator.is_main_process
                        ):
                            self.save_checkpoint()

                # Update progress bar description
                progress_bar.set_postfix(
                    loss=loss.detach().item(),
                    lr=self.lr_scheduler.get_last_lr()[0],
                    step=self.global_step,
                )

                if self.global_step >= self.config.max_train_steps:
                    break

            if self.global_step >= self.config.max_train_steps:
                break

        progress_bar.close()

        # Save final checkpoint
        if self.accelerator.is_main_process:
            self.save_checkpoint(is_final=True)

        self.accelerator.end_training()
