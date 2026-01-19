import logging
import math
from dataclasses import asdict
from pathlib import Path
import traceback

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from torchvision import transforms
from omegaconf import DictConfig, OmegaConf

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
    x0_from_epsilon,
)
from src.tools.utilities import (
    find_checkpoint,
    load_model_checkpoint,
    save_model_checkpoint,
    HFTqdm,
)

logger = logging.getLogger("FontDiffuserTrainer")


class FontDiffuserTrainer:
    """Main trainer class for FontDiffuser with Hydra configuration support."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.config = cfg.training
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
            log_with=self.config.logging.report_to,
            project_dir=f"{self.config.output_dir}/{self.config.logging.logging_dir}",
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

    def setup(self):
        """Setup all components for training."""
        if self.accelerator.is_main_process:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        if self.config.seed is not None:
            set_seed(self.config.seed)

        self._setup_models()
        self._setup_data()
        self._setup_optimizer()
        self._wrap_components()

        if self.accelerator.is_main_process:
            self._setup_logging()

    def _setup_models(self):
        """Initialize all model components."""
        # Build core components
        unet = build_unet(cfg=self.cfg)
        style_encoder = build_style_encoder(cfg=self.cfg)
        content_encoder = build_content_encoder(cfg=self.cfg)
        self.noise_scheduler = build_ddpm_scheduler(self.cfg)
        
        # Load phase 1 checkpoints if specified
        if self.config.checkpoint.phase_1_ckpt_dir is not None:
            self._load_phase1_checkpoints(
                unet=unet,
                style_encoder=style_encoder,
                content_encoder=content_encoder,
                ckpt_dir=self.config.checkpoint.phase_1_ckpt_dir,
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
        if self.config.phase.phase_2:
            self.scr = build_scr(cfg=self.cfg)
            if self.config.checkpoint.scr_ckpt_path:
                self._load_scr_checkpoint(self.config.checkpoint.scr_ckpt_path)
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
                    tuple(self.cfg.model.content_encoder.image_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        style_transforms = transforms.Compose(
            [
                transforms.Resize(
                    tuple(self.cfg.model.style_encoder.image_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        target_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (self.cfg.model.unet.sample_size, self.cfg.model.unet.sample_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        train_dataset = FontDataset(
            cfg=self.cfg,
            phase="train",
            transforms=[content_transforms, style_transforms, target_transforms],
            scr=self.config.phase.phase_2,
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.config.train_batch_size,
            collate_fn=CollateFN(),
            num_workers=self.config.dataloader.num_workers,
            pin_memory=self.config.dataloader.pin_memory,
            persistent_workers=self.config.dataloader.persistent_workers,
        )

    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Scale learning rate if requested
        learning_rate = self.config.optimizer.learning_rate
        if self.config.optimizer.scale_lr:
            learning_rate *= (
                self.config.gradient_accumulation_steps
                * self.config.train_batch_size
                * self.accelerator.num_processes
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            weight_decay=self.config.optimizer.adam_weight_decay,
            eps=self.config.optimizer.adam_epsilon,
        )

        self.lr_scheduler = get_scheduler(
            self.config.optimizer.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.optimizer.lr_warmup_steps
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
        self.accelerator.init_trackers(self.config.logging.experience_name)

        # Save configuration
        config_path = Path(self.config.output_dir) / f"{self.config.logging.experience_name}_config.yaml"
        with open(config_path, 'w') as f:
            OmegaConf.save(self.cfg, f)

        # Log configuration
        config_dict = {
            "training_config": OmegaConf.to_container(self.config, resolve=True),
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
            + self.config.loss.perceptual_coefficient * percep_loss
            + self.config.loss.offset_coefficient * offset_loss
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
        """Compute SCR loss for phase 2 training."""
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
            pred_original_sample_norm,
            target_images,
            neg_images,
            nce_layers=self.cfg.model.scr.nce_layers,
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
            self.config.training_params.drop_prob,
            samples=samples,
        )

        # Forward pass
        noise_pred, offset_out_sum = self.model(
            x_t=noisy_target_images,
            timesteps=timesteps,
            style_images=style_images,
            content_images=content_images,
            content_encoder_downsample_size=self.cfg.model.content_encoder.out_channels,
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
        if self.config.phase.phase_2 and self.scr is not None:
            neg_images = samples.get("neg_images")
            if neg_images is not None:
                sc_loss = self.compute_phase2_loss(
                    pred_original_sample_norm=pred_original_sample_norm,
                    target_images=target_images,
                    neg_images=neg_images,
                )
                total_loss += self.config.loss.sc_coefficient * sc_loss
                loss_dict["sc_loss"] = sc_loss.item()

        return total_loss, loss_dict

    def save_checkpoint(self, is_final: bool = False):
        """Save training checkpoint."""
        if not self.accelerator.is_main_process:
            return

        # Determine checkpoint name
        if is_final:
            save_dir = Path(self.config.output_dir) / "final"
        else:
            save_dir = (
                Path(self.config.output_dir) / f"checkpoint_step_{self.global_step}"
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
        if self.config.checkpoint.save_full_model:
            save_model_checkpoint(
                unwrapped_model.state_dict(), save_dir / "full_model.safetensors"
            )

        # Save SCR for phase 2
        if self.config.phase.phase_2 and self.scr is not None:
            save_model_checkpoint(self.scr.state_dict(), save_dir / "scr.safetensors")

        # Save optimizer and scheduler states
        torch.save(
            {
                "global_step": self.global_step,
                "epoch": self.current_epoch,
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "config": OmegaConf.to_container(self.config, resolve=True),
            },
            save_dir / "training_state.pt",
        )

        logger.info(f"Saved checkpoint to {save_dir}")
        self.accelerator.log(
            {"checkpoint_saved": True, "checkpoint_step": self.global_step}
        )

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Load model state
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.load_state_dict(checkpoint["model_state_dict"])

            # Load optimizer and scheduler
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

            # Restore training state
            self.global_step = checkpoint["global_step"]
            self.current_epoch = checkpoint["epoch"]

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
        if self.config.checkpoint.resume_from_checkpoint:
            if not self.load_checkpoint(self.config.checkpoint.resume_from_checkpoint):
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
                            self.model.parameters(), self.config.optimizer.max_grad_norm
                        )

                    # Optimization step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

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
                        if self.global_step % self.config.logging.log_interval == 0:
                            logger.info(
                                f"Step {self.global_step}: "
                                f"loss={avg_train_loss:.4f}, "
                                f"lr={self.lr_scheduler.get_last_lr()[0]:.6f}, "
                                f"grad_norm={grad_norm.item():.4f}"
                            )

                        # Save checkpoint
                        if (
                            self.global_step % self.config.checkpoint.ckpt_interval == 0
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