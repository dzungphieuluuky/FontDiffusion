"""
Trainer class for FontDiffuserWithFST.
Uses Hydra DictConfig for configuration management.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms

from src.dataset.font_dataset_fst import FontDataset as FontDatasetFST
from src.dataset.collate_fn_fst import CollateFN as CollateFNFST
from src import (
    ContentPerceptualLoss,
    FontDiffuserModel,
    build_content_encoder,
    build_ddpm_scheduler,
    build_scr,
    build_style_encoder,
    build_unet,
)
from src.models.fst_model import FontDiffuserWithFST
from src.modules.msse import MultiScaleStyleEncoder
from src.modules.fst import FontStyleTransformationModule
from src.tools.utilities import (
    find_checkpoint,
    HFTqdm,
    load_model_checkpoint,
    save_model_checkpoint,
)
from src.tools.utils import (
    normalize_mean_std,
    reNormalize_img,
    x0_from_epsilon,
)
from src.trainers.trainer import FontDiffuserTrainer

logger = logging.getLogger("FontDiffuserFSTTrainer")


class FontDiffuserFSTTrainer(FontDiffuserTrainer):
    """Trainer for FontDiffuserWithFST model with MSSE and FST modules."""

    def __init__(self, cfg: DictConfig):
        """Initialize FST trainer with Hydra config.

        Args:
            cfg: Hydra DictConfig with FST and training parameters
        """
        # Extract FST-specific parameters from Hydra config
        self.use_fst = cfg.get("use_fst", True)
        self.freeze_original_encoders = cfg.get("freeze_original_encoders", False)
        self.style_source_same_prob = cfg.get("style_source_same_prob", 0.5)
        self.save_full_model = cfg.get("save_full_model", True)

        # FST module configuration
        fst_channels = cfg.get("fst_feature_channels", [64, 128, 256, 512, 1024])
        self.fst_feature_channels = (
            fst_channels if isinstance(fst_channels, list) else list(fst_channels)
        )

        self.fst_num_queries = cfg.get("fst_num_queries", 220)
        self.fst_query_dim = cfg.get("fst_query_dim", 128)
        self.fst_num_scales = cfg.get("fst_num_scales", 5)

        # Parent initialization
        super().__init__(cfg)

    def _setup_models(self):
        """Initialize FST model components with proper configuration."""
        logger.info("Building core model components...")

        # Build base components
        unet = build_unet(cfg=self.cfg)
        style_encoder = build_style_encoder(cfg=self.cfg)
        content_encoder = build_content_encoder(cfg=self.cfg)
        self.noise_scheduler = build_ddpm_scheduler(self.cfg)

        # Load phase 1 checkpoints if available
        if hasattr(self.cfg, "phase_1_ckpt_dir") and self.cfg.phase_1_ckpt_dir:
            self._load_phase1_checkpoints(
                unet=unet,
                style_encoder=style_encoder,
                content_encoder=content_encoder,
                ckpt_dir=self.cfg.phase_1_ckpt_dir,
            )

        # Create base model
        base_model = FontDiffuserModel(
            unet=unet,
            style_encoder=style_encoder,
            content_encoder=content_encoder,
        )

        # Wrap with FST if enabled
        if self.use_fst:
            logger.info("Building FontDiffuserWithFST model")
            logger.info(f"  Feature channels: {self.fst_feature_channels}")
            logger.info(f"  Num queries: {self.fst_num_queries}")
            logger.info(f"  Query dim: {self.fst_query_dim}")

            self.model = FontDiffuserWithFST(
                base_model,
                feature_channels=self.fst_feature_channels,
                num_queries=self.fst_num_queries,
                query_dim=self.fst_query_dim,
                num_scales=self.fst_num_scales,
            )

            # Optionally freeze original encoders
            if self.freeze_original_encoders:
                logger.info("Freezing original encoders")
                for param in self.model.style_encoder.parameters():
                    param.requires_grad = False
                for param in self.model.content_encoder.parameters():
                    param.requires_grad = False

                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.model.parameters())
                logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        else:
            logger.info("Using base FontDiffuser model (no FST)")
            self.model = base_model

        # Perceptual loss
        self.perceptual_loss = ContentPerceptualLoss()

        # SCR for phase 2
        self.scr = None
        if self.cfg.get("phase_2", False):
            self.scr = build_scr(cfg=self.cfg)
            if self.cfg.get("scr_ckpt_path"):
                self._load_scr_checkpoint(self.cfg.scr_ckpt_path)
            self.scr.requires_grad_(False)

    def _setup_data(self):
        """Setup FST-compatible dataloaders."""
        logger.info("Setting up data transforms and dataloaders...")

        content_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (self.cfg.content_image_size, self.cfg.content_image_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        style_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (self.cfg.style_image_size, self.cfg.style_image_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        target_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (self.cfg.resolution, self.cfg.resolution),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # FST dataset
        train_dataset = FontDatasetFST(
            cfg=self.cfg,
            phase="train",
            transforms=[content_transforms, style_transforms, target_transforms],
            scr=self.cfg.get("phase_2", False),
            use_fst=self.use_fst,
            style_source_same_prob=self.style_source_same_prob,
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.cfg.train_batch_size,
            collate_fn=CollateFNFST(verbose=False),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

        logger.info(f"✓ Loaded FST dataset with {len(train_dataset)} samples")

    def _setup_optimizer(self):
        """Setup optimizer with Hydra config."""
        logger.info("Setting up optimizer and scheduler...")

        learning_rate = self.cfg.learning_rate
        if self.cfg.get("scale_lr", False):
            learning_rate *= (
                self.cfg.gradient_accumulation_steps
                * self.cfg.train_batch_size
                * self.accelerator.num_processes
            )

        # Select trainable parameters
        trainable_params = (
            [p for p in self.model.parameters() if p.requires_grad]
            if self.use_fst and self.freeze_original_encoders
            else self.model.parameters()
        )

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            weight_decay=self.cfg.adam_weight_decay,
            eps=self.cfg.adam_epsilon,
        )

        self.lr_scheduler = get_scheduler(
            self.cfg.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.lr_warmup_steps * self.cfg.gradient_accumulation_steps,
            num_training_steps=self.cfg.max_train_steps * self.cfg.gradient_accumulation_steps,
        )

    def apply_classifier_free_guidance(
        self,
        content_images: torch.Tensor,
        style_images: torch.Tensor,
        drop_prob: float,
        samples: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply classifier-free guidance."""
        content_images = content_images.clone()
        style_images = style_images.clone()

        bsz = content_images.shape[0]
        context_mask = torch.bernoulli(
            torch.zeros(bsz, device=content_images.device) + drop_prob
        )

        for i, mask_value in enumerate(context_mask):
            if mask_value == 1:
                content_images[i, :, :, :] = 1.0
                style_images[i, :, :, :] = 1.0

        if samples is not None and "style_source_image" in samples:
            style_source_images = samples["style_source_image"].clone()
            for i, mask_value in enumerate(context_mask):
                if mask_value == 1:
                    style_source_images[i, :, :, :] = 1.0
            samples["style_source_image"] = style_source_images

        return content_images, style_images

    def train_step(self, samples: dict) -> tuple[torch.Tensor, dict]:
        """Perform FST training step."""
        self.model.train()

        content_images = samples["content_image"]
        style_images = samples["style_image"]
        target_images = samples["target_image"]
        nonorm_target_images = samples["nonorm_target_image"]

        noise = torch.randn_like(target_images)
        bsz = target_images.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=target_images.device,
        ).long()

        noisy_target_images = self.noise_scheduler.add_noise(target_images, noise, timesteps)

        content_images, style_images = self.apply_classifier_free_guidance(
            content_images,
            style_images,
            self.cfg.drop_prob,
            samples=samples,
        )

        # Forward pass
        if self.use_fst:
            style_source_images = samples.get("style_source_image", style_images)
            outputs = self.model(
                noisy_latents=noisy_target_images,
                timestep=timesteps,
                content_img=content_images,
                style_source_img=style_source_images,
                style_target_img=style_images,
                content_encoder_downsample_size=self.cfg.content_encoder_downsample_size,
                return_dict=True,
            )
            noise_pred = outputs["noise_pred"]
            offset_out_sum = outputs.get("offset_out_sum")
        else:
            noise_pred, offset_out_sum = self.model(
                x_t=noisy_target_images,
                timesteps=timesteps,
                style_images=style_images,
                content_images=content_images,
                content_encoder_downsample_size=self.cfg.content_encoder_downsample_size,
            )

        total_loss, loss_dict, pred_original = self.compute_losses(
            noise_pred=noise_pred,
            noise=noise,
            offset_out_sum=offset_out_sum,
            noisy_target_images=noisy_target_images,
            nonorm_target_images=nonorm_target_images,
            timesteps=timesteps,
        )

        return total_loss, loss_dict

    def save_checkpoint(self, is_final: bool = False):
        """Save FST checkpoint with Hydra config."""
        if not self.accelerator.is_main_process:
            return

        save_dir = (
            Path(self.cfg.output_dir) / "final"
            if is_final
            else Path(self.cfg.output_dir) / f"checkpoint_step_{self.global_step}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.use_fst:
            logger.info("Saving FST model components")
            save_model_checkpoint(
                unwrapped_model.diffusion_unet.state_dict(),
                save_dir / "unet.safetensors",
            )
            save_model_checkpoint(
                unwrapped_model.style_encoder.state_dict(),
                save_dir / "style_encoder.safetensors",
            )
            save_model_checkpoint(
                unwrapped_model.content_encoder.state_dict(),
                save_dir / "content_encoder.safetensors",
            )
            save_model_checkpoint(
                unwrapped_model.mss_encoder.state_dict(),
                save_dir / "mss_encoder.safetensors",
            )
            save_model_checkpoint(
                unwrapped_model.fst_module.state_dict(),
                save_dir / "fst_module.safetensors",
            )

            if self.save_full_model:
                torch.save(unwrapped_model, save_dir / "total_model_fst.pth")

        # Save state
        torch.save(
            {
                "global_step": self.global_step,
                "epoch": self.current_epoch,
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "config": OmegaConf.to_container(self.cfg, resolve=True),
            },
            save_dir / "training_state.pt",
        )

        logger.info(f"✓ Saved FST checkpoint to {save_dir}")