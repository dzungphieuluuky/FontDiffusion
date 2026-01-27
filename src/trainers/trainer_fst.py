"""
Trainer class for FontDiffuserWithFST.
Extends base FontDiffuserTrainer with FST-specific functionality.
"""

import logging
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

from src.dataset.font_dataset_fst import FontDataset as FontDatasetFST
from src.dataset.collate_fn_fst import CollateFN as CollateFNFST
from src.losses.criterion import CombinedFSTLoss
from src import (
    ContentPerceptualLoss,
    FontDiffuserModel,
    build_content_encoder,
    build_ddpm_scheduler,
    build_scr,
    build_style_encoder,
    build_unet,
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
from src.trainers.trainer import FontDiffuserTrainer

logger = logging.getLogger(__name__)


class FontDiffuserFSTTrainer(FontDiffuserTrainer):
    """Trainer for FontDiffuserWithFST model with MSSE and FST modules."""

    def __init__(self, args):
        """Initialize FST trainer.

        Args:
            args: Training arguments with FST-specific parameters
        """
        # Store FST-specific args before calling super
        self.use_fst = getattr(args, "use_fst", True)
        self.fst_feature_channels = self._parse_feature_channels(
            getattr(args, "fst_feature_channels", "64,128,256,512,1024")
        )
        self.fst_num_queries = getattr(args, "fst_num_queries", 256)
        self.fst_query_dim = getattr(args, "fst_query_dim", 128)
        self.fst_num_scales = getattr(args, "fst_num_scales", 5)
        self.freeze_original_encoders = getattr(args, "freeze_original_encoders", True)

        # Consistency loss parameters
        self.use_consistency_loss = getattr(args, "use_consistency_loss", True)
        self.consistency_weight = getattr(args, "consistency_weight", 0.1)
        self.consistency_loss_type = getattr(args, "consistency_loss_type", "mse")
        self.consistency_num_pairs = getattr(args, "consistency_num_pairs", 2)

        # Call parent init
        super().__init__(args)

        # Setup consistency loss after model is created
        if self.use_fst and self.use_consistency_loss:
            self.combined_loss = CombinedFSTLoss(
                consistency_weight=self.consistency_weight,
                consistency_loss_type=self.consistency_loss_type,
                use_query_only=True,
                num_queries=self.fst_num_queries,
            )
            logger.info(
                f"Consistency loss enabled (weight={self.consistency_weight}, type={self.consistency_loss_type})"
            )

    def _parse_feature_channels(self, channels_str: str) -> list[int]:
        """Parse feature channels from comma-separated string."""
        if isinstance(channels_str, str):
            return [int(x.strip()) for x in channels_str.split(",")]
        return channels_str

    def _create_config(self, args) -> TrainingConfig:
        """Create TrainingConfig with FST-specific parameters."""
        config = super()._create_config(args)

        # Add FST-specific config (if needed for logging)
        config.use_fst = self.use_fst
        config.freeze_original_encoders = self.freeze_original_encoders
        config.style_source_same_prob = self.style_source_same_prob

        return config

    def _setup_models(self):
        """Initialize FST model components."""
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

        # Create base FontDiffuser model
        base_model = FontDiffuserModel(
            unet=unet,
            style_encoder=style_encoder,
            content_encoder=content_encoder,
        )

        # Wrap with FSTDiff enhancement
        if self.use_fst:
            logger.info("Creating FontDiffuserWithFST model...")
            self.model = FontDiffuserWithFST(
                original_fontdiffuser=base_model,
                feature_channels=self.fst_feature_channels,
                num_queries=self.fst_num_queries,
                query_dim=self.fst_query_dim,
                num_scales=self.fst_num_scales,
            )

            # Log model architecture and parameters
            self.model.log_model_info()
        else:
            self.model = base_model
            # Log base model parameters
            self.model.log_model_info()

        # Apply freezing if specified
        if self.use_fst and self.freeze_original_encoders:
            logger.info("Freezing original encoders...")
            for param in self.model.content_encoder.parameters():
                param.requires_grad = False
            for param in self.model.style_encoder.parameters():
                param.requires_grad = False
            for param in self.model.diffusion_unet.parameters():
                param.requires_grad = False
            logger.info("✓ Original encoders frozen")

            # Log updated trainable parameters after freezing
            logger.info("\nAfter freezing original encoders:")
            self.model.log_model_info()

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
        """Load phase 1 checkpoints with FST module support."""
        logger.info(f"Loading Phase 1 checkpoints from {ckpt_dir}...")

        # Try to load FST-enhanced checkpoint first
        fst_ckpt_path = Path(ckpt_dir) / "total_model_fst.pth"
        if fst_ckpt_path.exists() and self.use_fst:
            try:
                logger.info("Loading full FST model checkpoint")
                checkpoint = torch.load(fst_ckpt_path, map_location="cpu")
                # Will load into model after it's created
                self._fst_checkpoint = checkpoint
                return
            except Exception as e:
                logger.warning(f"Failed to load FST checkpoint: {e}")

        # Load individual components
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
                logger.info(f"✓ Loaded {name} from {ckpt_path}")

            except Exception as e:
                logger.error(f"Failed to load {name} from {ckpt_dir}: {e}")
                logger.debug(traceback.format_exc())

        # Try to load FST-specific modules if available
        if self.use_fst:
            fst_modules = ["mss_encoder", "fst_module", "fst_projection"]
            self._fst_module_states = {}

            for module_name in fst_modules:
                try:
                    ckpt_path = find_checkpoint(ckpt_dir, module_name)
                    if ckpt_path.exists():
                        state_dict = load_model_checkpoint(ckpt_path)
                        self._fst_module_states[module_name] = state_dict
                        logger.info(f"✓ Found {module_name} checkpoint")
                except Exception as e:
                    logger.debug(f"No checkpoint for {module_name}: {e}")

    def _setup_data(self):
        """Setup FST-compatible data transforms and dataloaders."""
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

        # Use FST-compatible dataset
        train_dataset = FontDatasetFST(
            args=self.args,
            phase="train",
            transforms=[content_transforms, style_transforms, target_transforms],
            scr=self.config.phase_2,
            use_fst=self.use_fst,
            style_source_same_prob=self.style_source_same_prob,
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.config.train_batch_size,
            collate_fn=CollateFNFST(verbose=True),
            num_workers=getattr(self.args, "num_workers", 4),
            pin_memory=True,
            persistent_workers=True,
        )

        logger.info(f"✓ Loaded FST dataset with {len(train_dataset)} samples")

    def _setup_optimizer(self):
        """Setup optimizer with selective parameter training for FST."""
        # Scale learning rate if requested
        learning_rate = self.config.learning_rate
        if self.args.scale_lr:
            learning_rate *= (
                self.config.gradient_accumulation_steps
                * self.config.train_batch_size
                * self.accelerator.num_processes
            )

        # Select trainable parameters
        if self.use_fst and self.freeze_original_encoders:
            # Only optimize new FST components
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            logger.info(
                f"Training {len(trainable_params)} parameter groups "
                f"({sum(p.numel() for p in trainable_params):,} parameters, FST only)"
            )
        else:
            trainable_params = self.model.parameters()

        self.optimizer = torch.optim.AdamW(
            trainable_params,
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
        """Wrap components with accelerator and load FST checkpoints if available."""
        # First wrap with accelerator
        super()._wrap_components()

        # Then load FST-specific checkpoints if they were found
        if hasattr(self, "_fst_checkpoint") and self.use_fst:
            try:
                unwrapped = self.accelerator.unwrap_model(self.model)
                unwrapped.load_state_dict(self._fst_checkpoint)
                logger.info("✓ Loaded full FST model state")
                del self._fst_checkpoint
            except Exception as e:
                logger.error(f"Failed to load FST checkpoint: {e}")

        elif hasattr(self, "_fst_module_states") and self.use_fst:
            try:
                unwrapped = self.accelerator.unwrap_model(self.model)
                for module_name, state_dict in self._fst_module_states.items():
                    if hasattr(unwrapped, module_name):
                        getattr(unwrapped, module_name).load_state_dict(state_dict)
                        logger.info(f"✓ Loaded {module_name} state")
                del self._fst_module_states
            except Exception as e:
                logger.error(f"Failed to load FST module states: {e}")

    def apply_classifier_free_guidance(
        self,
        content_images: torch.Tensor,
        style_images: torch.Tensor,
        drop_prob: float,
        samples: Optional[dict[str, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply classifier-free guidance including FST source style images."""
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

        # Mask source style images for FST
        if samples is not None and "style_source_image" in samples:
            style_source_images = samples["style_source_image"].clone()
            for i, mask_value in enumerate(context_mask):
                if mask_value == 1:
                    style_source_images[i, :, :, :] = 1.0
            samples["style_source_image"] = style_source_images

        return content_images, style_images

    def train_step(
        self,
        samples: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Single training step with FST-specific handling.

        For consistency loss, we need multiple reference characters with the same
        source→target style pair but different content.
        """
        self.model.train()

        # Extract data from batch
        target_img = samples["target_img"]
        style_img = samples["style_img"]
        content_img = samples["content_img"]

        # Additional reference images for consistency (if available)
        # These should have different content but same source/target styles
        ref_content_imgs = samples.get("ref_content_imgs", [])

        batch_size = target_img.shape[0]
        device = target_img.device

        # Sample random timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
        ).long()

        # Generate random noise
        noise = torch.randn_like(target_img)

        # Add noise to target image
        noisy_latents = self.noise_scheduler.add_noise(target_img, noise, timesteps)

        # Apply classifier-free guidance dropout if enabled
        if self.cfg_drop_prob > 0:
            content_img, style_img = self.apply_classifier_free_guidance(
                content_img, style_img, self.cfg_drop_prob
            )

        # Forward pass
        if self.use_fst:
            # For FST model, we need source and target style images
            # Assuming style_img is the target and we have a source style reference
            style_source_img = samples.get("style_source_img", style_img)

            outputs = self.model(
                noisy_latents=noisy_latents,
                timestep=timesteps,
                content_img=content_img,
                style_source_img=style_source_img,
                style_target_img=style_img,
                content_encoder_downsample_size=self.content_encoder_downsample_size,
                return_dict=True,
            )

            # Collect transformation features for consistency loss
            transformation_features_list = [outputs["transformation_features"]]

            # Process additional reference images if available
            if len(ref_content_imgs) > 0 and self.use_consistency_loss:
                num_refs = min(len(ref_content_imgs), self.consistency_num_pairs - 1)

                for ref_content in ref_content_imgs[:num_refs]:
                    # Use same style pair but different content
                    ref_outputs = self.model(
                        noisy_latents=noisy_latents,  # Same noise
                        timestep=timesteps,
                        content_img=ref_content,
                        style_source_img=style_source_img,
                        style_target_img=style_img,
                        content_encoder_downsample_size=self.content_encoder_downsample_size,
                        return_dict=True,
                    )
                    transformation_features_list.append(
                        ref_outputs["transformation_features"]
                    )

            # Compute loss with consistency
            if self.use_consistency_loss and len(transformation_features_list) >= 2:
                loss_dict = self.combined_loss(
                    noise_pred=outputs["noise_pred"],
                    target_noise=noise,
                    transformation_features_list=transformation_features_list,
                    offset_out_sum=outputs.get("offset_out_sum"),
                )
            else:
                # Fallback to basic loss
                loss_dict = self.model.get_loss_dict(outputs, noise)
        else:
            # Standard FontDiffuser forward
            noise_pred, offset_out_sum = self.model(
                noisy_latents,
                timesteps,
                style_img,
                content_img,
                self.content_encoder_downsample_size,
            )

            loss_dict = {
                "noise_loss": torch.nn.functional.mse_loss(noise_pred, noise),
                "offset_loss": (
                    offset_out_sum.mean()
                    if isinstance(offset_out_sum, torch.Tensor)
                    else torch.tensor(0.0, device=device)
                ),
            }
            loss_dict["total_loss"] = (
                loss_dict["noise_loss"] + 0.01 * loss_dict["offset_loss"]
            )

        loss = loss_dict["total_loss"]

        # Convert to float for logging
        loss_dict_float = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in loss_dict.items()
        }

        return loss, loss_dict_float

    def save_checkpoint(self, is_final: bool = False):
        """Save FST training checkpoint."""
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

        if self.use_fst:
            # Save FST-enhanced model components
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

            # Save FST-specific modules
            save_model_checkpoint(
                unwrapped_model.mss_encoder.state_dict(),
                save_dir / "mss_encoder.safetensors",
            )
            save_model_checkpoint(
                unwrapped_model.fst_module.state_dict(),
                save_dir / "fst_module.safetensors",
            )
            save_model_checkpoint(
                unwrapped_model.fst_projection.state_dict(),
                save_dir / "fst_projection.safetensors",
            )

            # Save full model
            if getattr(self.args, "save_full_model", True):
                torch.save(unwrapped_model, save_dir / "total_model_fst.pth")
        else:
            # Save original model
            save_model_checkpoint(
                unwrapped_model.config.unet.state_dict(),
                save_dir / "unet.safetensors",
            )
            save_model_checkpoint(
                unwrapped_model.config.style_encoder.state_dict(),
                save_dir / "style_encoder.safetensors",
            )
            save_model_checkpoint(
                unwrapped_model.config.content_encoder.state_dict(),
                save_dir / "content_encoder.safetensors",
            )

            if getattr(self.args, "save_full_model", False):
                torch.save(unwrapped_model, save_dir / "total_model.pth")

        # Save SCR for phase 2
        if self.config.phase_2 and self.scr is not None:
            save_model_checkpoint(self.scr.state_dict(), save_dir / "scr.safetensors")

        # Save optimizer and scheduler states
        torch.save(
            {
                "global_step": self.global_step,
                "epoch": self.current_epoch,
                "model_state_dict": unwrapped_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "config": asdict(self.config),
                "fst_config": {
                    "use_fst": self.use_fst,
                    "freeze_original_encoders": self.freeze_original_encoders,
                    "style_source_same_prob": self.style_source_same_prob,
                    "fst_feature_channels": self.fst_feature_channels,
                    "fst_num_queries": self.fst_num_queries,
                    "fst_query_dim": self.fst_query_dim,
                    "fst_num_scales": self.fst_num_scales,
                },
            },
            save_dir / "training_state.pt",
        )

        logger.info(f"✓ Saved FST checkpoint to {save_dir}")
        self.accelerator.log(
            {
                "checkpoint_saved": True,
                "checkpoint_step": self.global_step,
                "checkpoint_type": "fst" if self.use_fst else "base",
            }
        )

    def _setup_logging(self):
        """Setup logging and tracking with FST information."""
        super()._setup_logging()

        # Log FST-specific configuration
        if self.accelerator.is_main_process:
            fst_config = {
                "fst_enabled": self.use_fst,
                "freeze_original_encoders": self.freeze_original_encoders,
                "style_source_same_prob": self.style_source_same_prob,
                "fst_feature_channels": self.fst_feature_channels,
                "fst_num_queries": self.fst_num_queries,
                "fst_query_dim": self.fst_query_dim,
                "fst_num_scales": self.fst_num_scales,
            }

            if self.use_fst:
                unwrapped = self.accelerator.unwrap_model(self.model)
                fst_config["model_info"] = {
                    "mss_encoder_params": sum(
                        p.numel() for p in unwrapped.mss_encoder.parameters()
                    ),
                    "fst_module_params": sum(
                        p.numel() for p in unwrapped.fst_module.parameters()
                    ),
                    "fst_projection_params": sum(
                        p.numel() for p in unwrapped.fst_projection.parameters()
                    ),
                    "total_fst_params": sum(
                        p.numel()
                        for p in list(unwrapped.mss_encoder.parameters())
                        + list(unwrapped.fst_module.parameters())
                        + list(unwrapped.fst_projection.parameters())
                    ),
                }

            self.accelerator.log({"fst_config": fst_config})
            logger.info(f"FST Configuration: {fst_config}")
