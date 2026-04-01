"""
Trainer class for FontDiffuserWithMRL.
Extends FontDiffuserFSTTrainer with Matryoshka Representation Learning (MRL) functionality.
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
from src.modules.mrl_fontdiffuser import (
    build_mrl_components,
    MatryoshkaContentEncoder,
    MRLLossModule,
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


class FontDiffuserMRLTrainer(FontDiffuserFSTTrainer):
    """Trainer for FontDiffuserWithFST + MRL model."""

    def __init__(self, args: argparse.Namespace):
        # MRL-specific args
        self.use_mrl = getattr(args, "use_mrl", True)
        self.mrl_nesting_dims = self._parse_mrl_nesting_dims(
            getattr(args, "mrl_nesting_dims", "64,128,256,512")
        )
        self.mrl_freq_radii = self._parse_mrl_freq_radii(
            getattr(args, "mrl_freq_radii", "0.1,0.3,0.5")
        )
        self.mrl_content_weight = getattr(args, "mrl_content_weight", 1.0)
        self.mrl_fourier_weight = getattr(args, "mrl_fourier_weight", 0.3)
        self.mrl_temperature = getattr(args, "mrl_temperature", 0.07)
        self.use_mrl_fourier_alignment = getattr(
            args, "use_mrl_fourier_alignment", True
        )
        self.mrl_warmup_steps = getattr(args, "mrl_warmup_steps", 500)
        self.mrl_rampdown_steps = getattr(args, "mrl_rampdown_steps", 1000)
        self.mrl_start_weight = getattr(args, "mrl_start_weight", 1.0)
        self.mrl_final_weight = getattr(args, "mrl_final_weight", 0.3)

        # Initialize parent (FST) trainer
        super().__init__(args)

        # Initialize MRL components (will be set in _setup_models)
        self.mrl_encoder: Optional[MatryoshkaContentEncoder] = None
        self.mrl_loss_module: Optional[MRLLossModule] = None

    def _parse_mrl_nesting_dims(self, dims_str: str) -> list[int]:
        """Parse MRL nesting dimensions from comma-separated string."""
        if isinstance(dims_str, list):
            return dims_str
        return [int(x.strip()) for x in dims_str.split(",") if x.strip()]

    def _parse_mrl_freq_radii(self, radii_str: str) -> list[float]:
        """Parse MRL frequency radii from comma-separated string."""
        if isinstance(radii_str, list):
            return radii_str
        return [float(x.strip()) for x in radii_str.split(",") if x.strip()]

    def _setup_models(self):
        """Setup models with MRL components."""
        logger.info("Building model components...")
        unet, style_encoder, content_encoder = (
            build_unet(self.args),
            build_style_encoder(self.args),
            build_content_encoder(self.args),
        )
        self.noise_scheduler = build_ddpm_scheduler(self.args)

        if self.args.phase_1_ckpt_dir:
            self._load_phase1_checkpoints(
                unet, style_encoder, content_encoder, self.args.phase_1_ckpt_dir
            )
        else:
            logger.warning(
                "[WARNING] No phase_1_ckpt_dir specified - training from scratch!"
            )

        if self.use_fst:
            logger.info("Building FST-enhanced model...")
            mss_encoder = build_mss_encoder(args=self.args)
            fst_module = build_fst(args=self.args)
            cross_attn_dim = get_unet_cross_attention_dim(unet)
            fst_projection = build_fst_projection(
                self.fst_feature_channels[-1], cross_attn_dim
            )
            original_style_projection = build_original_style_projection(
                1024, cross_attn_dim
            )

            skeleton_transform = (
                build_skeleton_transform(self.args)
                if self.use_skeleton_content
                else None
            )
            if self.use_skeleton_content:
                logger.info("✓ Skeleton transform enabled")

            frequency_decomp = (
                build_frequency_decomposition(self.args)
                if self.use_frequency_decomp
                else None
            )

            # Build base FST model
            self.model = FontDiffuserWithFST(
                unet,
                style_encoder,
                content_encoder,
                mss_encoder,
                fst_module,
                fst_projection,
                original_style_projection,
                skeleton_transform,
                frequency_decomp,
            )

            if self.use_skeleton_content:
                logger.info("✓ Skeleton encoder built")
            if self.use_frequency_decomp:
                logger.info("✓ Frequency decomposition built")

            # Build MRL components
            if self.use_mrl:
                logger.info("Building MRL components...")
                
                # Dynamically determine the maximum actual embedding dimension based on resolution architecture
                actual_embedding_dim = self.model.content_encoder.arch["out_channels"][-1] if hasattr(self.model.content_encoder, "arch") else getattr(self.args, "content_encoder_dim", 512)

                # Filter out nesting dims that are larger than the actual model embedding size
                valid_nesting_dims = [d for d in self.mrl_nesting_dims if d <= actual_embedding_dim]
                
                # Adjust freq_radii to match the new nesting_dims length
                if len(valid_nesting_dims) < len(self.mrl_nesting_dims):
                    logger.warning(f"Truncated MRL nesting dims to {valid_nesting_dims} to fit model's embedding size ({actual_embedding_dim})")
                    valid_freq_radii = list(self.mrl_freq_radii)[:len(valid_nesting_dims) - 1]
                else:
                    valid_freq_radii = tuple(self.mrl_freq_radii)

                # Validate MRL dimensions
                if len(valid_freq_radii) != len(valid_nesting_dims) - 1:
                    raise ValueError(
                        f"MRL dimension mismatch! "
                        f"nesting_dims={valid_nesting_dims} (len={len(valid_nesting_dims)}), "
                        f"freq_radii={valid_freq_radii} (len={len(valid_freq_radii)}). "
                        f"Required: len(freq_radii) == len(nesting_dims) - 1"
                    )
                
                logger.info(
                    f"MRL Config: nesting_dims={valid_nesting_dims}, "
                    f"freq_radii={valid_freq_radii}"
                )
                
                self.mrl_encoder, self.mrl_loss_module = build_mrl_components(
                    content_encoder=self.model.content_encoder,
                    embedding_dim=actual_embedding_dim,
                    nesting_dims=valid_nesting_dims,
                    freq_radii=valid_freq_radii,
                    spatial_size=(64, 64),
                    use_fourier_alignment=self.use_mrl_fourier_alignment,
                )
                # Replace content encoder in model with MRL-wrapped version
                self.model.content_encoder = self.mrl_encoder
                logger.info("✓ MRL components built and integrated")

            self.model.log_model_info()
            self.identity_loss_module = (
                build_identity_loss_module(self.args)
                if self.num_identity_pairs > 0
                else None
            )
            if self.identity_loss_module:
                logger.info("✓ Identity loss module built")
        else:
            self.model = FontDiffuserModel(unet, style_encoder, content_encoder)
            self.identity_loss_module = None
            self.mrl_encoder = None
            self.mrl_loss_module = None
            self.model.log_model_info()

        self.perceptual_loss = ContentPerceptualLoss()
        self.scr = None
        if self.config.phase_2:
            self.scr = build_scr(args=self.args)
            if getattr(self.args, "scr_ckpt_path", None):
                self._load_scr_checkpoint(self.args.scr_ckpt_path)
            self.scr.requires_grad_(False)

        if self.freeze_modules:
            self._apply_module_freezing()

    def _setup_data(self):
        """Setup data loaders (inherited from FST trainer)."""
        content_tf = transforms.Compose(
            [
                transforms.Resize(
                    self.args.content_image_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        style_tf = transforms.Compose(
            [
                transforms.Resize(
                    self.args.style_image_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        target_tf = transforms.Compose(
            [
                transforms.Resize(
                    (self.args.resolution, self.args.resolution),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        sk_cfg = (
            {
                "method": self.skeleton_method,
                "distance_method": self.skeleton_distance_method,
                "max_distance": self.skeleton_max_distance,
                "sigma": self.skeleton_sigma,
                "output_mode": self.skeleton_output_mode,
                "normalize": True,
            }
            if self.use_skeleton_content
            else None
        )

        train_ds = FontDatasetFST(
            self.args,
            "train",
            [content_tf, style_tf, target_tf],
            self.config.phase_2,
            self.use_fst,
            self.style_source_same_prob,
            self.num_consistency_pairs,
            self.num_identity_pairs,
            self.identity_pair_mode,
            self.use_skeleton_content,
            sk_cfg,
        )

        self.train_dataloader = torch.utils.data.DataLoader(
            train_ds,
            shuffle=True,
            batch_size=self.config.train_batch_size,
            collate_fn=CollateFNFST(),
            num_workers=self.args.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        logger.info(f"✓ Loaded FST+MRL dataset ({len(train_ds)} samples)")

    def _setup_logging(self):
        """Setup logging with MRL configuration."""
        super()._setup_logging()
        if not self.accelerator.is_main_process:
            return

        mrl_cfg = {
            k: getattr(self, k)
            for k in [
                "use_mrl",
                "mrl_nesting_dims",
                "mrl_freq_radii",
                "mrl_content_weight",
                "mrl_fourier_weight",
                "mrl_temperature",
                "use_mrl_fourier_alignment",
                "mrl_warmup_steps",
                "mrl_rampdown_steps",
                "mrl_start_weight",
                "mrl_final_weight",
            ]
        }

        if self.use_mrl and self.mrl_loss_module:
            mrl_cfg["mrl_loss_params"] = {
                "granularity_weights": getattr(
                    self.mrl_loss_module.content_loss, "granularity_weights", []
                ),
                "temperature": self.mrl_temperature,
            }

        self.accelerator.log({"mrl_config": mrl_cfg})

    def _get_mrl_loss_weight(self, step: int) -> float:
        """Compute MRL loss weight as a function of training step.

        Three phases:
          Phase 1 (0 .. warmup):           1.0 (MRL only)
          Phase 2 (warmup .. warmup+rampdown): Linear anneal from start to final
          Phase 3 (warmup+rampdown .. end): final_weight (mixed with aux losses)
        """
        if step < self.mrl_warmup_steps:
            return self.mrl_start_weight

        steps_into_rampdown = step - self.mrl_warmup_steps
        if steps_into_rampdown < self.mrl_rampdown_steps:
            # Linear interpolation
            alpha = steps_into_rampdown / self.mrl_rampdown_steps
            return (
                self.mrl_start_weight * (1.0 - alpha)
                + self.mrl_final_weight * alpha
            )

        return self.mrl_final_weight

    def train_step(self, samples):
        """Training step with MRL loss."""
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

        # MRL loss computation
        if self.use_mrl and self.mrl_loss_module:
            try:
                # Get predicted x0 for MRL loss
                pred_x0 = x0_from_epsilon(
                    scheduler=self.noise_scheduler,
                    noise_pred=noise_pred,
                    x_t=noisy_targets,
                    timesteps=timesteps,
                )
                pred_x0_01 = normalize_mean_std(pred_x0)

                # Get content encoding with MRL projections
                if self.use_mrl:
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    _, pred_proj = unwrapped.content_encoder.forward_mrl(pred_x0_01)

                    with torch.no_grad():
                        _, content_proj = unwrapped.content_encoder.forward_mrl(
                            normalize_mean_std(content_images)
                        )

                    # Compute MRL loss
                    mrl_loss, mrl_metrics = self.mrl_loss_module(
                        pred_projections=pred_proj,
                        content_projections=content_proj,
                        content_images=normalize_mean_std(content_images),
                    )

                    # Apply dynamic weight scheduling
                    mrl_weight = self._get_mrl_loss_weight(self.global_step)
                    weighted_mrl_loss = mrl_weight * mrl_loss

                    total_loss += weighted_mrl_loss
                    loss_dict["mrl_loss"] = mrl_loss.item()
                    loss_dict["mrl_loss_weighted"] = weighted_mrl_loss.item()
                    loss_dict["mrl_weight"] = mrl_weight
                    loss_dict.update(
                        {f"mrl_{k}": v for k, v in mrl_metrics.items()}
                    )

            except Exception as e:
                logger.error(f"MRL loss computation failed: {e}", exc_info=True)
                # Continue training without MRL loss in this step
                pass

        return total_loss, loss_dict

    def save_checkpoint(self, is_final=False):
        """Save checkpoint with MRL components."""
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
                "mss_encoder": unwrapped.mss_encoder,
                "fst_module": unwrapped.fst_module,
                "fst_projection": unwrapped.fst_projection,
                "original_style_projection": unwrapped.original_style_projection,
            }

            # Save MRL-wrapped content encoder
            if self.use_mrl and self.mrl_encoder:
                components["mrl_encoder"] = self.mrl_encoder
            else:
                components["content_encoder"] = unwrapped.content_encoder

            for name, mod in components.items():
                save_model_checkpoint(
                    mod.state_dict(), save_dir / f"{name}.safetensors"
                )

            if self.identity_loss_module:
                save_model_checkpoint(
                    self.identity_loss_module.state_dict(),
                    save_dir / "identity_loss_module.safetensors",
                )

            # Save MRL loss module
            if self.use_mrl and self.mrl_loss_module:
                save_model_checkpoint(
                    self.mrl_loss_module.state_dict(),
                    save_dir / "mrl_loss_module.safetensors",
                )
        else:
            save_model_checkpoint(
                unwrapped.state_dict(), save_dir / "model.safetensors"
            )

        if self.config.phase_2 and self.scr:
            save_model_checkpoint(self.scr.state_dict(), save_dir / "scr.safetensors")

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
            "mrl_config": (
                {
                    k: getattr(self, k)
                    for k in [
                        "use_mrl",
                        "mrl_nesting_dims",
                        "mrl_freq_radii",
                        "mrl_content_weight",
                        "mrl_fourier_weight",
                        "mrl_temperature",
                        "use_mrl_fourier_alignment",
                        "mrl_warmup_steps",
                        "mrl_rampdown_steps",
                        "mrl_start_weight",
                        "mrl_final_weight",
                    ]
                }
                if self.use_mrl
                else {}
            ),
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
        torch.save(state, save_dir / "training_state.pt")
        logger.info(f"✓ Saved checkpoint to {save_dir}")

    def load_checkpoint(self, path):
        """Load checkpoint with MRL components."""
        if not Path(path).exists():
            logger.warning(f"Checkpoint not found {path}")
            return False
        try:
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
                state = torch.load(state_file, map_location="cpu")
                self.global_step, self.current_epoch = (
                    state.get("global_step", 0),
                    state.get("epoch", 0),
                )
                if "optimizer_state_dict" in state:
                    self.optimizer.load_state_dict(state["optimizer_state_dict"])
                if "lr_scheduler_state_dict" in state:
                    self.lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])

            unwrapped = self.accelerator.unwrap_model(self.model)
            if self.use_fst:
                comps = {
                    "unet": unwrapped.diffusion_unet,
                    "style_encoder": unwrapped.style_encoder,
                    "mss_encoder": unwrapped.mss_encoder,
                    "fst_module": unwrapped.fst_module,
                    "fst_projection": unwrapped.fst_projection,
                    "original_style_projection": unwrapped.original_style_projection,
                }

                # Load MRL encoder or content encoder
                if self.use_mrl and self.mrl_encoder:
                    comps["mrl_encoder"] = self.mrl_encoder
                else:
                    comps["content_encoder"] = unwrapped.content_encoder

                for name, mod in comps.items():
                    p = ckpt_dir / f"{name}.safetensors"
                    if p.exists():
                        mod.load_state_dict(load_model_checkpoint(p))

                # Load MRL loss module
                if (
                    self.use_mrl
                    and self.mrl_loss_module
                    and (ckpt_dir / "mrl_loss_module.safetensors").exists()
                ):
                    self.mrl_loss_module.load_state_dict(
                        load_model_checkpoint(ckpt_dir / "mrl_loss_module.safetensors")
                    )
            else:
                comps = {
                    "unet": unwrapped.config.unet,
                    "style_encoder": unwrapped.config.style_encoder,
                    "content_encoder": unwrapped.config.content_encoder,
                }
                for name, mod in comps.items():
                    p = ckpt_dir / f"{name}.safetensors"
                    if p.exists():
                        mod.load_state_dict(load_model_checkpoint(p))

            if (
                self.config.phase_2
                and self.scr
                and (ckpt_dir / "scr.safetensors").exists()
            ):
                self.scr.load_state_dict(
                    load_model_checkpoint(ckpt_dir / "scr.safetensors")
                )
            return True
        except Exception as e:
            logger.error(f"Failed loading checkpoint: {e}")
            return False

    def export_to_onnx(self):
        """Export model to ONNX format."""
        if not self.accelerator.is_main_process:
            return False
        try:
            export_dir = Path(
                getattr(self.args, "onnx_export_dir", None)
                or Path(self.args.output_dir) / "onnx"
            )
            export_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.eval()

            class ONNXWrapper(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, noisy_latents, ts, content, style_src, style_tgt):
                    d = self.m(
                        noisy_latents,
                        ts,
                        content,
                        style_src,
                        style_tgt,
                        4,
                        return_dict=True,
                    )
                    return (
                        d["noise_pred"],
                        d["offset_out_sum"],
                        d["content_features"],
                        d["transformation_features"],
                        d["fst_condition"],
                        d["source_style_features"],
                        d["target_style_features"],
                        d["orig_style_feat"],
                        d["orig_style_vec"],
                    )

            dummy_in = (
                torch.randn(1, 4, 12, 12, device=unwrapped.device),
                torch.tensor([0], device=unwrapped.device),
                torch.randn(1, 1, 96, 96, device=unwrapped.device),
                torch.randn(1, 1, 96, 96, device=unwrapped.device),
                torch.randn(1, 1, 96, 96, device=unwrapped.device),
            )
            onnx_path = export_dir / "fontdiffuser_mrl_model.onnx"
            torch.onnx.export(
                ONNXWrapper(unwrapped),
                dummy_in,
                str(onnx_path),
                input_names=[
                    "noisy_latents",
                    "timestep",
                    "content_images",
                    "style_source_images",
                    "style_target_images",
                ],
                output_names=[
                    "noise_pred",
                    "offset_out_sum",
                    "content_features",
                    "transformation_features",
                    "fst_condition",
                    "source_style_features",
                    "target_style_features",
                    "orig_style_feat",
                    "orig_style_vec",
                ],
                opset_version=self.args.onnx_opset_version,
                dynamic_axes={
                    n: {0: "batch"}
                    for n in [
                        "noisy_latents",
                        "content_images",
                        "style_source_images",
                        "style_target_images",
                        "noise_pred",
                        "offset_out_sum",
                    ]
                },
            )

            onnx.checker.check_model(onnx.load(str(onnx_path)))
            logger.info(
                f"✓ ONNX export complete: {onnx_path} ({onnx_path.stat().st_size / 1024**2:.2f} MB)"
            )
            return True
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False
