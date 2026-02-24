"""
Trainer class for FontDiffuserWithFST.
Extends base FontDiffuserTrainer with FST-specific functionality.
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
from src import (
    ContentPerceptualLoss, FontDiffuserModel, build_content_encoder, build_ddpm_scheduler,
    build_scr, build_style_encoder, build_unet, build_fst, build_mss_encoder,
    build_fst_projection, build_original_style_projection, get_unet_cross_attention_dim,
    build_identity_loss_module, build_skeleton_transform, build_dual_channel_content_encoder,
    build_frequency_decomposition,
)
from src.model import FontDiffuserWithFST
from src.tools.utilities import find_checkpoint, HFTqdm, load_model_checkpoint, save_model_checkpoint
from src.tools.utils import normalize_mean_std, reNormalize_img, save_args_to_yaml, x0_from_epsilon
from src.trainers.training_config import TrainingConfig
from src.trainers.trainer import FontDiffuserTrainer
from src.modules.skeleton_distance_transform import SkeletonDistanceTransform

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

class FontDiffuserFSTTrainer(FontDiffuserTrainer):
    """Trainer for FontDiffuserWithFST model with skeleton transform support."""

    def __init__(self, args: argparse.Namespace):
        # FST Args
        self.use_fst = getattr(args, "use_fst", True)
        self.style_source_same_prob = getattr(args, "style_source_same_prob", 0.5)
        self.freeze_modules = self._parse_freeze_modules(getattr(args, "freeze_modules", ""))
        self.fst_feature_channels = self._parse_feature_channels(getattr(args, "fst_feature_channels", "64,128,256,512,1024"))
        
        # Config args
        for k, v in {
            "fst_num_queries": 256, "fst_query_dim": 128, "fst_num_scales": 5,
            "num_consistency_pairs": 0, "consistency_loss_weight": 0.1,
            "num_identity_pairs": 0, "identity_loss_weight": 0.1, "identity_pair_mode": "random",
            "use_skeleton_content": False, "skeleton_method": "medial_axis",
            "skeleton_distance_method": "hybrid", "skeleton_max_distance": 10.0,
            "skeleton_sigma": 3.0, "skeleton_output_mode": "dual_channel",
            "skeleton_fusion_method": "concat", "use_frequency_decomp": False,
            "frequency_low_cutoff": 0.10, "frequency_mid_cutoff": 0.40,
            "frequency_filter_type": "gaussian", "frequency_normalize_bands": True,
            "frequency_use_mid_band": True, "frequency_mid_target": "both"
        }.items():
            setattr(self, k, getattr(args, k, v))
        
        super().__init__(args)

    def _parse_freeze_modules(self, modules_str: str) -> list[str]:
        if not modules_str or not modules_str.strip(): return []
        valid = {"unet", "style_encoder", "content_encoder", "mss_encoder", "fst_module", "fst_projection", "original_style_projection"}
        modules = [m.strip().lower() for m in modules_str.split(",") if m.strip()]
        invalid = set(modules) - valid
        if invalid: logger.warning(f"Invalid modules in --freeze_modules: {invalid}. Valid: {valid}")
        return [m for m in modules if m in valid]

    def _parse_feature_channels(self, channels_str: str) -> list[int]:
        return [int(x.strip()) for x in channels_str.split(",")] if isinstance(channels_str, str) else channels_str

    def _setup_models(self):
        logger.info("Building model components...")
        unet, style_encoder, content_encoder = build_unet(self.args), build_style_encoder(self.args), build_content_encoder(self.args)
        self.noise_scheduler = build_ddpm_scheduler(self.args)

        if self.args.phase_1_ckpt_dir:
            self._load_phase1_checkpoints(unet, style_encoder, content_encoder, self.args.phase_1_ckpt_dir)
        else:
            logger.warning("[WARNING] No phase_1_ckpt_dir specified - training from scratch!")

        if self.use_fst:
            logger.info("Building FST-enhanced model...")
            mss_encoder = build_mss_encoder(args=self.args)
            fst_module = build_fst(args=self.args)
            cross_attn_dim = get_unet_cross_attention_dim(unet)
            fst_projection = build_fst_projection(self.fst_feature_channels[-1], cross_attn_dim)
            original_style_projection = build_original_style_projection(1024, cross_attn_dim)
            
            skeleton_transform = build_skeleton_transform(self.args) if self.use_skeleton_content else None
            if self.use_skeleton_content: content_encoder = build_dual_channel_content_encoder(self.args)
            frequency_decomp = build_frequency_decomposition(self.args) if self.use_frequency_decomp else None
            
            self.model = FontDiffuserWithFST(unet, style_encoder, content_encoder, mss_encoder, fst_module, 
                                             fst_projection, original_style_projection, skeleton_transform, frequency_decomp)
            
            if self.use_skeleton_content: logger.info(f"Skeleton Config: method={self.skeleton_method}, dist={self.skeleton_distance_method}")
            if self.use_frequency_decomp: logger.info(f"Frequency Config: low={self.frequency_low_cutoff}, mid={self.frequency_mid_cutoff}")
            
            self.model.log_model_info()
            self.identity_loss_module = build_identity_loss_module(self.args) if self.num_identity_pairs > 0 else None
            if self.identity_loss_module: logger.info(f"✓ IdentityMappingLoss created")
        else:
            self.model = FontDiffuserModel(unet, style_encoder, content_encoder)
            self.identity_loss_module = None
            self.model.log_model_info()

        self.perceptual_loss = ContentPerceptualLoss()
        self.scr = None
        if self.config.phase_2:
            self.scr = build_scr(args=self.args)
            if getattr(self.args, "scr_ckpt_path", None): self._load_scr_checkpoint(self.args.scr_ckpt_path)
            self.scr.requires_grad_(False)
        
        if self.freeze_modules: self._apply_module_freezing()

    def _setup_data(self):
        content_tf = transforms.Compose([transforms.Resize(self.args.content_image_size, interpolation=transforms.InterpolationMode.BILINEAR), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        style_tf = transforms.Compose([transforms.Resize(self.args.style_image_size, interpolation=transforms.InterpolationMode.BILINEAR), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        target_tf = transforms.Compose([transforms.Resize((self.args.resolution, self.args.resolution), interpolation=transforms.InterpolationMode.BILINEAR), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        
        sk_cfg = {
            "method": self.skeleton_method, "distance_method": self.skeleton_distance_method,
            "max_distance": self.skeleton_max_distance, "sigma": self.skeleton_sigma,
            "output_mode": self.skeleton_output_mode, "normalize": True,
        } if self.use_skeleton_content else None

        train_ds = FontDatasetFST(self.args, "train", [content_tf, style_tf, target_tf], self.config.phase_2, 
                                  self.use_fst, self.style_source_same_prob, self.num_consistency_pairs,
                                  self.num_identity_pairs, self.identity_pair_mode, self.use_skeleton_content, sk_cfg)
        
        self.train_dataloader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=self.config.train_batch_size, 
                                                            collate_fn=CollateFNFST(), num_workers=self.args.num_workers, pin_memory=True, persistent_workers=True)
        logger.info(f"✓ Loaded FST dataset ({len(train_ds)} samples)")

    def _setup_logging(self):
        super()._setup_logging()
        if not self.accelerator.is_main_process: return
        
        cfg = {k: getattr(self, k) for k in ["use_fst", "freeze_modules", "style_source_same_prob", "fst_feature_channels", 
               "fst_num_queries", "fst_query_dim", "fst_num_scales", "num_consistency_pairs", "consistency_loss_weight", 
               "num_identity_pairs", "identity_loss_weight", "identity_pair_mode"]}
        
        if self.use_skeleton_content:
            cfg.update({k: getattr(self, k) for k in ["skeleton_method", "skeleton_distance_method", "skeleton_max_distance", "skeleton_sigma"]})
        if self.use_frequency_decomp:
            cfg.update({k: getattr(self, k) for k in ["frequency_low_cutoff", "frequency_mid_cutoff", "frequency_filter_type"]})

        if self.use_fst:
            unwrapped = self.accelerator.unwrap_model(self.model)
            cfg["model_info"] = {f"{n}_params": sum(p.numel() for p in getattr(unwrapped, n).parameters()) for n in ["mss_encoder", "fst_module", "fst_projection"]}
        
        self.accelerator.log({"fst_config": cfg})

    def _apply_module_freezing(self):
        if not self.freeze_modules: return
        logger.info("Applying module freezing...")
        
        if self.use_fst:
            module_map = {
                "unet": self.model.diffusion_unet, "style_encoder": self.model.style_encoder,
                "content_encoder": self.model.content_encoder, "mss_encoder": self.model.mss_encoder,
                "fst_module": self.model.fst_module, "fst_projection": self.model.fst_projection,
                "original_style_projection": self.model.original_style_projection,
            }
        else:
            module_map = {"unet": self.model.config.unet, "style_encoder": self.model.config.style_encoder, "content_encoder": self.model.config.content_encoder}

        for name in self.freeze_modules:
            if name not in module_map: continue
            module = module_map[name]
            module.requires_grad_(False)
            logger.info(f"✓ Frozen {name}")
        self.model.log_model_info()

    def _load_phase1_checkpoints(self, unet, style_encoder, content_encoder, ckpt_dir):
        logger.info(f"Loading Phase 1 from {ckpt_dir}...")
        for name, comp in [("unet", unet), ("style_encoder", style_encoder), ("content_encoder", content_encoder)]:
            try:
                path = find_checkpoint(ckpt_dir, name)
                if path.exists(): comp.load_state_dict(load_model_checkpoint(path)); logger.info(f"✓ Loaded {name}")
                else: logger.warning(f"[WARNING] {name} not found")
            except Exception as e: logger.error(f"Failed loading {name}: {e}")

        if self.use_fst:
            self._fst_module_states = {}
            for name in ["mss_encoder", "fst_module", "fst_projection", "original_style_projection"]:
                try:
                    path = find_checkpoint(ckpt_dir, name)
                    if path.exists(): self._fst_module_states[name] = load_model_checkpoint(path)
                except: pass

    def _setup_optimizer(self):
        lr = self.config.learning_rate * (self.config.gradient_accumulation_steps * self.config.train_batch_size * self.accelerator.num_processes if self.args.scale_lr else 1)
        params = [p for p in self.model.parameters() if p.requires_grad]
        logger.info(f"Optimizing {sum(p.numel() for p in params):,} params")
        self.optimizer = torch.optim.AdamW(params, lr=lr, betas=(self.config.adam_beta1, self.config.adam_beta2), weight_decay=self.config.adam_weight_decay, eps=self.config.adam_epsilon)
        self.lr_scheduler = get_scheduler(self.config.lr_scheduler, optimizer=self.optimizer, num_warmup_steps=self.config.lr_warmup_steps * self.config.gradient_accumulation_steps, num_training_steps=self.config.max_train_steps * self.config.gradient_accumulation_steps)

    def _wrap_components(self):
        super()._wrap_components()
        if self.use_fst and hasattr(self, "identity_loss_module") and self.identity_loss_module:
            self.identity_loss_module = self.accelerator.prepare(self.identity_loss_module)
        
        if hasattr(self, "_fst_module_states") and self.use_fst:
            unwrapped = self.accelerator.unwrap_model(self.model)
            for name, state in self._fst_module_states.items():
                if hasattr(unwrapped, name): getattr(unwrapped, name).load_state_dict(state)
            del self._fst_module_states

        if self.args.compile: self.model = torch.compile(self.model)

    def apply_classifier_free_guidance(self, content_images, style_images, drop_prob, samples=None):
        content_images, style_images = content_images.clone(), style_images.clone()
        bsz = content_images.shape[0]
        context_mask = torch.bernoulli(torch.zeros(bsz, device=content_images.device) + drop_prob)
        
        for i, mask in enumerate(context_mask):
            if mask == 1:
                content_images[i], style_images[i] = 1.0, 1.0
                if samples and "style_source_image" in samples: samples["style_source_image"][i] = 1.0
        return content_images, style_images

    def train_step(self, samples):
        self.model.train()
        content_images, style_images, target_images, nonorm_target = samples["content_image"], samples["style_image"], samples["target_image"], samples["nonorm_target_image"]
        noise = torch.randn_like(target_images)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (target_images.shape[0],), device=target_images.device).long()
        noisy_targets = self.noise_scheduler.add_noise(target_images, noise, timesteps)
        
        content_images, style_images = self.apply_classifier_free_guidance(content_images, style_images, self.config.drop_prob, samples)

        if self.use_fst:
            out = self.model(noisy_targets, timesteps, content_images, samples.get("style_source_image"), style_images, self.args.content_encoder_downsample_size)
            noise_pred, offset_out_sum = out["noise_pred"], out["offset_out_sum"]
        else:
            noise_pred, offset_out_sum = self.model(noisy_targets, timesteps, style_images, content_images, self.args.content_encoder_downsample_size)

        total_loss, loss_dict, pred_orig_norm = self.compute_losses(noise_pred, noise, offset_out_sum, noisy_targets, nonorm_target, timesteps)

        if self.config.phase_2 and self.scr and samples.get("neg_images") is not None:
            sc_loss = self.compute_phase2_loss(pred_orig_norm, target_images, samples["neg_images"])
            total_loss += self.config.sc_coefficient * sc_loss
            loss_dict["sc_loss"] = sc_loss.item()

        if self.use_fst and self.num_consistency_pairs > 0 and samples.get("consistency_source_images") is not None:
            if samples["consistency_source_images"].numel() > 0:
                model = self.accelerator.unwrap_model(self.model)
                c_loss = model.compute_consistency_loss(samples["consistency_source_images"], samples["consistency_target_images"])
                total_loss += self.consistency_loss_weight * c_loss
                loss_dict["consistency_loss"] = c_loss.item()

        if self.use_fst and self.num_identity_pairs > 0 and samples.get("num_identity_pairs_total", 0) > 0:
            model = self.accelerator.unwrap_model(self.model)
            id_loss, id_metrics = model.compute_identity_loss(samples["identity_pair_sources"], samples["identity_pair_targets"], self.fst_num_queries)
            total_loss += self.identity_loss_weight * id_loss
            loss_dict.update({"identity_loss": id_loss.item(), **{f"identity_{k}": v for k, v in id_metrics.items()}})
        
        return total_loss, loss_dict

    def train(self):
        num_update_steps = math.ceil(len(self.train_dataloader) / self.config.gradient_accumulation_steps)
        num_epochs = math.ceil(self.config.max_train_steps / num_update_steps)
        if getattr(self.args, "resume_from_checkpoint", None): self.load_checkpoint(self.args.resume_from_checkpoint)
        
        progress_bar = HFTqdm(range(self.config.max_train_steps), disable=not self.accelerator.is_local_main_process)
        loss_accum, count_accum = 0.0, 0

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            for step, samples in enumerate(self.train_dataloader):
                if self.global_step >= self.config.max_train_steps: break
                with self.accelerator.accumulate(self.model):
                    loss, loss_dict = self.train_step(samples)
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step(); self.lr_scheduler.step(); self.optimizer.zero_grad(set_to_none=True)
                    
                    loss_accum += loss.item(); count_accum += 1
                    if self.accelerator.sync_gradients:
                        progress_bar.update(1); self.global_step += 1
                        avg_loss = loss_accum / count_accum
                        logs = {"loss/avg_train_loss": avg_loss, "train/lr": self.lr_scheduler.get_last_lr()[0], "train/epoch": epoch + step/len(self.train_dataloader), "train/grad_norm": grad_norm.item() if self.accelerator.sync_gradients else 0.0, **{f"loss/{k}": v for k, v in loss_dict.items()}}
                        self.accelerator.log(logs, step=self.global_step)
                        loss_accum, count_accum = 0.0, 0
                        
                        if self.global_step % self.args.ckpt_interval == 0:
                            self.accelerator.wait_for_everyone()
                            if self.accelerator.is_main_process:
                                self.save_checkpoint()
                progress_bar.set_postfix(loss=loss.item(), lr=self.lr_scheduler.get_last_lr()[0], step=self.global_step)
        
        progress_bar.close()
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process: self.save_checkpoint(is_final=True)
        self.accelerator.end_training()

    def save_checkpoint(self, is_final=False):
        unwrapped = self.accelerator.unwrap_model(self.model)
        if not self.accelerator.is_main_process: return
        
        save_dir = Path(self.args.output_dir) / ("final" if is_final else f"checkpoint_step_{self.global_step}")
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.use_fst:
            components = {
                "unet": unwrapped.diffusion_unet, "style_encoder": unwrapped.style_encoder,
                "content_encoder": unwrapped.content_encoder, "mss_encoder": unwrapped.mss_encoder,
                "fst_module": unwrapped.fst_module, "fst_projection": unwrapped.fst_projection,
                "original_style_projection": unwrapped.original_style_projection,
            }
            for name, mod in components.items(): save_model_checkpoint(mod.state_dict(), save_dir / f"{name}.safetensors")
            if self.identity_loss_module: save_model_checkpoint(self.identity_loss_module.state_dict(), save_dir / "identity_loss_module.safetensors")
        else:
            save_model_checkpoint(unwrapped.state_dict(), save_dir / "model.safetensors")

        if self.config.phase_2 and self.scr: save_model_checkpoint(self.scr.state_dict(), save_dir / "scr.safetensors")

        state = {
            "global_step": self.global_step, "epoch": self.current_epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "config": asdict(self.config),
            "fst_config": {k: getattr(self, k) for k in ["use_fst", "style_source_same_prob", "fst_num_queries", "fst_query_dim", "num_consistency_pairs", "num_identity_pairs"]},
            **({"skeleton_config": {k: getattr(self, k) for k in ["use_skeleton_content", "skeleton_method"]}} if self.use_skeleton_content else {}),
            **({"frequency_config": {k: getattr(self, k) for k in ["use_frequency_decomp", "frequency_low_cutoff"]}} if self.use_frequency_decomp else {}),
        }
        torch.save(state, save_dir / "training_state.pt")
        logger.info(f"✓ Saved checkpoint to {save_dir}")

    def load_checkpoint(self, path):
        if not Path(path).exists(): logger.warning(f"Checkpoint not found {path}"); return False
        try:
            ckpt_dir = Path(path)
            state_file = next((ckpt_dir / f for f in ["training_state.pt", "training_state.pth"] if (ckpt_dir / f).exists()), None)
            if state_file:
                state = torch.load(state_file, map_location="cpu")
                self.global_step, self.current_epoch = state.get("global_step", 0), state.get("epoch", 0)
                if "optimizer_state_dict" in state: self.optimizer.load_state_dict(state["optimizer_state_dict"])
                if "lr_scheduler_state_dict" in state: self.lr_scheduler.load_state_dict(state["lr_scheduler_state_dict"])
            
            unwrapped = self.accelerator.unwrap_model(self.model)
            if self.use_fst:
                comps = {
                    "unet": unwrapped.diffusion_unet, "style_encoder": unwrapped.style_encoder,
                    "content_encoder": unwrapped.content_encoder, "mss_encoder": unwrapped.mss_encoder,
                    "fst_module": unwrapped.fst_module, "fst_projection": unwrapped.fst_projection,
                    "original_style_projection": unwrapped.original_style_projection,
                }
            else:
                comps = {"unet": unwrapped.config.unet, "style_encoder": unwrapped.config.style_encoder, "content_encoder": unwrapped.config.content_encoder}
            
            for name, mod in comps.items():
                p = ckpt_dir / f"{name}.safetensors"
                if p.exists(): mod.load_state_dict(load_model_checkpoint(p))
            
            if self.config.phase_2 and self.scr and (ckpt_dir / "scr.safetensors").exists():
                self.scr.load_state_dict(load_model_checkpoint(ckpt_dir / "scr.safetensors"))
            return True
        except Exception as e:
            logger.error(f"Failed loading checkpoint: {e}")
            return False

    def export_to_onnx(self):
        if not self.accelerator.is_main_process: return False
        try:
            export_dir = Path(getattr(self.args, "onnx_export_dir", None) or Path(self.args.output_dir) / "onnx")
            export_dir.mkdir(parents=True, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.eval()
            
            class ONNXWrapper(torch.nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, noisy_latents, ts, content, style_src, style_tgt):
                    d = self.m(noisy_latents, ts, content, style_src, style_tgt, 4, return_dict=True)
                    return (d["noise_pred"], d["offset_out_sum"], d["content_features"], d["transformation_features"],
                            d["fst_condition"], d["source_style_features"], d["target_style_features"], d["orig_style_feat"], d["orig_style_vec"])

            dummy_in = (
                torch.randn(1, 4, 12, 12, device=unwrapped.device), torch.tensor([0], device=unwrapped.device),
                torch.randn(1, 1, 96, 96, device=unwrapped.device), torch.randn(1, 1, 96, 96, device=unwrapped.device),
                torch.randn(1, 1, 96, 96, device=unwrapped.device)
            )
            onnx_path = export_dir / "fontdiffuser_fst_model.onnx"
            torch.onnx.export(ONNXWrapper(unwrapped), dummy_in, str(onnx_path), 
                              input_names=["noisy_latents", "timestep", "content_images", "style_source_images", "style_target_images"],
                              output_names=["noise_pred", "offset_out_sum", "content_features", "transformation_features", "fst_condition", "source_style_features", "target_style_features", "orig_style_feat", "orig_style_vec"],
                              opset_version=self.args.onnx_opset_version, dynamic_axes={n: {0: "batch"} for n in ["noisy_latents", "content_images", "style_source_images", "style_target_images", "noise_pred", "offset_out_sum"]})
            
            onnx.checker.check_model(onnx.load(str(onnx_path)))
            logger.info(f"✓ ONNX export complete: {onnx_path} ({onnx_path.stat().st_size / 1024**2:.2f} MB)")
            return True
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False