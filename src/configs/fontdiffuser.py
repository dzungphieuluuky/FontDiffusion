import os
import argparse

import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{__name__}.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def get_parser():
    """
    Centralized argument parser for FontDiffuser.
    Used by all training and inference scripts.
    """
    parser = argparse.ArgumentParser(
        description="FontDiffuser - Font Style Transfer with Diffusion Models"
    )

    # ==================== Experience & Paths ====================
    experience_group = parser.add_argument_group("Experiment Configuration")
    experience_group.add_argument(
        "--seed", type=int, default=123, help="Random seed for reproducibility"
    )
    experience_group.add_argument(
        "--experience_name",
        type=str,
        default="fontdiffuser_training",
        help="Experiment name for logging",
    )
    experience_group.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Font dataset root path (for training)",
    )
    experience_group.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints/results",
    )
    experience_group.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="Checkpoint directory (for loading pretrained models)",
    )
    experience_group.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="Logging library (wandb/tensorboard)",
    )
    experience_group.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="TensorBoard log directory",
    )

    # ==================== Model Architecture ====================
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument(
        "--resolution",
        type=int,
        default=96,
        help="Image resolution (all images resized to this)",
    )
    model_group.add_argument(
        "--unet_channels",
        type=tuple,
        default=(64, 128, 256, 512),
        help="U-Net channel dimensions",
    )
    model_group.add_argument(
        "--style_image_size", type=int, default=96, help="Style image size"
    )
    model_group.add_argument(
        "--content_image_size", type=int, default=96, help="Content image size"
    )
    model_group.add_argument(
        "--content_encoder_downsample_size",
        type=int,
        default=3,
        help="Content encoder downsample factor",
    )
    model_group.add_argument(
        "--channel_attn",
        action="store_true",
        default=True,
        help="Use SE attention in U-Net",
    )
    model_group.add_argument(
        "--content_start_channel",
        type=int,
        default=64,
        help="Content encoder first layer channels",
    )
    model_group.add_argument(
        "--style_start_channel",
        type=int,
        default=64,
        help="Style encoder first layer channels",
    )
    model_group.add_argument(
        "--freeze_modules",
        type=str,
        default="",
        help="Comma-separated list of modules to freeze during FST training. "
        "Options: unet, style_encoder, content_encoder, mss_encoder, fst_module, "
        "fst_projection, original_style_projection. "
        "Example: 'unet,style_encoder,content_encoder'",
    )

    # ==================== Training Configuration ====================
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument(
        "--phase_2", action="store_true", help="Train in phase 2 with SCR module"
    )
    training_group.add_argument(
        "--phase_1",
        action="store_true",
        help="Train in phase 1 without SCR module",
    )
    training_group.add_argument(
        "--phase_1_ckpt_dir",
        type=str,
        default=None,
        help="Phase 1 checkpoint directory (for phase 2 training)",
    )

    # SCR Module
    training_group.add_argument(
        "--temperature", type=float, default=0.07, help="SCR temperature parameter"
    )
    training_group.add_argument(
        "--mode", type=str, default="refinement", help="SCR mode"
    )
    training_group.add_argument(
        "--scr_image_size", type=int, default=96, help="SCR image size"
    )
    training_group.add_argument(
        "--scr_ckpt_path", type=str, default=None, help="SCR checkpoint path"
    )
    training_group.add_argument(
        "--num_neg", type=int, default=16, help="Number of negative samples for SCR"
    )
    training_group.add_argument(
        "--nce_layers", type=str, default="0,1,2,3", help="NCE layer indices"
    )
    training_group.add_argument(
        "--sc_coefficient",
        type=float,
        default=0.01,
        help="Style consistency loss coefficient",
    )

    # Batch & Steps
    training_group.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Training batch size per device",
    )
    training_group.add_argument(
        "--max_train_steps",
        type=int,
        default=440000,
        help="Total training steps",
    )
    training_group.add_argument(
        "--ckpt_interval", type=int, default=40000, help="Checkpoint save interval"
    )
    training_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    training_group.add_argument(
        "--log_interval", type=int, default=100, help="Training log interval"
    )
    training_group.add_argument(
        "--val_interval", type=int, default=100, help="Validation interval"
    )

    # Loss Coefficients
    training_group.add_argument(
        "--perceptual_coefficient",
        type=float,
        default=0.01,
        help="Perceptual loss coefficient",
    )
    training_group.add_argument(
        "--offset_coefficient", type=float, default=0.5, help="Offset loss coefficient"
    )
    training_group.add_argument(
        "--style_transform_coefficient",
        type=float,
        default=0.1,
        help="Style transformation loss coefficient",
    )

    # Learning Rate
    training_group.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    training_group.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale LR by GPUs/grad_accum/batch_size",
    )
    training_group.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear",
        help="LR scheduler type",
    )
    training_group.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10000,
        help="LR warmup steps",
    )

    # Classifier-Free Guidance
    training_group.add_argument(
        "--drop_prob",
        type=float,
        default=0.1,
        help="Unconditional training dropout probability",
    )

    # Scheduler
    training_group.add_argument(
        "--beta_scheduler",
        type=str,
        default="scaled_linear",
        help="Beta scheduler for DDPM",
    )

    # Optimizer
    training_group.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta1 parameter",
    )
    training_group.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta2 parameter",
    )
    training_group.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Adam weight decay"
    )
    training_group.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Adam epsilon value",
    )
    training_group.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm"
    )
    training_group.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training mode",
    )

    # ==================== Sampling/Inference ====================
    sampling_group = parser.add_argument_group("Sampling Configuration")
    sampling_group.add_argument(
        "--algorithm_type",
        type=str,
        default="dpmsolver++",
        help="Sampling algorithm",
    )
    sampling_group.add_argument(
        "--guidance_type",
        type=str,
        default="classifier-free",
        help="Guidance type",
    )
    sampling_group.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale",
    )
    sampling_group.add_argument(
        "--num_inference_steps", type=int, default=20, help="Number of inference steps"
    )
    sampling_group.add_argument(
        "--model_type", type=str, default="noise", help="Model type for sampling"
    )
    sampling_group.add_argument("--order", type=int, default=2, help="DPM-Solver order")
    sampling_group.add_argument(
        "--skip_type", type=str, default="time_uniform", help="DPM-Solver skip type"
    )
    sampling_group.add_argument(
        "--method", type=str, default="multistep", help="DPM-Solver method"
    )
    sampling_group.add_argument(
        "--correcting_x0_fn",
        type=str,
        default=None,
        help="DPM-Solver x0 correction function",
    )
    sampling_group.add_argument(
        "--t_start", type=str, default=None, help="DPM-Solver start time"
    )
    sampling_group.add_argument(
        "--t_end", type=str, default=None, help="DPM-Solver end time"
    )

    # ==================== Inference-Specific Arguments ====================
    inference_group = parser.add_argument_group("Inference-Specific")
    inference_group.add_argument("--demo", action="store_true", help="Run in demo mode")
    inference_group.add_argument(
        "--character_input",
        action="store_true",
        help="Use character input instead of image",
    )
    inference_group.add_argument(
        "--content_character",
        type=str,
        default=None,
        help="Single character, comma-separated list, or path to txt file",
    )
    inference_group.add_argument(
        "--characters_file",
        type=str,
        default=None,
        help="Path to text file with one character per line",
    )
    inference_group.add_argument(
        "--content_image_path", type=str, default=None, help="Path to content image"
    )
    inference_group.add_argument(
        "--style_image_path", type=str, default=None, help="Path to style image"
    )
    inference_group.add_argument(
        "--save_image", action="store_true", help="Save generated images"
    )
    inference_group.add_argument(
        "--save_image_dir", type=str, default=None, help="Image save directory"
    )
    inference_group.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use"
    )
    inference_group.add_argument(
        "--ttf_path",
        type=str,
        default="ttf/KaiXinSongA.ttf",
        help="Path to TTF font file or directory",
    )

    # ==================== Batch Sampling Arguments ====================
    batch_group = parser.add_argument_group("Batch Sampling")
    batch_group.add_argument(
        "--characters",
        type=str,
        default=None,
        help="Comma-separated characters or path to text file",
    )
    batch_group.add_argument(
        "--start_line",
        type=int,
        default=1,
        help="Start line for character file (1-indexed)",
    )
    batch_group.add_argument(
        "--end_line",
        type=int,
        default=None,
        help="End line for character file (inclusive)",
    )
    batch_group.add_argument(
        "--style_images",
        type=str,
        default=None,
        help="Comma-separated paths, directory, or glob pattern",
    )
    batch_group.add_argument(
        "--ground_truth_dir",
        type=str,
        default=None,
        help="Ground truth directory for evaluation",
    )
    batch_group.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save checkpoint every N styles",
    )
    batch_group.add_argument(
        "--dataset_split",
        type=str,
        default="train_original",
        help="Dataset split name",
    )

    # ==================== Style Transformation ====================
    style_transform_group = parser.add_argument_group("Style Transformation")
    style_transform_group.add_argument(
        "--enable_style_transform",
        action="store_true",
        default=False,
        help="Enable style transformation module",
    )
    style_transform_group.add_argument(
        "--feature_dim",
        type=int,
        default=512,
        help="Feature dimension for style transformation",
    )
    style_transform_group.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension for style transformation",
    )
    style_transform_group.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    style_transform_group.add_argument(
        "--ffn_dim", type=int, default=2048, help="FFN dimension"
    )

    # ==================== FST (Font Style Transformation) ====================
    fst_group = parser.add_argument_group("FST Enhancement")
    fst_group.add_argument(
        "--use_fst",
        action="store_true",
        default=False,
        help="Use FST-enhanced model",
    )
    fst_group.add_argument(
        "--fst_ckpt_path",
        type=str,
        default=None,
        help="Path to FST module checkpoint",
    )
    fst_group.add_argument(
        "--fst_num_queries",
        type=int,
        default=220,
        help="Number of learnable queries in FST (default 220 for 256 total)",
    )
    fst_group.add_argument(
        "--fst_query_dim",
        type=int,
        default=256,
        help="Dimension of query vectors in FST, must be divisible by num_heads",
    )
    fst_group.add_argument(
        "--fst_num_scales",
        type=int,
        default=5,
        help="Number of multi-scale features in MSSE",
    )
    fst_group.add_argument(
        "--fst_feature_channels",
        type=str,
        default="64,128,256,512,1024",
        help="Feature channels for FST module (comma-separated)",
    )
    fst_group.add_argument(
        "--style_source_same_prob",
        type=float,
        default=0.5,
        help="Probability that source and target style use same font style",
    )
    # MSS Encoder specific
    fst_group.add_argument(
        "--mss_base_channels",
        type=int,
        default=64,
        help="Base channels for MSSE (default: 64)",
    )
    fst_group.add_argument(
        "--mss_num_scales",
        type=int,
        default=5,
        help="Number of scales in MSSE (if None, uses fst_num_scales)",
    )

    fst_group.add_argument(
        "--num_consistency_pairs",
        type=int,
        default=0,
        help="Number of same-style pairs for style consistency loss (0 to disable). "
        "Example: 3 = load 3 pairs per batch where both images have same style.",
    )
    fst_group.add_argument(
        "--consistency_loss_weight",
        type=float,
        default=0.1,
        help="Weight for style consistency loss (0.0-1.0). "
        "Higher values enforce stronger style consistency.",
    )

    # ==================== Identity Loss for FST ====================
    fst_group.add_argument(
        "--num_identity_pairs",
        type=int,
        default=0,
        help="Number of same-style pairs for identity mapping loss (0 to disable). "
        "Example: 3 = load 3 pairs per batch where both images have same style.",
    )
    fst_group.add_argument(
        "--identity_loss_weight",
        type=float,
        default=0.1,
        help="Weight for identity mapping loss (0.0-1.0). "
        "Higher values enforce stronger identity constraint.",
    )
    fst_group.add_argument(
        "--identity_pair_mode",
        type=str,
        default="random",
        choices=["random", "same_style"],
        help="How to sample identity pairs:\n"
        "  'random': Each pair can have different style\n"
        "  'same_style': All pairs use same style as main sample (stronger constraint)",
    )

    # ADD THESE NEW PARAMETERS:
    fst_group.add_argument(
        "--identity_loss_type",
        type=str,
        default="frobenius",
        choices=["frobenius", "mse", "cosine"],
        help="Distance metric for identity mapping loss:\n"
        "  'frobenius': Frobenius norm ||T - I||_F (default, most stable)\n"
        "  'mse': Mean squared error\n"
        "  'cosine': Cosine distance (normalized)",
    )
    fst_group.add_argument(
        "--identity_regularization",
        type=str,
        default="orthogonal",
        choices=["orthogonal", "spectral", None],
        help="Regularization for transformation matrix:\n"
        "  'orthogonal': Enforce T^T T ≈ I (orthogonality)\n"
        "  'spectral': Penalize large singular values\n"
        "  None: No additional regularization",
    )
    fst_group.add_argument(
        "--identity_reg_weight",
        type=float,
        default=0.01,
        help="Weight for identity regularization term (0.0-1.0). "
        "Controls strength of orthogonality/spectral constraints.",
    )
    fst_group.add_argument(
        "--identity_matrix_size",
        type=int,
        default=None,
        help="Size of transformation matrix for identity loss. "
        "If None, defaults to fst_num_queries. "
        "Must match FST output query dimension.",
    )
    fst_group.add_argument(
        "--use_adaptive_identity_loss",
        action="store_true",
        default=False,
        help="Use AdaptiveIdentityMappingLoss that adjusts loss weight based on "
        "style similarity between pairs (more robust to style variations).",
    )
    fst_group.add_argument(
        "--identity_similarity_threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for adaptive identity loss (0.0-1.0). "
        "If actual style similarity > threshold, apply strong identity constraint. "
        "Otherwise, apply weak constraint to avoid false penalties.",
    )
    fst_group.add_argument(
        "--identity_adaptive_max_weight",
        type=float,
        default=1.0,
        help="Maximum weight for adaptive identity loss when styles are very similar.",
    )
    fst_group.add_argument(
        "--identity_adaptive_min_weight",
        type=float,
        default=0.1,
        help="Minimum weight for adaptive identity loss when styles are dissimilar.",
    )
    fst_group.add_argument(
        "--use_pooled_identity_loss",
        action="store_true",
        default=False,
        help="Use PooledIdentityMappingLoss that computes identity constraint over "
        "multiple same-style pairs simultaneously (stronger constraint).",
    )
    fst_group.add_argument(
        "--identity_pooled_reduction",
        type=str,
        default="mean",
        choices=["mean", "sum", "max"],
        help="Reduction method for pooled identity loss:\n"
        "  'mean': Average loss across all pairs\n"
        "  'sum': Sum loss across all pairs\n"
        "  'max': Maximum loss across all pairs",
    )
    fst_group.add_argument(
        "--identity_log_metrics",
        action="store_true",
        default=True,
        help="Log detailed identity loss metrics (diagonal/off-diagonal analysis, "
        "eigenvalue stats) to W&B and console.",
    )
    fst_group.add_argument(
        "--identity_metric_interval",
        type=int,
        default=100,
        help="Log identity loss metrics every N steps.",
    )
    # ==================== Skeleton Distance Transform ====================
    skeleton_group = parser.add_argument_group("Skeleton Distance Transform")
    skeleton_group.add_argument(
        "--use_skeleton_content",
        action="store_true",
        help="Use skeleton-distance transform for content images (prevents style leakage)",
    )

    skeleton_group.add_argument(
        "--skeleton_method",
        type=str,
        default="medial_axis",
        choices=["skeletonize", "medial_axis", "zhang_suen"],
        help="Skeletonization algorithm: "
            "'skeletonize' (morphological thinning), "
            "'medial_axis' (distance-based, robust for fonts), "
            "'zhang_suen' (Zhang-Suen algorithm)",
    )

    skeleton_group.add_argument(
        "--skeleton_distance_method",
        type=str,
        default="hybrid",
        choices=["edt", "gaussian", "hybrid"],
        help="Distance field generation method: "
            "'edt' (Euclidean Distance Transform), "
            "'gaussian' (Gaussian blur of skeleton), "
            "'hybrid' (EDT + Gaussian smoothing)",
    )

    skeleton_group.add_argument(
        "--skeleton_max_distance",
        type=float,
        default=12.0,
        help="Maximum influence radius for skeleton distance field (in pixels). "
            "Smaller values = tighter guidance, larger values = more diffuse influence.",
    )

    skeleton_group.add_argument(
        "--skeleton_sigma",
        type=float,
        default=1.5,
        help="Gaussian sigma for distance field smoothing (used in 'gaussian' and 'hybrid' methods)",
    )

    skeleton_group.add_argument(
        "--skeleton_output_mode",
        type=str,
        default="dual_channel",
        choices=["skeleton_only", "distance_only", "dual_channel"],
        help="Output mode: "
            "'skeleton_only' (binary 1-channel), "
            "'distance_only' (smooth 1-channel), "
            "'dual_channel' (skeleton + distance, 2-channel)",
    )

    skeleton_group.add_argument(
        "--skeleton_fusion_method",
        type=str,
        default="concat",
        choices=["concat", "add", "weighted"],
        help="How to fuse skeleton and distance channels in content encoder: "
            "'concat' (1x1 conv to merge), "
            "'add' (simple addition), "
            "'weighted' (learnable weighted sum)",
    )
    # ==================== Fourier-Based Frequency Decomposition ====================
    frequency_group = parser.add_argument_group("Frequency Decomposition")
    # Frequency Decomposition Arguments
    frequency_group.add_argument(
        "--use_frequency_decomp",
        action="store_true",
        help="Use frequency decomposition for content-style separation",
    )

    frequency_group.add_argument(
        "--frequency_low_cutoff",
        type=float,
        default=0.10,
        help="Boundary between low and mid frequencies (fraction of max freq)",
    )

    frequency_group.add_argument(
        "--frequency_mid_cutoff",
        type=float,
        default=0.40,
        help="Boundary between mid and high frequencies (fraction of max freq)",
    )

    frequency_group.add_argument(
        "--frequency_filter_type",
        type=str,
        default="gaussian",
        choices=["ideal", "butterworth", "gaussian"],
        help="Type of frequency filter",
    )

    frequency_group.add_argument(
        "--frequency_use_mid_band",
        action="store_true",
        default=True,
        help="Whether to use mid-frequency band",
    )

    frequency_group.add_argument(
        "--frequency_mid_target",
        type=str,
        default="both",
        choices=["content", "style", "both"],
        help="Where to send mid-frequency band",
    )
    # ==================== Direct Reward Optimization (DRO) ====================
    dro_group = parser.add_argument_group("DRO — Direct Reward Optimization")
    dro_group.add_argument(
        "--use_dro",
        action="store_true",
        help="Enable Direct Reward Optimization on top of FST diffusion loss.",
    )
    dro_group.add_argument(
        "--dro_weight",
        type=float,
        default=0.1,
        help="Weight applied to the DRO reward loss term (default: 0.1).",
    )
    dro_group.add_argument(
        "--dro_ssim_weight",
        type=float,
        default=1.0,
        help="Weight for SSIM content-fidelity reward inside DRO (default: 1.0).",
    )
    dro_group.add_argument(
        "--dro_lpips_weight",
        type=float,
        default=1.0,
        help="Weight for LPIPS style-similarity penalty inside DRO (default: 1.0).",
    )
    dro_group.add_argument(
        "--dro_reward_scale",
        type=float,
        default=1.0,
        help="Global scale applied to the composite reward (default: 1.0).",
    )
    dro_group.add_argument(
        "--dro_warmup_steps",
        type=int,
        default=0,
        help="Number of steps before DRO reward is activated (default: 0).",
    )
    # --- missing args below ---
    dro_group.add_argument(
        "--dro_max_timestep_frac",
        type=float,
        default=0.3,
        help="Only evaluate DRO reward at timesteps below this fraction of "
             "num_train_timesteps. Lower = more reliable pred_x0 (default: 0.3).",
    )
    dro_group.add_argument(
        "--dro_sharp_weight",
        type=float,
        default=0.0,
        help="Weight for sharpness reward inside DRO (default: 0.0, disabled).",
    )
    dro_group.add_argument(
        "--dro_div_weight",
        type=float,
        default=0.0,
        help="Weight for diversity penalty inside DRO (default: 0.0, disabled).",
    )
    dro_group.add_argument(
        "--dro_normalise_reward",
        action="store_true",
        default=False,
        help="Normalise composite reward to unit variance before scaling (default: False).",
    )

    # ==================== GRPO ====================
    grpo_group = parser.add_argument_group("GRPO Policy Optimization")
    grpo_group.add_argument(
        "--use_grpo",
        action="store_true",
        default=False,
        help="Enable GRPO policy gradient on top of DRO losses",
    )
    grpo_group.add_argument(
        "--grpo_group_size",
        type=int,
        default=4,
        help="Number of rollouts per prompt for group-relative baseline",
    )
    grpo_group.add_argument(
        "--grpo_clip_eps",
        type=float,
        default=0.2,
        help="PPO-style clipping epsilon for importance ratio",
    )
    grpo_group.add_argument(
        "--grpo_pg_weight",
        type=float,
        default=0.01,
        help="Weight of GRPO policy gradient loss",
    )
    grpo_group.add_argument(
        "--grpo_sample_steps",
        type=int,
        default=5,
        help="Denoising steps per rollout (shorter = faster, noisier signal)",
    )
    grpo_group.add_argument(
        "--grpo_warmup_steps",
        type=int,
        default=1000,
        help="Steps before GRPO activates",
    )
    grpo_group.add_argument(
        "--grpo_kl_coeff",
        type=float,
        default=0.01,
        help="KL penalty coefficient against frozen reference policy",
    )
    grpo_group.add_argument(
        "--grpo_reward_clip",
        type=float,
        default=5.0,
        help="Clip reward to [-clip, clip] before normalization",
    )

    # ==================== Optimization Flags ====================
    optimization_group = parser.add_argument_group("Performance Optimization")
    optimization_group.add_argument(
        "--fp16", action="store_true", default=False, help="Use FP16 precision"
    )
    optimization_group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    optimization_group.add_argument(
        "--channels_last",
        action="store_true",
        default=False,
        help="Use channels-last memory format",
    )
    optimization_group.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Use deterministic algorithms",
    )
    optimization_group.add_argument(
        "--compile", action="store_true", default=False, help="Use torch.compile"
    )
    optimization_group.add_argument(
        "--enable_xformers", action="store_true", default=False, help="Enable xFormers"
    )
    optimization_group.add_argument(
        "--enable_attention_slicing",
        action="store_true",
        default=False,
        help="Enable attention slicing",
    )
    optimization_group.add_argument(
        "--fast_sampling",
        action="store_true",
        default=False,
        help="Use fast sampling mode",
    )
    optimization_group.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count() - 1,
        help="Number of DataLoader workers",
    )

    # ==================== Evaluation ====================
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument(
        "--evaluate",
        action="store_true",
        default=False,
        help="Evaluate generated images",
    )
    eval_group.add_argument(
        "--compute_fid",
        action="store_true",
        default=False,
        help="Compute FID score",
    )

    # ==================== Logging & Tracking ====================
    logging_group = parser.add_argument_group("Logging & Tracking")
    logging_group.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Log to Weights & Biases",
    )
    logging_group.add_argument(
        "--wandb_project",
        type=str,
        default="fontdiffuser-eval",
        help="W&B project name",
    )
    logging_group.add_argument(
        "--wandb_run_name", type=str, default=None, help="W&B run name"
    )

    # ==================== Model Summary ====================
    summary_group = parser.add_argument_group("Model Summary")
    summary_group.add_argument(
        "--summary",
        action="store_true",
        default=False,
        help="If set, outputs a torchinfo summary of all core models to model_summary.txt",
    )

    # ==================== ONNX Export ====================
    export_group = parser.add_argument_group("Model Export")
    export_group.add_argument(
        "--export_onnx",
        action="store_true",
        default=False,
        help="Export model to ONNX format after training for visualization with netron.app",
    )
    export_group.add_argument(
        "--onnx_opset_version",
        type=int,
        default=17,
        help="ONNX opset version (default: 17 for better operator support)",
    )
    export_group.add_argument(
        "--onnx_export_dir",
        type=str,
        default=None,
        help="Directory to save ONNX models (defaults to output_dir/onnx)",
    )

    # ==================== Distributed Training ====================
    distributed_group = parser.add_argument_group("Distributed Training")
    distributed_group.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training",
    )

    # ==================== Legacy/Advanced ====================
    advanced_group = parser.add_argument_group("Advanced")
    advanced_group.add_argument(
        "--controlnet",
        action="store_true",
        default=False,
        help="Use ControlNet",
    )
    advanced_group.add_argument(
        "--instructpix2pix",
        action="store_true",
        default=False,
        help="Use InstructPix2Pix",
    )
    advanced_group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from.",
    )

    return parser
