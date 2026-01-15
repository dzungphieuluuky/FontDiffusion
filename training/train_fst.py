"""
Training script for FontDiffuserWithFST.
Integrates MSSE and FST modules into the training pipeline.
"""

import os
import math
import time
import logging
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler

from dataset.font_dataset import FontDataset
from dataset.collate_fn import CollateFN
from configs.fontdiffuser import get_parser
from src import (
    FontDiffuserModel,
    ContentPerceptualLoss,
    build_unet,
    build_style_encoder,
    build_content_encoder,
    build_ddpm_scheduler,
    build_scr,
)
from src.model import FontDiffuserWithFST
from tools import (
    save_args_to_yaml,
    x0_from_epsilon,
    reNormalize_img,
    normalize_mean_std,
)


logger = get_logger(__name__)


def get_args():
    parser = get_parser()
    
    # Add FSTDiff-specific arguments
    parser.add_argument(
        "--use_fst",
        action="store_true",
        help="Use FSTDiff enhancement with MSSE and FST modules"
    )
    parser.add_argument(
        "--fst_feature_channels",
        type=str,
        default="64,128,256,512,1024",
        help="Feature channels for FST module (comma-separated)"
    )
    parser.add_argument(
        "--fst_num_queries",
        type=int,
        default=256,
        help="Number of learnable queries in FST"
    )
    parser.add_argument(
        "--fst_query_dim",
        type=int,
        default=128,
        help="Dimension of query vectors in FST"
    )
    parser.add_argument(
        "--fst_num_scales",
        type=int,
        default=5,
        help="Number of multi-scale features in MSSE"
    )
    parser.add_argument(
        "--style_source_ratio",
        type=float,
        default=0.5,
        help="Ratio of samples that use different source/target style images"
    )
    parser.add_argument(
        "--freeze_original_encoders",
        action="store_true",
        help="Freeze original style and content encoders during training"
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def build_fontdiffuser_with_fst(args):
    """Build FontDiffuserWithFST model."""
    # First build the original FontDiffuser
    unet = build_unet(args=args)
    style_encoder = build_style_encoder(args=args)
    content_encoder = build_content_encoder(args=args)
    
    # Load pretrained weights if specified
    if args.phase_2:
        unet.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/unet.pth"))
        style_encoder.load_state_dict(
            torch.load(f"{args.phase_1_ckpt_dir}/style_encoder.pth")
        )
        content_encoder.load_state_dict(
            torch.load(f"{args.phase_1_ckpt_dir}/content_encoder.pth")
        )
    
    # Create original FontDiffuser model
    original_model = FontDiffuserModel(
        unet=unet, 
        style_encoder=style_encoder, 
        content_encoder=content_encoder
    )
    
    # Wrap with FSTDiff enhancement
    model = FontDiffuserWithFST(original_model)
    
    # Optionally freeze original encoders
    if args.freeze_original_encoders:
        logger.info("Freezing original style and content encoders")
        for param in model.style_encoder.parameters():
            param.requires_grad = False
        for param in model.content_encoder.parameters():
            param.requires_grad = False
    
    return model


def main():
    args = get_args()

    logging_dir = f"{args.output_dir}/{args.logging_dir}"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir,
    )

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        filename=f"{args.output_dir}/fontdiffuser_fst_training.log",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set training seed
    if args.seed is not None:
        set_seed(args.seed)

    # Load model and noise_scheduler
    if args.use_fst:
        logger.info("Building FontDiffuserWithFST model")
        model = build_fontdiffuser_with_fst(args)
    else:
        logger.info("Building original FontDiffuser model")
        unet = build_unet(args=args)
        style_encoder = build_style_encoder(args=args)
        content_encoder = build_content_encoder(args=args)
        if args.phase_2:
            unet.load_state_dict(torch.load(f"{args.phase_1_ckpt_dir}/unet.pth"))
            style_encoder.load_state_dict(
                torch.load(f"{args.phase_1_ckpt_dir}/style_encoder.pth")
            )
            content_encoder.load_state_dict(
                torch.load(f"{args.phase_1_ckpt_dir}/content_encoder.pth")
            )
        model = FontDiffuserModel(
            unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
        )
    
    noise_scheduler = build_ddpm_scheduler(args)

    # Build content perceptual Loss
    perceptual_loss = ContentPerceptualLoss()

    # Load SCR module for supervision
    if args.phase_2:
        scr = build_scr(args=args)
        scr.load_state_dict(torch.load(args.scr_ckpt_path))
        scr.requires_grad_(False)

    # Load the datasets
    content_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.content_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    style_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.style_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    target_transforms = transforms.Compose(
        [
            transforms.Resize(
                (args.resolution, args.resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    train_font_dataset = FontDataset(
        args=args,
        phase="train",
        transforms=[content_transforms, style_transforms, target_transforms],
        scr=args.phase_2,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_font_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        collate_fn=CollateFN(),
    )

    # Build optimizer and learning rate
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )
    
    # Configure optimizer parameters
    if args.use_fst and args.freeze_original_encoders:
        # Only optimize new FST components
        trainable_params = [
            p for p in model.parameters() if p.requires_grad
        ]
        logger.info(f"Training {len(trainable_params)} parameter groups (FST only)")
    else:
        trainable_params = model.parameters()
    
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Accelerate preparation
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move scr module to the target devices
    if args.phase_2:
        scr = scr.to(accelerator.device)

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(args.experience_name)
        save_args_to_yaml(
            args=args,
            output_file=f"{args.output_dir}/{args.experience_name}_config.yaml",
        )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")

    # Convert to the training epoch
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    for epoch in range(num_train_epochs):
        train_loss = 0.0
        for step, samples in enumerate(train_dataloader):
            model.train()
            content_images = samples["content_image"]
            style_images = samples["style_image"]
            target_images = samples["target_image"]
            nonorm_target_images = samples["nonorm_target_image"]

            with accelerator.accumulate(model):
                # Sample noise that we'll add to the samples
                noise = torch.randn_like(target_images)
                bsz = target_images.shape[0]
                
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps,
                    (bsz,),
                    device=target_images.device,
                )
                timesteps = timesteps.long()

                # Add noise to the target_images according to the noise magnitude at each timestep
                noisy_target_images = noise_scheduler.add_noise(
                    target_images, noise, timesteps
                )

                # Classifier-free training strategy
                context_mask = torch.bernoulli(torch.zeros(bsz) + args.drop_prob)
                for i, mask_value in enumerate(context_mask):
                    if mask_value == 1:
                        content_images[i, :, :, :] = 1
                        style_images[i, :, :, :] = 1

                # Forward pass - different for FST model
                if args.use_fst:
                    # For FST, we need style source and target images
                    # Option 1: Use same style image for both (simpler)
                    # Option 2: Use different reference characters (more complex)
                    
                    # Here we use the same style image for simplicity
                    # In practice, you might want to sample different reference chars
                    style_source_images = style_images.clone()
                    style_target_images = style_images
                    
                    # Randomly use different source images for diversity
                    if torch.rand(1).item() < args.style_source_ratio:
                        # Shuffle to create different source-target pairs
                        perm = torch.randperm(bsz)
                        style_source_images = style_images[perm]
                    
                    outputs = model(
                        noisy_latents=noisy_target_images,
                        timestep=timesteps,
                        content_img=content_images,
                        style_source_img=style_source_images,
                        style_target_img=style_target_images,
                        content_encoder_downsample_size=args.content_encoder_downsample_size,
                        return_dict=True,
                    )
                    noise_pred = outputs['noise_pred']
                    offset_out_sum = outputs['offset_out_sum']
                else:
                    # Original model forward
                    noise_pred, offset_out_sum = model(
                        x_t=noisy_target_images,
                        timesteps=timesteps,
                        style_images=style_images,
                        content_images=content_images,
                        content_encoder_downsample_size=args.content_encoder_downsample_size,
                    )

                # Compute losses
                diff_loss = F.mse_loss(
                    noise_pred.float(), noise.float(), reduction="mean"
                )
                offset_loss = offset_out_sum / 2

                # Output processing for content perceptual loss
                pred_original_sample_norm = x0_from_epsilon(
                    scheduler=noise_scheduler,
                    noise_pred=noise_pred,
                    x_t=noisy_target_images,
                    timesteps=timesteps,
                )
                pred_original_sample = reNormalize_img(pred_original_sample_norm)
                norm_pred_ori = normalize_mean_std(pred_original_sample)
                norm_target_ori = normalize_mean_std(nonorm_target_images)
                percep_loss = perceptual_loss.calculate_loss(
                    generated_images=norm_pred_ori,
                    target_images=norm_target_ori,
                    device=target_images.device,
                )

                loss = (
                    diff_loss
                    + args.perceptual_coefficient * percep_loss
                    + args.offset_coefficient * offset_loss
                )

                # Phase 2: Add SCR loss
                if args.phase_2:
                    neg_images = samples["neg_images"]
                    (
                        sample_style_embeddings,
                        pos_style_embeddings,
                        neg_style_embeddings,
                    ) = scr(
                        pred_original_sample_norm,
                        target_images,
                        neg_images,
                        nce_layers=args.nce_layers,
                    )
                    sc_loss = scr.calculate_nce_loss(
                        sample_s=sample_style_embeddings,
                        pos_s=pos_style_embeddings,
                        neg_s=neg_style_embeddings,
                    )
                    loss += args.sc_coefficient * sc_loss

                # Gather the losses across all processes for logging
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % args.ckpt_interval == 0:
                        save_dir = f"{args.output_dir}/global_step_{global_step}"
                        os.makedirs(save_dir, exist_ok=True)
                        
                        if args.use_fst:
                            # Save FST-enhanced model
                            torch.save(
                                model.diffusion_unet.state_dict(), 
                                f"{save_dir}/unet.pth"
                            )
                            torch.save(
                                model.style_encoder.state_dict(),
                                f"{save_dir}/style_encoder.pth",
                            )
                            torch.save(
                                model.content_encoder.state_dict(),
                                f"{save_dir}/content_encoder.pth",
                            )
                            # Save FST-specific modules
                            torch.save(
                                model.mss_encoder.state_dict(),
                                f"{save_dir}/mss_encoder.pth",
                            )
                            torch.save(
                                model.fst_module.state_dict(),
                                f"{save_dir}/fst_module.pth",
                            )
                            torch.save(
                                model.fst_projection.state_dict(),
                                f"{save_dir}/fst_projection.pth",
                            )
                            torch.save(model, f"{save_dir}/total_model_fst.pth")
                        else:
                            # Save original model
                            torch.save(model.unet.state_dict(), f"{save_dir}/unet.pth")
                            torch.save(
                                model.style_encoder.state_dict(),
                                f"{save_dir}/style_encoder.pth",
                            )
                            torch.save(
                                model.content_encoder.state_dict(),
                                f"{save_dir}/content_encoder.pth",
                            )
                            torch.save(model, f"{save_dir}/total_model.pth")
                        
                        logging.info(
                            f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}] "
                            f"Save the checkpoint on global step {global_step}"
                        )
                        print(f"Save the checkpoint on global step {global_step}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            if global_step % args.log_interval == 0:
                logging.info(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}] "
                    f"Global Step {global_step} => train_loss = {loss}"
                )
            progress_bar.set_postfix(**logs)

            # Quit
            if global_step >= args.max_train_steps:
                break

    accelerator.end_training()


if __name__ == "__main__":
    main()


"""
Example training command for FontDiffuserWithFST:

# Phase 1: Train with FST modules
python train_fst.py \
    --use_fst \
    --experience_name="fontdiffuser_fst_phase1" \
    --train_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --max_train_steps=100000 \
    --learning_rate=5e-5 \
    --ckpt_interval=5000 \
    --log_interval=100 \
    --output_dir="outputs/fst_training" \
    --style_source_ratio=0.5 \
    --mixed_precision="fp16"

# Phase 2: Fine-tune with SCR loss
python train_fst.py \
    --use_fst \
    --phase_2 \
    --phase_1_ckpt_dir="outputs/fst_training/global_step_100000" \
    --scr_ckpt_path="ckpt/scr.pth" \
    --experience_name="fontdiffuser_fst_phase2" \
    --train_batch_size=4 \
    --max_train_steps=50000 \
    --learning_rate=1e-5 \
    --output_dir="outputs/fst_training_phase2" \
    --freeze_original_encoders

# Train original model (without FST)
python train_fst.py \
    --experience_name="fontdiffuser_baseline" \
    --train_batch_size=4 \
    --max_train_steps=100000 \
    --output_dir="outputs/baseline_training"
"""