"""
Optimized sampling for FontDiffuser - SAFE with pretrained weights
Only applies optimizations that don't change model architecture
"""

import os
import cv2
import time
import random
import numpy as np
from PIL import Image
from functools import lru_cache

import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed

from src import (FontDiffuserDPMPipeline,
                 FontDiffuserModelDPM)
from utils import (ttf2im,
                   load_ttf,
                   is_char_in_font,
                   save_args_to_yaml,
                   save_single_image,
                   save_image_with_content_style)


def arg_parse_optimized():
    """
    Parse arguments with SAFE optimization options
    """
    from configs.fontdiffuser import get_parser

    parser = get_parser()
    
    # Original arguments
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--controlnet", type=bool, default=False)
    parser.add_argument("--character_input", action="store_true")
    parser.add_argument("--content_character", type=str, default=None)
    parser.add_argument("--content_image_path", type=str, default=None)
    parser.add_argument("--style_image_path", type=str, default=None)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--save_image_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ttf_path", type=str, default="ttf/KaiXinSongA.ttf")
    
    # SAFE optimization arguments (don't affect model weights)
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use half precision (FP16) - SAFE for inference")
    parser.add_argument("--compile", action="store_true", default=False,
                       help="Use torch.compile() - PyTorch 2.0 only, SAFE")
    parser.add_argument("--channels_last", action="store_true", default=True,
                       help="Use channels last memory format - SAFE")
    parser.add_argument("--cache_models", action="store_true", default=True,
                       help="Cache model builds - SAFE")
    parser.add_argument("--fast_sampling", action="store_true", default=False,
                       help="Use fewer sampling steps - SAFE (affects quality)")
    parser.add_argument("--inference_steps", type=int, default=None,
                       help="Override num_inference_steps - SAFE")
    parser.add_argument("--warmup", action="store_true", default=False,
                       help="Run warmup iteration - SAFE")
    
    args = parser.parse_args()
    
    # Apply image size transformations
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)
    
    # Adjust steps for fast sampling (SAFE - doesn't affect weights)
    if args.fast_sampling and args.inference_steps is None:
        args.inference_steps = min(15, args.num_inference_steps)  # Cap at 15
        print(f"Fast sampling: Reducing steps to {args.inference_steps}")
    
    if args.inference_steps:
        args.num_inference_steps = args.inference_steps
    
    return args


@lru_cache(maxsize=128)
def load_ttf_cached(ttf_path):
    """Cached font loading"""
    return load_ttf(ttf_path=ttf_path)


@lru_cache(maxsize=1024)
def is_char_in_font_cached(font_path, char):
    """Cached character check"""
    return is_char_in_font(font_path=font_path, char=char)


def image_process_optimized(args, content_image=None, style_image=None):
    """Optimized image processing with caching"""
    if not args.demo:
        if args.character_input:
            assert args.content_character is not None, "content_character required"
            
            if not is_char_in_font_cached(args.ttf_path, args.content_character):
                return None, None, None
            
            font = load_ttf_cached(args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
            content_image_pil = content_image.copy()
        else:
            content_image = Image.open(args.content_image_path).convert('RGB')
            content_image_pil = None
        
        style_image = Image.open(args.style_image_path).convert('RGB')
    else:
        assert style_image is not None, "style image required"
        if args.character_input:
            assert args.content_character is not None, "content_character required"
            if not is_char_in_font_cached(args.ttf_path, args.content_character):
                return None, None, None
            font = load_ttf_cached(args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
        else:
            assert content_image is not None, "content image required"
        content_image_pil = None
    
    # Use cached transforms
    content_transform = get_content_transform(args)
    style_transform = get_style_transform(args)
    
    content_image = content_transform(content_image)[None, :]
    style_image = style_transform(style_image)[None, :]

    return content_image, style_image, content_image_pil


@lru_cache(maxsize=2)
def get_content_transform(args):
    """Cached content transform"""
    return transforms.Compose([
        transforms.Resize(args.content_image_size, 
                         interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


@lru_cache(maxsize=2)
def get_style_transform(args):
    """Cached style transform"""
    return transforms.Compose([
        transforms.Resize(args.style_image_size,
                         interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


def load_fontdiffuser_pipeline_safe(args):
    """
    SAFE pipeline loading - maintains exact model architecture
    """
    print("Loading FontDiffuser model with SAFE optimizations...")
    
    # Determine precision
    dtype = torch.float16 if args.fp16 else torch.float32
    
    # Import original build functions to ensure architecture compatibility
    try:
        # Try to import optimized versions
        from src import (
            build_unet_optimized,
            build_style_encoder_optimized,
            build_content_encoder_optimized,
            build_ddpm_scheduler_optimized
        )
        
        # Use optimized builders (they maintain architecture)
        unet = build_unet_optimized(args, optimize=args.channels_last)
        style_encoder = build_style_encoder_optimized(args, optimize=args.channels_last)
        content_encoder = build_content_encoder_optimized(args, optimize=args.channels_last)
        
    except ImportError:
        # Fallback to original build.py
        print("⚠ Using original build.py for maximum compatibility")
        from src import (
            build_unet,
            build_style_encoder,
            build_content_encoder,
            build_ddpm_scheduler
        )
        unet = build_unet(args)
        style_encoder = build_style_encoder(args)
        content_encoder = build_content_encoder(args)
        build_ddpm_scheduler_optimized = build_ddpm_scheduler
    
    # Load weights (these MUST match the architecture)
    print(f"Loading weights from {args.ckpt_dir}...")
    
    # Load with map_location to handle device placement
    unet_state_dict = torch.load(f"{args.ckpt_dir}/unet.pth", map_location='cpu')
    style_state_dict = torch.load(f"{args.ckpt_dir}/style_encoder.pth", map_location='cpu')
    content_state_dict = torch.load(f"{args.ckpt_dir}/content_encoder.pth", map_location='cpu')
    
    # Load state dicts
    unet.load_state_dict(unet_state_dict)
    style_encoder.load_state_dict(style_state_dict)
    content_encoder.load_state_dict(content_state_dict)
    
    print("✓ Model weights loaded successfully")
    
    # Apply precision conversion (SAFE - after loading weights)
    if args.fp16:
        print("Converting to FP16 precision...")
        unet = unet.half()
        style_encoder = style_encoder.half()
        content_encoder = content_encoder.half()
    
    # Apply torch.compile if requested (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling models with torch.compile...")
        try:
            unet = torch.compile(unet, mode="reduce-overhead")
            style_encoder = torch.compile(style_encoder, mode="reduce-overhead")
            content_encoder = torch.compile(content_encoder, mode="reduce-overhead")
            print("✓ Models compiled successfully")
        except Exception as e:
            print(f"⚠ torch.compile failed: {e}, continuing without compilation")
    
    # Create model
    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder
    )
    
    # Move to device with proper dtype
    model.to(args.device, dtype=dtype)
    
    print("✓ Model moved to device")
    
    # Load scheduler
    train_scheduler = build_ddpm_scheduler_optimized(args)
    print("✓ DDPM scheduler loaded")
    
    # Create pipeline
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    
    print("✓ FontDiffuser pipeline loaded with SAFE optimizations!")
    return pipe


@lru_cache(maxsize=1)
def get_fontdiffuser_pipeline_cached(args):
    """Cached pipeline for repeated inference"""
    return load_fontdiffuser_pipeline_safe(args)


def sampling_optimized(args, pipe=None, content_image=None, style_image=None):
    """Optimized sampling with safe optimizations"""
    if not args.demo:
        os.makedirs(args.save_image_dir, exist_ok=True)
        save_args_to_yaml(args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml")

    if args.seed:
        set_seed(seed=args.seed)
    
    # Use cached pipeline
    if pipe is None and args.cache_models:
        pipe = get_fontdiffuser_pipeline_cached(args)
    
    content_image_tensor, style_image_tensor, content_image_pil = image_process_optimized(
        args=args, 
        content_image=content_image, 
        style_image=style_image
    )
    
    if content_image_tensor is None:
        print("Character not in font. Please change character or font.")
        return None

    with torch.no_grad():
        # Move to device with proper dtype
        dtype = torch.float16 if args.fp16 else torch.float32
        content_image_tensor = content_image_tensor.to(args.device, dtype=dtype)
        style_image_tensor = style_image_tensor.to(args.device, dtype=dtype)
        
        print(f"Sampling with {args.num_inference_steps} steps...")
        start = time.perf_counter()
        
        images = pipe.generate(
            content_images=content_image_tensor,
            style_images=style_image_tensor,
            batch_size=1,
            order=args.order,
            num_inference_step=args.num_inference_steps,
            content_encoder_downsample_size=args.content_encoder_downsample_size,
            t_start=args.t_start,
            t_end=args.t_end,
            dm_size=args.content_image_size,
            algorithm_type=args.algorithm_type,
            skip_type=args.skip_type,
            method=args.method,
            correcting_x0_fn=args.correcting_x0_fn)
        
        end = time.perf_counter()
        inference_time = end - start
        
        if args.save_image:
            print(f"Saving image...")
            save_single_image(save_dir=args.save_image_dir, image=images[0])
            if args.character_input:
                save_image_with_content_style(
                    save_dir=args.save_image_dir,
                    image=images[0],
                    content_image_pil=content_image_pil,
                    content_image_path=None,
                    style_image_path=args.style_image_path,
                    resolution=args.resolution
                )
            else:
                save_image_with_content_style(
                    save_dir=args.save_image_dir,
                    image=images[0],
                    content_image_pil=None,
                    content_image_path=args.content_image_path,
                    style_image_path=args.style_image_path,
                    resolution=args.resolution
                )
            
            avg_time_per_step = inference_time / args.num_inference_steps * 1000
            print(f"✓ Completed in {inference_time:.2f}s ({avg_time_per_step:.1f}ms/step)")
        
        return images[0]


def main():
    """Main function with safe optimizations"""
    args = arg_parse_optimized()
    
    print("\n" + "="*60)
    print("FONTDIFFUSER - SAFE OPTIMIZED INFERENCE")
    print("="*60)
    print(f"Model weights: {args.ckpt_dir}")
    print(f"Device: {args.device}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Optimizations:")
    print(f"  • FP16: {args.fp16}")
    print(f"  • torch.compile: {args.compile}")
    print(f"  • Channels last: {args.channels_last}")
    print(f"  • Model caching: {args.cache_models}")
    print(f"  • Fast sampling: {args.fast_sampling}")
    print("="*60 + "\n")
    
    # Warmup if requested
    if args.warmup:
        print("Running warmup...")
        warmup_args = type('Args', (), {})()
        warmup_args.device = args.device
        warmup_args.content_image_size = (256, 256)
        warmup_args.style_image_size = (256, 256)
        
        dummy_content = torch.randn(1, 3, 256, 256)
        dummy_style = torch.randn(1, 3, 256, 256)
        
        # Create minimal pipeline for warmup
        from src import build_unet, build_style_encoder, build_content_encoder
        unet = build_unet(args)
        style_encoder = build_style_encoder(args)
        content_encoder = build_content_encoder(args)
        
        model = FontDiffuserModelDPM(unet=unet, style_encoder=style_encoder, content_encoder=content_encoder)
        model.to(args.device)
        model.eval()
        
        with torch.no_grad():
            _ = model(dummy_content.to(args.device), dummy_style.to(args.device), timestep=torch.tensor([500]).to(args.device))
        
        print("Warmup complete\n")
    
    # Load and run
    pipe = load_fontdiffuser_pipeline_safe(args)
    out_image = sampling_optimized(args=args, pipe=pipe)
    
    return out_image


if __name__ == "__main__":
    main()


"""SAFE Optimized Example (compatible with pretrained weights):
python sample_optimized.py \
    --ckpt_dir="ckpt/" \
    --style_image_path="data_examples/sampling/example_style.jpg" \
    --save_image \
    --character_input \
    --content_character="隆" \
    --save_image_dir="outputs_optimized/" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" \
    --guidance_scale=7.5 \
    --num_inference_steps=15 \  # Can reduce from 20
    --method="multistep" \
    --fp16 \  # SAFE - reduces memory
    --channels_last \  # SAFE - memory optimization
    --fast_sampling  # SAFE - reduces steps
"""