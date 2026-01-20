from hydra import initialize, compose
from omegaconf import DictConfig
import os
from pathlib import Path
from PIL import Image
import torch

from inference.sample import (
    load_fontdiffuser_pipeline,
    load_controlnet_pipeline,
    load_instructpix2pix_pipeline,
    sampling,
    controlnet,
    instructpix2pix,
)


def post_process_with_controlnet(
    generated_image: Image.Image,
    pipe_controlnet,
    text_prompt: str = "high quality font style",
) -> Image.Image:
    """Apply ControlNet post-processing to generated image"""
    print("Applying ControlNet post-processing...")
    processed_image = controlnet(
        text_prompt=text_prompt, pil_image=generated_image, pipe=pipe_controlnet
    )
    return processed_image


def post_process_with_instructpix2pix(
    generated_image: Image.Image,
    pipe_instructpix2pix,
    text_instruction: str = "enhance the font quality and details",
) -> Image.Image:
    """Apply InstructPix2Pix post-processing to generated image"""
    print("Applying InstructPix2Pix post-processing...")
    processed_image = instructpix2pix(
        pil_image=generated_image,
        text_prompt=text_instruction,
        pipe=pipe_instructpix2pix,
    )
    return processed_image


def post_process_pipeline(
    args, generated_image: Image.Image, cfg: DictConfig
) -> Image.Image:
    """Apply selected post-processing pipelines to generated image"""
    processed_image = generated_image

    if cfg.post_processing.use_controlnet:
        pipe_controlnet = load_controlnet_pipeline(
            args,
            config_path=cfg.post_processing.controlnet.config_path,
            ckpt_path=cfg.post_processing.controlnet.ckpt_path,
        )
        processed_image = post_process_with_controlnet(
            generated_image=processed_image,
            pipe_controlnet=pipe_controlnet,
            text_prompt=cfg.post_processing.controlnet.text_prompt,
        )
        del pipe_controlnet
        torch.cuda.empty_cache()

    if cfg.post_processing.use_instructpix2pix:
        pipe_instructpix2pix = load_instructpix2pix_pipeline(
            args, ckpt_path=cfg.post_processing.instructpix2pix.ckpt_path
        )
        processed_image = post_process_with_instructpix2pix(
            generated_image=processed_image,
            pipe_instructpix2pix=pipe_instructpix2pix,
            text_instruction=cfg.post_processing.instructpix2pix.text_instruction,
        )
        del pipe_instructpix2pix
        torch.cuda.empty_cache()

    return processed_image


def save_post_processed_image(
    save_dir: str, image: Image.Image, suffix: str = "_post_processed"
):
    """Save post-processed image"""
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"output{suffix}.png")
    image.save(output_path)
    print(f"Post-processed image saved to: {output_path}")


if __name__ == "__main__":
    from inference.sample import arg_parse

    args = arg_parse()

    # Initialize Hydra config
    with initialize(version_base=None, config_path="configs/inference"):
        cfg = compose(config_name="post_process")

        # Load FontDiffuser pipeline
        pipe_fontdiffuser = load_fontdiffuser_pipeline(args=args)

        # Generate image with FontDiffuser
        generated_image = sampling(args=args, pipe=pipe_fontdiffuser)

        if generated_image is not None:
            # Convert tensor to PIL Image if needed
            if isinstance(generated_image, torch.Tensor):
                generated_image = (
                    Image.fromarray(
                        (generated_image.cpu().numpy() * 255).astype("uint8")
                    )
                    if generated_image.dim() == 3
                    else Image.fromarray(
                        (generated_image.squeeze().cpu().numpy() * 255).astype(
                            "uint8"
                        )
                    )
                )

            # Apply post-processing
            processed_image = post_process_pipeline(args, generated_image, cfg)

            # Save post-processed image
            save_post_processed_image(
                save_dir=args.save_image_dir, image=processed_image
            )

"""Example usage:
    # Post-process with ControlNet only
python post_process.py --ckpt_dir="ckpt/" --device="cuda:0"

# Override Hydra config
python post_process.py --ckpt_dir="ckpt/" post_processing.use_instructpix2pix=true post_processing.use_controlnet=false
    """