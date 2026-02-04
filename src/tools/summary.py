import argparse
import torch
from torch import nn
from torchinfo import summary

from src.modules import UNet, StyleEncoder, ContentEncoder
from src import (
    build_content_encoder,
    build_style_encoder,
    build_unet,
)
from src.model import FontDiffuserModel, FontDiffuserWithFST
from src.builders.build import build_mss_encoder, build_fst_module


def print_model_summary(model: nn.Module, input_shapes: dict, device: str = "cpu"):
    """
    Print a torchinfo summary of the model.

    Args:
        model: The PyTorch model to summarize.
        input_shapes: Dict mapping input names to shapes.
        device: Device to use for summary ("cpu" or "cuda").
    """
    try:
        from torchinfo import summary
    except ImportError:
        raise ImportError("Please install torchinfo: pip install torchinfo")

    # Prepare dummy inputs as a tuple in the order expected by the model's forward
    dummy_inputs = tuple(
        torch.zeros(shape).to(device) for shape in input_shapes.values()
    )
    summary(
        model.to(device),
        input_data=dummy_inputs,
        depth=3,
        col_names=("input_size", "output_size", "num_params", "trainable"),
    )


def build_fontdiffuser_model(device: str = "cpu") -> FontDiffuserModel:
    """
    Build base FontDiffuser model.

    Args:
        device: Device to place model on.

    Returns:
        FontDiffuserModel instance.
    """
    unet = build_unet()
    style_encoder = build_style_encoder()
    content_encoder = build_content_encoder()
    model = FontDiffuserModel(unet, style_encoder, content_encoder)
    return model.to(device)


def build_fst_model(device: str = "cpu") -> FontDiffuserWithFST:
    """
    Build FontDiffuser with FST enhancement.

    Args:
        device: Device to place model on.

    Returns:
        FontDiffuserWithFST instance.
    """
    unet = build_unet()
    style_encoder = build_style_encoder()
    content_encoder = build_content_encoder()
    mss_encoder = build_mss_encoder()
    fst_module = build_fst_module()
    
    # Build projection layers
    fst_projection = nn.Linear(fst_module.output_dim, 768)  # Adjust dims as needed
    original_style_projection = nn.Linear(style_encoder.output_dim, 768)
    
    model = FontDiffuserWithFST(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder,
        mss_encoder=mss_encoder,
        fst_module=fst_module,
        fst_projection=fst_projection,
        original_style_projection=original_style_projection,
    )
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Print model architecture summary")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["base", "fst"],
        default="base",
        help="Model type to summarize: 'base' for FontDiffuser, 'fst' for FontDiffuserWithFST",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to use for summary",
    )
    
    args = parser.parse_args()

    # Build appropriate model
    if args.model_type == "base":
        model = build_fontdiffuser_model(device=args.device)
        input_shapes = {
            "x_t": (1, 4, 96, 96),
            "timesteps": (1,),
            "style_images": (1, 1, 96, 96),
            "content_images": (1, 1, 96, 96),
            "content_encoder_downsample_size": (),
        }
        print(f"\n{'='*80}")
        print("FontDiffuser Base Model Summary")
        print(f"{'='*80}\n")
        model.log_model_info()
        
    elif args.model_type == "fst":
        model = build_fst_model(device=args.device)
        input_shapes = {
            "noisy_latents": (1, 4, 96, 96),
            "timestep": (1,),
            "content_img": (1, 1, 96, 96),
            "style_source_img": (1, 1, 96, 96),
            "style_target_img": (1, 1, 96, 96),
            "content_encoder_downsample_size": (),
        }
        print(f"\n{'='*80}")
        print("FontDiffuser with FST Model Summary")
        print(f"{'='*80}\n")
        model.log_model_info()

    print_model_summary(model, input_shapes, device=args.device)


if __name__ == "__main__":
    main()