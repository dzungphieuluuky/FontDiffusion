import torch
from torch import nn
from torchinfo import summary

from src.modules import UNet, StyleEncoder, ContentEncoder
from src import (
    build_content_encoder,
    build_style_encoder,
    build_unet,
)


def print_model_summary(model: nn.Module, input_shapes: dict, device: str = "cpu"):
    """
    Print a torchinfo summary of the model.

    Args:
        model: The PyTorch model to summarize.
        input_shapes: Dict mapping input names to shapes, e.g.,
            {"x_t": (1, 4, 96, 96), "timesteps": (1,), "style_images": (1, 1, 96, 96), "content_images": (1, 1, 96, 96)}
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


from src.model import FontDiffuserModel

unet = build_unet()
style_encoder = build_style_encoder()
content_encoder = build_content_encoder()
# Instantiate your model (example)
model = FontDiffuserModel(unet, style_encoder, content_encoder)

# Define input shapes as required by your model's forward
input_shapes = {
    "x_t": (1, 4, 96, 96),
    "timesteps": (1,),
    "style_images": (1, 1, 96, 96),
    "content_images": (1, 1, 96, 96),
    "content_encoder_downsample_size": (),  # If scalar, can omit or use (1,)
}

print_model_summary(model, input_shapes, device="cpu")
