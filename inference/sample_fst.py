"""
Sampling script for FontDiffuserWithFST.
Supports both original and FST-enhanced models.
"""

import os
import time

import torch
import torchvision.transforms as transforms
from PIL import Image
from accelerate.utils import set_seed

from src import (
    FontDiffuserDPMPipeline,
    FontDiffuserModelDPM,
    build_ddpm_scheduler,
    build_unet,
    build_content_encoder,
    build_style_encoder,
)
from src.model import FontDiffuserWithFST
from tools.utils import (
    ttf2im,
    load_ttf,
    is_char_in_font,
    save_args_to_yaml,
    save_single_image,
    save_image_with_content_style,
)


def arg_parse():
    from src.configs.fontdiffuser import get_parser

    parser = get_parser()
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--controlnet",
        type=bool,
        default=False,
        help="If in demo mode, the controlnet can be added.",
    )
    parser.add_argument("--character_input", action="store_true")
    parser.add_argument("--content_character", type=str, default=None)
    parser.add_argument("--content_image_path", type=str, default=None)
    parser.add_argument("--style_image_path", type=str, default=None)
    parser.add_argument(
        "--style_source_image_path",
        type=str,
        default=None,
        help="Optional: different source font for style transformation",
    )
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument(
        "--save_image_dir", type=str, default=None, help="The saving directory."
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ttf_path", type=str, default="fonts/KaiXinSongA.ttf")

    # FST-specific arguments
    parser.add_argument(
        "--use_fst", action="store_true", help="Use FST-enhanced model for sampling"
    )
    parser.add_argument(
        "--fst_ckpt_dir",
        type=str,
        default=None,
        help="Directory containing FST module checkpoints",
    )

    args = parser.parse_args()
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


def image_process(args, content_image=None, style_image=None, style_source_image=None):
    """Process input images for the model."""
    if not args.demo:
        # Read content image and style image
        if args.character_input:
            assert args.content_character is not None, (
                "The content_character should not be None."
            )
            if not is_char_in_font(
                font_path=args.ttf_path, char=args.content_character
            ):
                return None, None, None, None
            font = load_ttf(ttf_path=args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
            content_image_pil = content_image.copy()
        else:
            content_image = Image.open(args.content_image_path).convert("RGB")
            content_image_pil = None

        style_image = Image.open(args.style_image_path).convert("RGB")

        # Load optional style source image for FST
        if args.use_fst and args.style_source_image_path is not None:
            style_source_image = Image.open(args.style_source_image_path).convert("RGB")
        else:
            style_source_image = style_image  # Use same image for both
    else:
        assert style_image is not None, "The style image should not be None."
        if args.character_input:
            assert args.content_character is not None, (
                "The content_character should not be None."
            )
            if not is_char_in_font(
                font_path=args.ttf_path, char=args.content_character
            ):
                return None, None, None, None
            font = load_ttf(ttf_path=args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
        else:
            assert content_image is not None, "The content image should not be None."
        content_image_pil = None

        if style_source_image is None:
            style_source_image = style_image

    # Dataset transform
    content_inference_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.content_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    style_inference_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.style_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    content_image = content_inference_transforms(content_image)[None, :]
    style_image = style_inference_transforms(style_image)[None, :]
    style_source_image = style_inference_transforms(style_source_image)[None, :]

    return content_image, style_image, style_source_image, content_image_pil


def load_fontdiffuser_fst_pipeline(args):
    """Load FontDiffuserWithFST pipeline."""
    # Load base components
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))

    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))

    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))

    # Create base model
    base_model = FontDiffuserModelDPM(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
    )

    # Wrap with FST enhancement
    model = FontDiffuserWithFST(base_model)

    # Load FST-specific weights
    fst_ckpt_dir = args.fst_ckpt_dir or args.ckpt_dir

    if os.path.exists(f"{fst_ckpt_dir}/mss_encoder.pth"):
        model.mss_encoder.load_state_dict(torch.load(f"{fst_ckpt_dir}/mss_encoder.pth"))
        print("Loaded MSSE encoder state_dict")

    if os.path.exists(f"{fst_ckpt_dir}/fst_module.pth"):
        model.fst_module.load_state_dict(torch.load(f"{fst_ckpt_dir}/fst_module.pth"))
        print("Loaded FST module state_dict")

    if os.path.exists(f"{fst_ckpt_dir}/fst_projection.pth"):
        model.fst_projection.load_state_dict(
            torch.load(f"{fst_ckpt_dir}/fst_projection.pth")
        )
        print("Loaded FST projection state_dict")

    model.to(args.device)
    print("Loaded the FST-enhanced model state_dict successfully!")

    # Load the training ddpm_scheduler
    train_scheduler = build_ddpm_scheduler(args=args)
    print("Loaded training DDPM scheduler successfully!")

    # Create custom DPM pipeline for FST model
    pipe = FontDiffuserFSTPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    print("Loaded FST DPM-Solver pipeline successfully!")

    return pipe


def load_fontdiffuser_pipeline(args):
    """Load original FontDiffuser pipeline."""
    # Load the model state_dict
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))
    model = FontDiffuserModelDPM(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
    )
    model.to(args.device)
    print("Loaded the model state_dict successfully!")

    # Load the training ddpm_scheduler
    train_scheduler = build_ddpm_scheduler(args=args)
    print("Loaded training DDPM scheduler successfully!")

    # Load the DPM_Solver to generate the sample
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    print("Loaded DPM-Solver pipeline successfully!")

    return pipe


class FontDiffuserFSTPipeline(FontDiffuserDPMPipeline):
    """Custom pipeline for FST-enhanced model."""

    def generate(
        self,
        content_images,
        style_images,
        style_source_images=None,
        batch_size=1,
        order=2,
        num_inference_step=15,
        content_encoder_downsample_size=3,
        t_start=1.0,
        t_end=1e-3,
        dm_size=(64, 64),
        algorithm_type="dpmsolver++",
        skip_type="time_uniform",
        method="multistep",
        correcting_x0_fn=None,
    ):
        """Generate samples using FST-enhanced model."""
        from src.dpm_solver import NoiseScheduleVP, DPM_Solver

        # Use same style image for source if not provided
        if style_source_images is None:
            style_source_images = style_images

        # Classifier-free guidance setup
        if self.guidance_type == "classifier-free":
            content_images_uncond = torch.ones_like(content_images)
            style_images_uncond = torch.ones_like(style_images)
            style_source_images_uncond = torch.ones_like(style_source_images)

            content_images = torch.cat([content_images, content_images_uncond], dim=0)
            style_images = torch.cat([style_images, style_images_uncond], dim=0)
            style_source_images = torch.cat(
                [style_source_images, style_source_images_uncond], dim=0
            )

        # Define model function for DPM-Solver
        def model_fn(x, t_continuous):
            t = t_continuous * self.ddpm_train_scheduler.num_train_timesteps

            # FST model forward
            with torch.inference_mode():
                outputs = self.model(
                    noisy_latents=x,
                    timestep=t,
                    content_img=content_images,
                    style_source_img=style_source_images,
                    style_target_img=style_images,
                    content_encoder_downsample_size=content_encoder_downsample_size,
                    return_dict=True,
                )
                noise_pred = outputs["noise_pred"]

            # Classifier-free guidance
            if self.guidance_type == "classifier-free":
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            return noise_pred

        # Initialize noise schedule
        noise_schedule = NoiseScheduleVP(schedule="discrete", betas=self.betas)

        # Initialize latent
        x_T = torch.randn(
            (batch_size, 1, dm_size[0], dm_size[1]), device=content_images.device
        )

        # Create DPM-Solver
        dpm_solver = DPM_Solver(
            model_fn=model_fn,
            noise_schedule=noise_schedule,
            algorithm_type=algorithm_type,
            correcting_x0_fn=correcting_x0_fn,
        )

        # Generate samples
        x_sample = dpm_solver.sample(
            x=x_T,
            steps=num_inference_step,
            t_start=t_start,
            t_end=t_end,
            order=order,
            skip_type=skip_type,
            method=method,
        )

        return x_sample


def sampling(args, pipe, content_image=None, style_image=None, style_source_image=None):
    """Main sampling function."""
    if not args.demo:
        os.makedirs(args.save_image_dir, exist_ok=True)
        # Saving sampling config
        save_args_to_yaml(
            args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml"
        )

    if args.seed:
        set_seed(seed=args.seed)

    # Process images
    result = image_process(
        args=args,
        content_image=content_image,
        style_image=style_image,
        style_source_image=style_source_image,
    )

    if result[0] is None:
        print(
            f"The content_character you provided is not in the ttf. "
            f"Please change the content_character or you can change the ttf."
        )
        return None

    content_image, style_image, style_source_image, content_image_pil = result

    with torch.inference_mode():
        content_image = content_image.to(args.device)
        style_image = style_image.to(args.device)
        style_source_image = style_source_image.to(args.device)

        print(f"Sampling by DPM-Solver++ ......")
        start = time.time()

        if args.use_fst:
            # Use FST pipeline
            images = pipe.generate(
                content_images=content_image,
                style_images=style_image,
                style_source_images=style_source_image,
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
                correcting_x0_fn=args.correcting_x0_fn,
            )
        else:
            # Use original pipeline
            images = pipe.generate(
                content_images=content_image,
                style_images=style_image,
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
                correcting_x0_fn=args.correcting_x0_fn,
            )

        end = time.time()

        if args.save_image:
            print(f"Saving the image ......")
            save_single_image(save_dir=args.save_image_dir, image=images[0])
            if args.character_input:
                save_image_with_content_style(
                    save_dir=args.save_image_dir,
                    image=images[0],
                    content_image_pil=content_image_pil,
                    content_image_path=None,
                    style_image_path=args.style_image_path,
                    resolution=args.resolution,
                )
            else:
                save_image_with_content_style(
                    save_dir=args.save_image_dir,
                    image=images[0],
                    content_image_pil=None,
                    content_image_path=args.content_image_path,
                    style_image_path=args.style_image_path,
                    resolution=args.resolution,
                )
            print(f"Finish the sampling process, costing time {end - start}s")

        return images[0]


if __name__ == "__main__":
    args = arg_parse()

    # Load appropriate pipeline
    if args.use_fst:
        pipe = load_fontdiffuser_fst_pipeline(args=args)
    else:
        pipe = load_fontdiffuser_pipeline(args=args)

    out_image = sampling(args=args, pipe=pipe)


"""Sample With FST Model and Character Input
python sample_fst.py \
    --use_fst \
    --ckpt_dir="ckpt/fst_model/" \
    --fst_ckpt_dir="ckpt/fst_model/" \
    --style_image_path="data_examples/sampling/example_style.jpg" \
    --style_source_image_path="data_examples/sampling/example_style_source.jpg" \
    --save_image \
    --character_input \
    --content_character="隆" \
    --save_image_dir="outputs/fst_samples/" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" \
    --guidance_scale=7.5 \
    --num_inference_steps=20 \
    --method="multistep"
"""

"""Sample With FST Model and Content Image
python sample_fst.py \
    --use_fst \
    --ckpt_dir="ckpt/fst_model/" \
    --content_image_path="data_examples/sampling/example_content.jpg" \
    --style_image_path="data_examples/sampling/example_style.jpg" \
    --save_image \
    --save_image_dir="outputs/fst_samples/" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" \
    --guidance_scale=7.5 \
    --num_inference_steps=20 \
    --method="multistep"
"""

"""Sample With Original Model (Baseline)
python sample_fst.py \
    --ckpt_dir="ckpt/original_model/" \
    --content_image_path="data_examples/sampling/example_content.jpg" \
    --style_image_path="data_examples/sampling/example_style.jpg" \
    --save_image \
    --save_image_dir="outputs/baseline_samples/" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" \
    --guidance_scale=7.5 \
    --num_inference_steps=20 \
    --method="multistep"
"""
