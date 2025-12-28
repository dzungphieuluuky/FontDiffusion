"""
Batch sampling and evaluation for FontDiffuser
Generates images for multiple Sino-Nom characters and evaluates quality
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# Import evaluation metrics
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("Warning: lpips not available. Install with: pip install lpips")
    LPIPS_AVAILABLE = False

try:
    from pytorch_fid import fid_score
    FID_AVAILABLE = True
except ImportError:
    print("Warning: pytorch-fid not available. Install with: pip install pytorch-fid")
    FID_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not available. Install with: pip install scikit-image")
    SSIM_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not available. Install with: pip install wandb")
    WANDB_AVAILABLE = False

# Import FontDiffuser modules
from sample_optimized import (
    load_fontdiffuser_pipeline_safe,
    sampling_batch_optimized,
    image_process_batch_optimized,
    get_content_transform,
    get_style_transform
)


class QualityEvaluator:
    """Evaluates generated images using LPIPS, SSIM, and FID"""
    
    def __init__(self, device='cuda:0'):
        self.device = device
        
        # Initialize LPIPS
        if LPIPS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
            self.lpips_fn.eval()
        else:
            self.lpips_fn = None
        
        self.transform_to_tensor = transforms.ToTensor()
    
    def compute_lpips(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute LPIPS between two images"""
        if not LPIPS_AVAILABLE or self.lpips_fn is None:
            return -1.0
        
        # Convert to tensors [-1, 1]
        img1_tensor = self.transform_to_tensor(img1).unsqueeze(0).to(self.device) * 2 - 1
        img2_tensor = self.transform_to_tensor(img2).unsqueeze(0).to(self.device) * 2 - 1
        
        with torch.no_grad():
            lpips_value = self.lpips_fn(img1_tensor, img2_tensor).item()
        
        return lpips_value
    
    def compute_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute SSIM between two images"""
        if not SSIM_AVAILABLE:
            return -1.0
        
        # Convert to grayscale numpy arrays
        img1_gray = np.array(img1.convert('L'))
        img2_gray = np.array(img2.convert('L'))
        
        ssim_value = ssim(img1_gray, img2_gray, data_range=255)
        return ssim_value
    
    def compute_fid(self, real_dir: str, fake_dir: str) -> float:
        """Compute FID between two directories of images"""
        if not FID_AVAILABLE:
            return -1.0
        
        try:
            fid_value = fid_score.calculate_fid_given_paths(
                [real_dir, fake_dir],
                batch_size=50,
                device=self.device,
                dims=2048
            )
            return fid_value
        except Exception as e:
            print(f"Error computing FID: {e}")
            return -1.0
    
    def save_image(self, image: Image.Image, path: str):
        """Save PIL image to path"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Batch sampling and evaluation')
    
    # Input/Output
    parser.add_argument('--characters', type=str, required=True,
                       help='Comma-separated list of Sino-Nom characters or path to text file')
    parser.add_argument('--style_images', type=str, required=True,
                       help='Comma-separated paths to style images or directory')
    parser.add_argument('--output_dir', type=str, default='data_examples/train',
                       help='Output directory')
    parser.add_argument('--ground_truth_dir', type=str, default=None,
                       help='Directory with ground truth images for evaluation')
    
    # Model configuration
    parser.add_argument('--ckpt_dir', type=str, required=True,
                       help='Checkpoint directory')
    parser.add_argument('--ttf_path', type=str, default='ttf/KaiXinSongA.ttf',
                       help='TTF font path')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    
    # Generation parameters
    parser.add_argument('--num_inference_steps', type=int, default=15,
                       help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                       help='Guidance scale')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for generation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Optimization flags
    parser.add_argument('--fp16', action='store_true', default=True,
                       help='Use FP16 precision')
    parser.add_argument('--compile', action='store_true', default=False,
                       help='Use torch.compile')
    parser.add_argument('--channels_last', action='store_true', default=True,
                       help='Use channels last memory format')
    parser.add_argument('--enable_xformers', action='store_true', default=False,
                       help='Enable xformers')
    parser.add_argument('--fast_sampling', action='store_true', default=True,
                       help='Use fast sampling mode')
    
    # Evaluation flags
    parser.add_argument('--evaluate', action='store_true', default=True,
                       help='Evaluate generated images')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                       help='Compute FID (requires ground truth)')
    
    # Wandb configuration
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Log results to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='fontdiffuser-eval',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name')
    
    return parser.parse_args()


def load_characters(characters_arg: str) -> List[str]:
    """Load characters from comma-separated string or file"""
    if os.path.isfile(characters_arg):
        with open(characters_arg, 'r', encoding='utf-8') as f:
            chars = [line.strip() for line in f if line.strip()]
    else:
        chars = [c.strip() for c in characters_arg.split(',')]
    
    return chars


def load_style_images(style_images_arg: str) -> List[str]:
    """Load style image paths from comma-separated string or directory"""
    if os.path.isdir(style_images_arg):
        # Load all images from directory
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        style_paths = [
            os.path.join(style_images_arg, f)
            for f in os.listdir(style_images_arg)
            if os.path.splitext(f)[1].lower() in image_exts
        ]
        style_paths.sort()
    else:
        style_paths = [p.strip() for p in style_images_arg.split(',')]
    
    return style_paths


def create_args_namespace(args):
    """Create args namespace for FontDiffuser pipeline"""
    from argparse import Namespace
    
    # Import default config
    try:
        from configs.fontdiffuser import get_parser
        parser = get_parser()
        default_args = parser.parse_args([])
    except:
        default_args = Namespace()
    
    # Override with user arguments
    for key, value in vars(args).items():
        setattr(default_args, key, value)
    
    # Set required attributes
    default_args.demo = False
    default_args.character_input = True
    default_args.save_image = True
    default_args.cache_models = True
    default_args.controlnet = False
    
    # Image sizes
    if not hasattr(default_args, 'style_image_size'):
        default_args.style_image_size = (96, 96)
    if not hasattr(default_args, 'content_image_size'):
        default_args.content_image_size = (96, 96)
    
    # Generation parameters
    if not hasattr(default_args, 'algorithm_type'):
        default_args.algorithm_type = 'dpmsolver++'
    if not hasattr(default_args, 'guidance_type'):
        default_args.guidance_type = 'classifier-free'
    if not hasattr(default_args, 'method'):
        default_args.method = 'multistep'
    if not hasattr(default_args, 'order'):
        default_args.order = 2
    if not hasattr(default_args, 'model_type'):
        default_args.model_type = 'noise'
    if not hasattr(default_args, 't_start'):
        default_args.t_start = 1.0
    if not hasattr(default_args, 't_end'):
        default_args.t_end = 1e-3
    if not hasattr(default_args, 'skip_type'):
        default_args.skip_type = 'time_uniform'
    if not hasattr(default_args, 'correcting_x0_fn'):
        default_args.correcting_x0_fn = None
    if not hasattr(default_args, 'content_encoder_downsample_size'):
        default_args.content_encoder_downsample_size = 3
    if not hasattr(default_args, 'resolution'):
        default_args.resolution = 96
    
    return default_args


def generate_content_images(characters: List[str], ttf_path: str, 
                           output_dir: str, args) -> Dict[str, str]:
    """Generate and save content character images"""
    from utils import load_ttf, ttf2im, is_char_in_font
    
    content_dir = os.path.join(output_dir, 'ContentImage')
    os.makedirs(content_dir, exist_ok=True)
    
    font = load_ttf(ttf_path)
    char_paths = {}
    
    print(f"\nGenerating content images for {len(characters)} characters...")
    for i, char in enumerate(tqdm(characters)):
        if not is_char_in_font(ttf_path, char):
            print(f"Warning: '{char}' not in font, skipping...")
            continue
        
        content_img = ttf2im(font=font, char=char)
        char_path = os.path.join(content_dir, f'char{i}.png')
        content_img.save(char_path)
        char_paths[char] = char_path
    
    return char_paths


def batch_generate_images(pipe, characters: List[str], style_paths: List[str],
                          output_dir: str, args, evaluator: QualityEvaluator):
    """Generate images in batches and evaluate"""
    
    results = {
        'generations': [],
        'metrics': {
            'lpips': [],
            'ssim': [],
            'inference_times': []
        }
    }
    
    target_dir = os.path.join(output_dir, 'TargetImage')
    os.makedirs(target_dir, exist_ok=True)
    
    total_chars = len(characters)
    total_styles = len(style_paths)
    
    print(f"\nGenerating {total_chars} characters × {total_styles} styles = {total_chars * total_styles} images")
    print(f"Batch size: {args.batch_size}")
    print(f"Inference steps: {args.num_inference_steps}")
    print("="*60)
    
    for style_idx, style_path in enumerate(style_paths):
        style_name = f"style{style_idx}"
        style_dir = os.path.join(target_dir, style_name)
        os.makedirs(style_dir, exist_ok=True)
        
        print(f"\n[{style_idx+1}/{total_styles}] Processing style: {style_name}")
        print(f"Style image: {os.path.basename(style_path)}")
        
        # Generate in batches
        all_images = []
        all_chars = []
        all_times = []
        
        for batch_start in range(0, len(characters), args.batch_size):
            batch_chars = characters[batch_start:batch_start + args.batch_size]
            
            try:
                images, valid_chars, batch_time = sampling_batch_optimized(
                    args, pipe, batch_chars, style_path
                )
                
                if images is None:
                    continue
                
                all_images.extend(images)
                all_chars.extend(valid_chars)
                all_times.append(batch_time)
                
                print(f"  Batch {batch_start//args.batch_size + 1}: "
                      f"{len(valid_chars)} images in {batch_time:.2f}s "
                      f"({batch_time/len(valid_chars):.3f}s/img)")
                
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        # Save generated images and compute metrics
        print(f"  Saving and evaluating {len(all_images)} images...")
        
        for char_idx, (char, img) in enumerate(zip(all_chars, all_images)):
            # Save image
            img_name = f"{style_name}+char{characters.index(char)}.png"
            img_path = os.path.join(style_dir, img_name)
            evaluator.save_image(img, img_path)
            
            # Store generation info
            results['generations'].append({
                'character': char,
                'style': style_name,
                'style_path': style_path,
                'output_path': img_path
            })
        
        # Compute average inference time
        if all_times:
            avg_time = sum(all_times) / len(all_times)
            results['metrics']['inference_times'].append({
                'style': style_name,
                'avg_batch_time': avg_time,
                'total_images': len(all_images),
                'avg_time_per_image': avg_time / args.batch_size if args.batch_size > 0 else 0
            })
        
        print(f"  ✓ Style {style_name} complete: {len(all_images)} images generated")
    
    print("\n" + "="*60)
    print(f"✓ Generation complete! Total images: {len(results['generations'])}")
    
    return results


def evaluate_results(results: Dict, evaluator: QualityEvaluator, 
                     ground_truth_dir: str = None, compute_fid: bool = False):
    """Evaluate generated images against ground truth"""
    
    if not results['generations']:
        print("No images to evaluate")
        return results
    
    print("\n" + "="*60)
    print("EVALUATING IMAGE QUALITY")
    print("="*60)
    
    lpips_scores = []
    ssim_scores = []
    
    if ground_truth_dir and os.path.isdir(ground_truth_dir):
        print(f"\nComputing LPIPS and SSIM against ground truth...")
        
        for gen_info in tqdm(results['generations']):
            char = gen_info['character']
            style = gen_info['style']
            gen_path = gen_info['output_path']
            
            # Find corresponding ground truth
            gt_pattern = f"*{char}*.png"
            gt_files = list(Path(ground_truth_dir).glob(gt_pattern))
            
            if not gt_files:
                continue
            
            gt_path = str(gt_files[0])
            
            try:
                gen_img = Image.open(gen_path).convert('RGB')
                gt_img = Image.open(gt_path).convert('RGB')
                
                # Resize to same size if needed
                if gen_img.size != gt_img.size:
                    gt_img = gt_img.resize(gen_img.size, Image.BILINEAR)
                
                lpips_val = evaluator.compute_lpips(gen_img, gt_img)
                ssim_val = evaluator.compute_ssim(gen_img, gt_img)
                
                if lpips_val >= 0:
                    lpips_scores.append(lpips_val)
                if ssim_val >= 0:
                    ssim_scores.append(ssim_val)
                
            except Exception as e:
                print(f"Error evaluating {char}: {e}")
    
    # Compute FID if requested
    fid_score = -1.0
    if compute_fid and ground_truth_dir and FID_AVAILABLE:
        print("\nComputing FID score...")
        
        # Get first style directory as example
        if results['generations']:
            gen_dir = os.path.dirname(results['generations'][0]['output_path'])
            fid_score = evaluator.compute_fid(ground_truth_dir, gen_dir)
    
    # Store metrics
    if lpips_scores:
        results['metrics']['lpips'] = {
            'mean': float(np.mean(lpips_scores)),
            'std': float(np.std(lpips_scores)),
            'min': float(np.min(lpips_scores)),
            'max': float(np.max(lpips_scores))
        }
    
    if ssim_scores:
        results['metrics']['ssim'] = {
            'mean': float(np.mean(ssim_scores)),
            'std': float(np.std(ssim_scores)),
            'min': float(np.min(ssim_scores)),
            'max': float(np.max(ssim_scores))
        }
    
    if fid_score >= 0:
        results['metrics']['fid'] = float(fid_score)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    if lpips_scores:
        print(f"LPIPS: {results['metrics']['lpips']['mean']:.4f} ± {results['metrics']['lpips']['std']:.4f}")
    
    if ssim_scores:
        print(f"SSIM:  {results['metrics']['ssim']['mean']:.4f} ± {results['metrics']['ssim']['std']:.4f}")
    
    if fid_score >= 0:
        print(f"FID:   {fid_score:.2f}")
    
    print("="*60)
    
    return results


def log_to_wandb(results: Dict, args):
    """Log results to Weights & Biases"""
    if not WANDB_AVAILABLE or not args.use_wandb:
        return
    
    run_name = args.wandb_run_name or f"fontdiffuser_{time.strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            'num_characters': len(results['generations']),
            'num_styles': len(set(g['style'] for g in results['generations'])),
            'num_inference_steps': args.num_inference_steps,
            'guidance_scale': args.guidance_scale,
            'batch_size': args.batch_size,
            'fp16': args.fp16,
            'fast_sampling': args.fast_sampling,
        }
    )
    
    # Log metrics
    if 'lpips' in results['metrics'] and isinstance(results['metrics']['lpips'], dict):
        wandb.log({
            'lpips_mean': results['metrics']['lpips']['mean'],
            'lpips_std': results['metrics']['lpips']['std'],
        })
    
    if 'ssim' in results['metrics'] and isinstance(results['metrics']['ssim'], dict):
        wandb.log({
            'ssim_mean': results['metrics']['ssim']['mean'],
            'ssim_std': results['metrics']['ssim']['std'],
        })
    
    if 'fid' in results['metrics']:
        wandb.log({'fid': results['metrics']['fid']})
    
    # Log inference times
    if results['metrics']['inference_times']:
        total_time = sum(t['avg_batch_time'] for t in results['metrics']['inference_times'])
        total_images = sum(t['total_images'] for t in results['metrics']['inference_times'])
        
        wandb.log({
            'total_inference_time': total_time,
            'total_images': total_images,
            'avg_time_per_image': total_time / total_images if total_images > 0 else 0
        })
    
    # Log sample images
    sample_images = []
    for gen_info in results['generations'][:10]:  # First 10 images
        if os.path.exists(gen_info['output_path']):
            sample_images.append(
                wandb.Image(gen_info['output_path'], 
                           caption=f"{gen_info['character']} - {gen_info['style']}")
            )
    
    if sample_images:
        wandb.log({"sample_generations": sample_images})
    
    wandb.finish()
    print("\n✓ Results logged to W&B")


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("FONTDIFFUSER BATCH GENERATION & EVALUATION")
    print("="*60)
    
    # Load characters and styles
    characters = load_characters(args.characters)
    style_paths = load_style_images(args.style_images)
    
    print(f"\nCharacters: {len(characters)}")
    print(f"Styles: {len(style_paths)}")
    print(f"Total images to generate: {len(characters) * len(style_paths)}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate content images
    char_paths = generate_content_images(characters, args.ttf_path, 
                                         args.output_dir, args)
    
    # Create args namespace for pipeline
    pipeline_args = create_args_namespace(args)
    
    # Load pipeline
    print("\nLoading FontDiffuser pipeline...")
    pipe = load_fontdiffuser_pipeline_safe(pipeline_args)
    
    # Initialize evaluator
    evaluator = QualityEvaluator(device=args.device)
    
    # Generate images
    results = batch_generate_images(
        pipe, characters, style_paths, args.output_dir, 
        pipeline_args, evaluator
    )
    
    # Evaluate if requested
    if args.evaluate:
        results = evaluate_results(
            results, evaluator, args.ground_truth_dir, args.compute_fid
        )
    
    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # Log to wandb
    if args.use_wandb:
        log_to_wandb(results, args)
    
    print("\n" + "="*60)
    print("✓ ALL DONE!")
    print("="*60)


if __name__ == "__main__":
    main()