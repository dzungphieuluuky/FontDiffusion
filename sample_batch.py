"""
FontDiffuser Batch Generation and Evaluation Script
Optimized for memory efficiency, resource safety, and maintainability
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Union, Generator, Any
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# Import FontDiffuser modules
from src.dpm_solver.pipeline_dpm_solver import FontDiffuserDPMPipeline
from src.model import StyleTransformationModule
from utilities import (
    save_model_checkpoint,
    load_model_checkpoint,
    get_hf_bar,
)
from logging_utils import setup_logging
from sample_optimized import (
    load_fontdiffuser_pipeline,
    get_content_transform,
    get_style_transform,
)
from utils import (
    load_ttf,
    ttf2im,
    is_char_in_font,
)
from filename_utils import (
    get_content_filename,
    get_target_filename,
    compute_file_hash,
)

# Constants
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_FONT_EXTENSIONS = {".ttf", ".otf", ".TTF", ".OTF"}
MAX_BATCH_SIZE = 32
MAX_FONTS_IN_MEMORY = 10
MAX_OPEN_FILES = 100

# Configure optional imports with proper error handling
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "LPIPS not available. Install with: pip install lpips"
    )

try:
    from pytorch_fid import fid_score
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "FID not available. Install with: pip install pytorch-fid"
    )

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "SSIM not available. Install with: pip install scikit-image"
    )

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "WandB not available. Install with: pip install wandb"
    )

logger = setup_logging(level=logging.INFO, name="BatchSampler")


@dataclass
class GenerationConfig:
    """Validated configuration for batch generation"""
    batch_size: int = 4
    num_inference_steps: int = 15
    guidance_scale: float = 7.5
    enable_style_transform: bool = False
    fp16: bool = False
    compile_model: bool = False
    channels_last: bool = True
    enable_xformers: bool = False
    enable_attention_slicing: bool = False
    device: str = "cuda"
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        self.batch_size = max(1, min(self.batch_size, MAX_BATCH_SIZE))
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if self.guidance_scale < 1.0:
            raise ValueError("guidance_scale must be >= 1.0")
    
    @classmethod
    def from_args(cls, args: Namespace) -> 'GenerationConfig':
        """Create config from command line arguments"""
        return cls(
            batch_size=getattr(args, 'batch_size', 4),
            num_inference_steps=getattr(args, 'num_inference_steps', 15),
            guidance_scale=getattr(args, 'guidance_scale', 7.5),
            enable_style_transform=getattr(args, 'enable_style_transform', False),
            fp16=getattr(args, 'fp16', False),
            compile_model=getattr(args, 'compile', False),
            channels_last=getattr(args, 'channels_last', True),
            enable_xformers=getattr(args, 'enable_xformers', False),
            enable_attention_slicing=getattr(args, 'enable_attention_slicing', False),
            device=getattr(args, 'device', 'cuda'),
        )


class ResourceManager:
    """Manages file resources with automatic cleanup"""
    
    def __init__(self, max_open_files: int = MAX_OPEN_FILES):
        self.max_open_files = max_open_files
        self._open_images: List[Image.Image] = []
        self._open_files = []
    
    @contextmanager
    def open_image(self, path: str) -> Generator[Image.Image, None, None]:
        """Context manager for opening images"""
        img = None
        try:
            img = Image.open(path)
            self._track_resource(img, path)
            yield img
        finally:
            if img:
                self._close_resource(img, path)
    
    def _track_resource(self, resource: Any, path: str) -> None:
        """Track opened resource"""
        if len(self._open_images) >= self.max_open_files:
            self._cleanup_oldest()
        self._open_images.append(resource)
    
    def _close_resource(self, resource: Any, path: str) -> None:
        """Close resource and remove from tracking"""
        try:
            if hasattr(resource, 'close'):
                resource.close()
        except Exception as e:
            logger.debug(f"Error closing resource {path}: {e}")
        finally:
            if resource in self._open_images:
                self._open_images.remove(resource)
    
    def _cleanup_oldest(self) -> None:
        """Clean up oldest resources"""
        if self._open_images:
            oldest = self._open_images.pop(0)
            try:
                if hasattr(oldest, 'close'):
                    oldest.close()
            except:
                pass
    
    def cleanup_all(self) -> None:
        """Clean up all tracked resources"""
        for resource in self._open_images[:]:
            self._close_resource(resource, "cleanup")
        self._open_images.clear()


class FontManager:
    """Manages font files with lazy loading and memory limits"""
    
    def __init__(self, ttf_path: str, max_fonts_in_memory: int = MAX_FONTS_IN_MEMORY):
        self.max_fonts_in_memory = max_fonts_in_memory
        self.font_metadata: Dict[str, Dict[str, Any]] = {}
        self._loaded_fonts: Dict[str, Any] = {}
        self._load_order: List[str] = []
        self._load_font_metadata(ttf_path)
    
    def _load_font_metadata(self, ttf_path: str) -> None:
        """Load font metadata without loading fonts into memory"""
        font_paths = self._discover_fonts(ttf_path)
        
        if not font_paths:
            raise ValueError(f"No valid font files found at: {ttf_path}")
        
        logger.info(f"Found {len(font_paths)} font files")
        
        for font_path in font_paths:
            font_name = Path(font_path).stem
            self.font_metadata[font_name] = {
                'path': font_path,
                'name': font_name,
                'loaded': False
            }
        
        logger.info(f"Registered {len(self.font_metadata)} fonts")
    
    def _discover_fonts(self, path: str) -> List[str]:
        """Discover font files from various input types"""
        import glob
        
        if "*" in path:
            font_files = glob.glob(path, recursive=True)
        elif os.path.isdir(path):
            font_files = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if Path(f).suffix.lower() in VALID_FONT_EXTENSIONS
            ]
        elif os.path.isfile(path):
            font_files = [path]
        else:
            raise ValueError(f"Invalid font path: {path}")
        
        # Filter and validate
        valid_fonts = []
        for font_file in sorted(font_files):
            if Path(font_file).suffix.lower() in VALID_FONT_EXTENSIONS:
                if os.path.getsize(font_file) > 0:
                    valid_fonts.append(font_file)
                else:
                    logger.warning(f"Skipping empty font file: {font_file}")
        
        return valid_fonts
    
    def get_font(self, font_name: str) -> Any:
        """Get font object with lazy loading and LRU cache"""
        if font_name not in self.font_metadata:
            raise ValueError(f"Font not registered: {font_name}")
        
        # Check if font is already loaded
        if font_name in self._loaded_fonts:
            # Update access order
            if font_name in self._load_order:
                self._load_order.remove(font_name)
            self._load_order.append(font_name)
            return self._loaded_fonts[font_name]
        
        # Load font
        font_path = self.font_metadata[font_name]['path']
        try:
            font = load_ttf(font_path)
            
            # Manage memory limit
            if len(self._loaded_fonts) >= self.max_fonts_in_memory:
                oldest_font = self._load_order.pop(0)
                del self._loaded_fonts[oldest_font]
                self.font_metadata[oldest_font]['loaded'] = False
            
            # Store font
            self._loaded_fonts[font_name] = font
            self._load_order.append(font_name)
            self.font_metadata[font_name]['loaded'] = True
            
            logger.debug(f"Loaded font: {font_name}")
            return font
            
        except Exception as e:
            logger.error(f"Failed to load font {font_name}: {e}")
            raise
    
    def get_font_names(self) -> List[str]:
        """Get list of available font names"""
        return list(self.font_metadata.keys())
    
    def get_font_path(self, font_name: str) -> str:
        """Get font file path"""
        if font_name not in self.font_metadata:
            raise ValueError(f"Font not found: {font_name}")
        return self.font_metadata[font_name]['path']
    
    def is_char_in_font(self, font_name: str, char: str) -> bool:
        """Check if character exists in font"""
        font_path = self.get_font_path(font_name)
        return is_char_in_font(font_path, char)
    
    def get_available_chars(self, font_name: str, characters: List[str]) -> List[str]:
        """Get list of characters available in specific font"""
        return [char for char in characters if self.is_char_in_font(font_name, char)]
    
    def cleanup(self) -> None:
        """Clean up loaded fonts"""
        self._loaded_fonts.clear()
        self._load_order.clear()
        for meta in self.font_metadata.values():
            meta['loaded'] = False


class GenerationTracker:
    """Tracks generated images with hash-based deduplication"""
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.generated_hashes: Set[str] = set()
        self.generations: List[Dict[str, Any]] = []
        self.checkpoint_path = checkpoint_path
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load existing generations from checkpoint"""
        try:
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            generations = data.get('generations', [])
            seen_hashes = set()
            unique_generations = []
            
            for gen in generations:
                target_hash = gen.get('target_hash')
                if not target_hash:
                    # Compute hash if missing
                    char = gen.get('character', '')
                    style = gen.get('style', '')
                    font = gen.get('font', '')
                    if char and style:
                        target_hash = compute_file_hash(char, style, font)
                
                if target_hash and target_hash not in seen_hashes:
                    seen_hashes.add(target_hash)
                    unique_generations.append(gen)
            
            self.generated_hashes = seen_hashes
            self.generations = unique_generations
            
            logger.info(f"Loaded {len(self.generations)} unique generations from checkpoint")
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in checkpoint {checkpoint_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise
    
    def is_generated(self, char: str, style: str, font: str = "") -> bool:
        """Check if combination has been generated"""
        target_hash = compute_file_hash(char, style, font)
        return target_hash in self.generated_hashes
    
    def mark_generated(self, char: str, style: str, font: str = "") -> str:
        """Mark combination as generated and return hash"""
        target_hash = compute_file_hash(char, style, font)
        self.generated_hashes.add(target_hash)
        return target_hash
    
    def add_generation(self, generation: Dict[str, Any]) -> None:
        """Add generation record"""
        self.generations.append(generation)
        # Update hash set
        char = generation.get('character', '')
        style = generation.get('style', '')
        font = generation.get('font', '')
        if char and style:
            self.mark_generated(char, style, font)
    
    def save_checkpoint(self, output_dir: str, results: Dict[str, Any]) -> None:
        """Save current state to checkpoint"""
        checkpoint_path = os.path.join(output_dir, "results_checkpoint.json")
        
        # Ensure results includes current generations
        results['generations'] = self.generations
        
        try:
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved checkpoint with {len(self.generations)} generations")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise


class QualityEvaluator:
    """Evaluates generated images with proper resource management"""
    
    def __init__(self, device: str = "cuda", resource_manager: Optional[ResourceManager] = None):
        self.device = device
        self.resource_manager = resource_manager or ResourceManager()
        
        # Initialize LPIPS if available
        self.lpips_fn = None
        if LPIPS_AVAILABLE:
            try:
                self.lpips_fn = lpips.LPIPS(net='alex').to(device)
                self.lpips_fn.eval()
            except Exception as e:
                logger.warning(f"Failed to initialize LPIPS: {e}")
        
        self.transform_to_tensor = transforms.ToTensor()
    
    def compute_lpips(self, img1_path: str, img2_path: str) -> float:
        """Compute LPIPS between two images from paths"""
        if not self.lpips_fn:
            return -1.0
        
        try:
            with self.resource_manager.open_image(img1_path) as img1, \
                 self.resource_manager.open_image(img2_path) as img2:
                
                # Convert to tensors
                img1_tensor = self._pil_to_tensor(img1).to(self.device)
                img2_tensor = self._pil_to_tensor(img2).to(self.device)
                
                with torch.inference_mode():
                    lpips_value = self.lpips_fn(img1_tensor, img2_tensor).item()
                
                return lpips_value
                
        except Exception as e:
            logger.error(f"Error computing LPIPS: {e}")
            return -1.0
    
    def compute_ssim(self, img1_path: str, img2_path: str) -> float:
        """Compute SSIM between two images from paths"""
        if not SSIM_AVAILABLE:
            return -1.0
        
        try:
            with self.resource_manager.open_image(img1_path) as img1, \
                 self.resource_manager.open_image(img2_path) as img2:
                
                # Convert to grayscale
                img1_gray = np.array(img1.convert('L'))
                img2_gray = np.array(img2.convert('L'))
                
                ssim_value = ssim(img1_gray, img2_gray, data_range=255)
                return ssim_value
                
        except Exception as e:
            logger.error(f"Error computing SSIM: {e}")
            return -1.0
    
    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL image to normalized tensor"""
        tensor = self.transform_to_tensor(img).unsqueeze(0)
        return tensor * 2 - 1  # Normalize to [-1, 1]
    
    def compute_fid(self, real_dir: str, fake_dir: str) -> float:
        """Compute FID between two directories"""
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
            logger.error(f"Error computing FID: {e}")
            return -1.0


class BatchGenerator:
    """Main batch generation controller"""
    
    def __init__(self, args: Namespace):
        self.args = args
        self.config = GenerationConfig.from_args(args)
        self.resource_manager = ResourceManager()
        
        # Initialize components
        self.font_manager: Optional[FontManager] = None
        self.generation_tracker: Optional[GenerationTracker] = None
        self.evaluator: Optional[QualityEvaluator] = None
        self.pipeline: Optional[FontDiffuserDPMPipeline] = None
        
        # Results storage
        self.results: Dict[str, Any] = {
            'generations': [],
            'metrics': {
                'lpips': [],
                'ssim': [],
                'inference_times': [],
            },
            'dataset_split': args.dataset_split,
        }
    
    def setup(self) -> None:
        """Setup all components"""
        logger.info("Setting up batch generator...")
        
        # Create output directory
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Initialize font manager
        self.font_manager = FontManager(self.args.ttf_path)
        
        # Initialize generation tracker
        checkpoint_path = os.path.join(self.args.output_dir, "results_checkpoint.json")
        self.generation_tracker = GenerationTracker(
            checkpoint_path if os.path.exists(checkpoint_path) else None
        )
        
        # Initialize evaluator
        self.evaluator = QualityEvaluator(
            device=self.config.device,
            resource_manager=self.resource_manager
        )
        
        # Load pipeline
        self.pipeline = self._load_pipeline()
        
        logger.info("Setup complete")
    
    def _load_pipeline(self) -> FontDiffuserDPMPipeline:
        """Load and configure FontDiffuser pipeline"""
        pipeline_args = self._create_pipeline_args()
        
        logger.info("Loading FontDiffuser pipeline...")
        pipeline = load_fontdiffuser_pipeline(pipeline_args)
        
        # Apply optimizations
        if self.config.compile_model:
            self._compile_model(pipeline)
        
        if self.config.enable_attention_slicing:
            pipeline.enable_attention_slicing()
        
        return pipeline
    
    def _create_pipeline_args(self) -> Namespace:
        """Create pipeline arguments namespace"""
        pipeline_args = Namespace()
        
        # Copy relevant arguments
        for key, value in vars(self.args).items():
            setattr(pipeline_args, key, value)
        
        # Ensure required attributes
        pipeline_args.demo = False
        pipeline_args.character_input = True
        pipeline_args.save_image = True
        pipeline_args.cache_models = True
        
        # Set image sizes
        if isinstance(pipeline_args.style_image_size, int):
            pipeline_args.style_image_size = (pipeline_args.style_image_size,) * 2
        if isinstance(pipeline_args.content_image_size, int):
            pipeline_args.content_image_size = (pipeline_args.content_image_size,) * 2
        
        # Generation parameters
        pipeline_args.algorithm_type = getattr(pipeline_args, 'algorithm_type', 'dpmsolver++')
        pipeline_args.guidance_type = getattr(pipeline_args, 'guidance_type', 'classifier-free')
        pipeline_args.method = getattr(pipeline_args, 'method', 'multistep')
        pipeline_args.order = getattr(pipeline_args, 'order', 2)
        pipeline_args.t_start = getattr(pipeline_args, 't_start', 1.0)
        pipeline_args.t_end = getattr(pipeline_args, 't_end', 1e-3)
        
        return pipeline_args
    
    def _compile_model(self, pipeline: FontDiffuserDPMPipeline) -> None:
        """Compile model components for performance"""
        try:
            logger.info("Compiling model components...")
            
            model_config = pipeline.model.config
            
            if hasattr(model_config, 'unet'):
                model_config.unet = torch.compile(model_config.unet)
            
            if hasattr(model_config, 'style_encoder'):
                model_config.style_encoder = torch.compile(model_config.style_encoder)
            
            if hasattr(model_config, 'content_encoder'):
                model_config.content_encoder = torch.compile(model_config.content_encoder)
            
            logger.info("Model compilation complete")
            
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
    
    def generate_content_images(self, characters: List[str]) -> Dict[str, str]:
        """Generate content images for characters"""
        content_dir = os.path.join(self.args.output_dir, "ContentImage")
        os.makedirs(content_dir, exist_ok=True)
        
        font_names = self.font_manager.get_font_names()
        if not font_names:
            raise ValueError("No fonts available")
        
        char_paths = {}
        generated = 0
        skipped = 0
        failed = 0
        
        logger.info(f"Generating content images for {len(characters)} characters...")
        
        for char in get_hf_bar(characters, desc="Content Images"):
            # Find font that supports character
            font_name = self._find_supporting_font(char, font_names)
            if not font_name:
                logger.warning(f"No font supports character: '{char}'")
                failed += 1
                continue
            
            # Check if already exists
            content_filename = get_content_filename(char)
            content_path = os.path.join(content_dir, content_filename)
            
            if os.path.exists(content_path):
                char_paths[char] = content_path
                skipped += 1
                continue
            
            # Generate new image
            try:
                font = self.font_manager.get_font(font_name)
                content_img = ttf2im(font=font, char=char)
                content_img.save(content_path)
                char_paths[char] = content_path
                generated += 1
                
            except Exception as e:
                logger.error(f"Failed to generate content for '{char}': {e}")
                failed += 1
        
        logger.info(f"Content images: {generated} new, {skipped} existing, {failed} failed")
        return char_paths
    
    def _find_supporting_font(self, char: str, font_names: List[str]) -> Optional[str]:
        """Find first font that supports character"""
        for font_name in font_names:
            if self.font_manager.is_char_in_font(font_name, char):
                return font_name
        return None
    
    def generate_batch(
        self,
        characters: List[str],
        style_images: List[Tuple[str, str]]
    ) -> Dict[str, Any]:
        """Generate images for all character-style combinations"""
        
        # Generate content images
        char_paths = self.generate_content_images(characters)
        if not char_paths:
            raise ValueError("No valid content images generated")
        
        # Setup target directory
        target_base_dir = os.path.join(self.args.output_dir, "TargetImage")
        os.makedirs(target_base_dir, exist_ok=True)
        
        # Statistics
        total_generated = 0
        total_skipped = 0
        total_failed = 0
        start_time = time.time()
        
        logger.info(f"Starting batch generation with {len(style_images)} styles...")
        
        # Process each style
        for style_idx, (style_path, style_name) in enumerate(style_images):
            try:
                style_start = time.time()
                
                # Create style directory
                style_dir = os.path.join(target_base_dir, style_name)
                os.makedirs(style_dir, exist_ok=True)
                
                # Generate images for this style
                style_generated, style_skipped, style_failed = self._generate_for_style(
                    characters, char_paths, style_path, style_name, style_dir
                )
                
                # Update statistics
                total_generated += style_generated
                total_skipped += style_skipped
                total_failed += style_failed
                
                # Log progress
                style_time = time.time() - style_start
                logger.info(
                    f"Style {style_idx+1}/{len(style_images)} - {style_name}: "
                    f"{style_generated} generated, {style_skipped} skipped, "
                    f"{style_failed} failed in {style_time:.1f}s"
                )
                
                # Save checkpoint periodically
                if self.args.save_interval > 0 and (style_idx + 1) % self.args.save_interval == 0:
                    self._save_progress()
                
            except Exception as e:
                logger.error(f"Failed to process style {style_name}: {e}")
                total_failed += len(characters)
        
        # Final statistics
        total_time = time.time() - start_time
        total_pairs = len(characters) * len(style_images)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch Generation Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Total time:          {total_time/60:.1f} minutes")
        logger.info(f"Total pairs:         {total_pairs}")
        logger.info(f"Generated:           {total_generated}")
        logger.info(f"Skipped (existing):  {total_skipped}")
        logger.info(f"Failed:              {total_failed}")
        logger.info(f"Success rate:        {(total_generated/total_pairs)*100:.1f}%")
        logger.info(f"{'='*60}")
        
        return self.results
    
    def _generate_for_style(
        self,
        characters: List[str],
        char_paths: Dict[str, str],
        style_path: str,
        style_name: str,
        style_dir: str
    ) -> Tuple[int, int, int]:
        """Generate images for a single style"""
        generated = 0
        skipped = 0
        failed = 0
        
        # Filter characters that need generation
        chars_to_generate = []
        for char in characters:
            if char not in char_paths:
                failed += 1
                continue
            
            if self.generation_tracker.is_generated(char, style_name):
                skipped += 1
                continue
            
            chars_to_generate.append(char)
        
        if not chars_to_generate:
            return 0, skipped, failed
        
        # Generate in batches
        batch_size = self.config.batch_size
        for batch_start in range(0, len(chars_to_generate), batch_size):
            batch_chars = chars_to_generate[batch_start:batch_start + batch_size]
            
            try:
                # Generate batch
                batch_images = self._generate_batch_images(batch_chars, style_path, style_name)
                
                if not batch_images:
                    failed += len(batch_chars)
                    continue
                
                # Save images and update tracker
                for char, img in zip(batch_chars, batch_images):
                    try:
                        self._save_generated_image(
                            char, style_name, style_dir, img, char_paths[char]
                        )
                        generated += 1
                    except Exception as e:
                        logger.error(f"Failed to save image for '{char}': {e}")
                        failed += 1
                        
            except Exception as e:
                logger.error(f"Batch generation failed: {e}")
                failed += len(batch_chars)
        
        return generated, skipped, failed
    
    def _generate_batch_images(
        self,
        characters: List[str],
        style_path: str,
        style_name: str
    ) -> Optional[List[Image.Image]]:
        """Generate images for a batch of characters"""
        try:
            # Load style image
            with self.resource_manager.open_image(style_path) as style_img:
                style_img = style_img.convert('RGB')
                
                # Get font (use first available)
                font_name = self.font_manager.get_font_names()[0]
                font = self.font_manager.get_font(font_name)
                
                # Generate content images
                content_images = []
                content_transforms = get_content_transform(self.args.content_image_size)
                
                for char in characters:
                    content_pil = ttf2im(font=font, char=char)
                    content_tensor = content_transforms(content_pil)
                    content_images.append(content_tensor)
                
                if not content_images:
                    return None
                
                # Prepare batches
                style_transform = get_style_transform(self.args.style_image_size)
                style_tensor = style_transform(style_img)[None, :]
                style_batch = style_tensor.repeat(len(content_images), 1, 1, 1)
                content_batch = torch.stack(content_images)
                
                # Move to device
                dtype = torch.float16 if self.config.fp16 else torch.float32
                content_batch = content_batch.to(self.config.device, dtype=dtype)
                style_batch = style_batch.to(self.config.device, dtype=dtype)
                
                # Generate images
                with torch.inference_mode():
                    images = self.pipeline.generate(
                        content_images=content_batch,
                        style_images=style_batch,
                        batch_size=len(content_batch),
                        order=self.args.order,
                        num_inference_step=self.config.num_inference_steps,
                        content_encoder_downsample_size=self.args.content_encoder_downsample_size,
                        t_start=self.args.t_start,
                        t_end=self.args.t_end,
                        algorithm_type=self.args.algorithm_type,
                        enable_style_transform=self.config.enable_style_transform,
                    )
                
                return images
                
        except Exception as e:
            logger.error(f"Error in batch image generation: {e}")
            return None
    
    def _save_generated_image(
        self,
        char: str,
        style_name: str,
        style_dir: str,
        image: Image.Image,
        content_path: str
    ) -> None:
        """Save generated image and update metadata"""
        # Generate filename
        target_filename = get_target_filename(char, style_name)
        target_path = os.path.join(style_dir, target_filename)
        
        # Save image
        image.save(target_path)
        
        # Create generation record
        content_filename = os.path.basename(content_path)
        content_hash = compute_file_hash(char, "", self.font_manager.get_font_names()[0])
        target_hash = compute_file_hash(char, style_name, self.font_manager.get_font_names()[0])
        
        generation_record = {
            'character': char,
            'char_code': f"U+{ord(char):04X}",
            'style': style_name,
            'font': self.font_manager.get_font_names()[0],
            'content_image_path': f"ContentImage/{content_filename}",
            'target_image_path': f"TargetImage/{style_name}/{target_filename}",
            'content_hash': content_hash,
            'target_hash': target_hash,
            'content_filename': content_filename,
            'target_filename': target_filename,
        }
        
        # Add to tracker and results
        self.generation_tracker.add_generation(generation_record)
        self.results['generations'].append(generation_record)
    
    def _save_progress(self) -> None:
        """Save current progress to checkpoint"""
        if self.generation_tracker:
            self.generation_tracker.save_checkpoint(self.args.output_dir, self.results)
    
    def evaluate(self, ground_truth_dir: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate generated images against ground truth"""
        if not ground_truth_dir or not os.path.exists(ground_truth_dir):
            logger.warning("No ground truth directory provided, skipping evaluation")
            return self.results
        
        logger.info("Starting evaluation...")
        
        lpips_scores = []
        ssim_scores = []
        evaluated = 0
        missing = 0
        
        for gen in get_hf_bar(self.results['generations'], desc="Evaluating"):
            try:
                # Get paths
                target_path = os.path.join(self.args.output_dir, gen['target_image_path'])
                gt_path = self._find_ground_truth_path(gen, ground_truth_dir)
                
                if not gt_path:
                    missing += 1
                    continue
                
                # Compute metrics
                if LPIPS_AVAILABLE:
                    lpips_score = self.evaluator.compute_lpips(target_path, gt_path)
                    if lpips_score >= 0:
                        lpips_scores.append(lpips_score)
                        gen['lpips'] = lpips_score
                
                if SSIM_AVAILABLE:
                    ssim_score = self.evaluator.compute_ssim(target_path, gt_path)
                    if ssim_score >= 0:
                        ssim_scores.append(ssim_score)
                        gen['ssim'] = ssim_score
                
                evaluated += 1
                
            except Exception as e:
                logger.error(f"Error evaluating {gen.get('character', '?')}: {e}")
        
        # Update metrics
        if lpips_scores:
            self.results['metrics']['lpips'] = {
                'mean': float(np.mean(lpips_scores)),
                'std': float(np.std(lpips_scores)),
                'min': float(np.min(lpips_scores)),
                'max': float(np.max(lpips_scores)),
                'median': float(np.median(lpips_scores)),
                'samples': len(lpips_scores),
            }
        
        if ssim_scores:
            self.results['metrics']['ssim'] = {
                'mean': float(np.mean(ssim_scores)),
                'std': float(np.std(ssim_scores)),
                'min': float(np.min(ssim_scores)),
                'max': float(np.max(ssim_scores)),
                'median': float(np.median(ssim_scores)),
                'samples': len(ssim_scores),
            }
        
        logger.info(f"Evaluation complete: {evaluated} evaluated, {missing} missing ground truth")
        return self.results
    
    def _find_ground_truth_path(
        self,
        generation: Dict[str, Any],
        ground_truth_dir: str
    ) -> Optional[str]:
        """Find ground truth image path"""
        char = generation['character']
        style = generation['style']
        
        # Try different possible locations
        possible_paths = [
            os.path.join(ground_truth_dir, "TargetImage", style, generation['target_filename']),
            os.path.join(ground_truth_dir, style, generation['target_filename']),
            os.path.join(ground_truth_dir, generation['target_filename']),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def log_to_wandb(self) -> None:
        """Log results to Weights & Biases"""
        if not WANDB_AVAILABLE or not self.args.use_wandb:
            return
        
        try:
            wandb.init(
                project=self.args.wandb_project,
                name=self.args.wandb_run_name or f"batch_{int(time.time())}",
                config=asdict(self.config),
            )
            
            # Log metrics
            metrics = self.results.get('metrics', {})
            wandb.log({
                'generations': len(self.results['generations']),
                **{f'metrics/{k}': v for k, v in metrics.items() if isinstance(v, (int, float))},
            })
            
            # Log sample images
            sample_generations = self.results['generations'][:min(20, len(self.results['generations']))]
            sample_images = []
            
            for gen in sample_generations:
                img_path = os.path.join(self.args.output_dir, gen['target_image_path'])
                if os.path.exists(img_path):
                    with self.resource_manager.open_image(img_path) as img:
                        sample_images.append(wandb.Image(
                            img,
                            caption=f"{gen['character']} - {gen['style']}"
                        ))
            
            if sample_images:
                wandb.log({'sample_images': sample_images})
            
            wandb.finish()
            logger.info("Logged results to Weights & Biases")
            
        except Exception as e:
            logger.error(f"Failed to log to WandB: {e}")
    
    def cleanup(self) -> None:
        """Clean up all resources"""
        if self.font_manager:
            self.font_manager.cleanup()
        if self.resource_manager:
            self.resource_manager.cleanup_all()
        if self.generation_tracker:
            self._save_progress()


def load_characters(
    characters_arg: str,
    start_line: int = 1,
    end_line: Optional[int] = None
) -> List[str]:
    """Load characters from file or comma-separated string"""
    if os.path.isfile(characters_arg):
        with open(characters_arg, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Apply line range
        start_idx = max(0, start_line - 1)
        end_idx = len(lines) if end_line is None else min(len(lines), end_line)
        
        if start_idx >= end_idx:
            raise ValueError(f"Invalid line range: {start_line}-{end_line}")
        
        characters = []
        for line_num in range(start_idx, end_idx):
            line = lines[line_num].strip()
            if line and len(line) == 1:
                characters.append(line)
            elif line:
                logger.warning(f"Line {line_num+1} contains {len(line)} characters: '{line}'")
        
    else:
        characters = [c.strip() for c in characters_arg.split(',') if c.strip()]
    
    if not characters:
        raise ValueError("No valid characters loaded")
    
    logger.info(f"Loaded {len(characters)} characters")
    return characters


def load_style_images(style_images_arg: str) -> List[Tuple[str, str]]:
    """Load style image paths with names"""
    import glob
    
    image_paths = []
    
    # Determine input type
    if os.path.isdir(style_images_arg):
        # Directory
        for file in os.listdir(style_images_arg):
            if Path(file).suffix.lower() in VALID_IMAGE_EXTENSIONS:
                path = os.path.join(style_images_arg, file)
                if os.path.isfile(path):
                    image_paths.append(path)
    
    elif "*" in style_images_arg:
        # Glob pattern
        image_paths = glob.glob(style_images_arg, recursive=True)
    
    else:
        # Comma-separated list or single file
        paths = [p.strip() for p in style_images_arg.split(',') if p.strip()]
        for path in paths:
            if os.path.isfile(path) and Path(path).suffix.lower() in VALID_IMAGE_EXTENSIONS:
                image_paths.append(path)
    
    # Validate and extract names
    style_images = []
    for path in sorted(image_paths):
        if os.path.isfile(path):
            style_name = Path(path).stem
            style_images.append((path, style_name))
    
    if not style_images:
        raise ValueError(f"No valid style images found: {style_images_arg}")
    
    logger.info(f"Loaded {len(style_images)} style images")
    return style_images


def parse_args() -> Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="FontDiffuser Batch Generation")
    
    # Input/Output
    parser.add_argument("--characters", type=str, required=True,
                       help="Characters file or comma-separated list")
    parser.add_argument("--style_images", type=str, required=True,
                       help="Style images directory, glob, or comma-separated list")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--ground_truth_dir", type=str, default=None,
                       help="Ground truth directory for evaluation")
    
    # Model
    parser.add_argument("--ckpt_dir", type=str, required=True,
                       help="Model checkpoint directory")
    parser.add_argument("--ttf_path", type=str, required=True,
                       help="Font file or directory")
    
    # Generation parameters
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for generation")
    parser.add_argument("--num_inference_steps", type=int, default=15,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Optimizations
    parser.add_argument("--fp16", action="store_true",
                       help="Use FP16 precision")
    parser.add_argument("--compile", action="store_true",
                       help="Compile model with torch.compile")
    parser.add_argument("--channels_last", action="store_true",
                       help="Use channels last memory format")
    parser.add_argument("--enable_xformers", action="store_true",
                       help="Enable xformers optimizations")
    parser.add_argument("--enable_attention_slicing", action="store_true",
                       help="Enable attention slicing")
    
    # Features
    parser.add_argument("--enable_style_transform", action="store_true",
                       help="Enable style transformation")
    parser.add_argument("--save_interval", type=int, default=10,
                       help="Save checkpoint every N styles")
    
    # Evaluation
    parser.add_argument("--evaluate", action="store_true", default=True,
                       help="Evaluate generated images")
    parser.add_argument("--compute_fid", action="store_true",
                       help="Compute FID score")
    
    # WandB
    parser.add_argument("--use_wandb", action="store_true", default=True,
                       help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="fontdiffuser",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")
    
    # Dataset
    parser.add_argument("--dataset_split", type=str, default="train",
                       help="Dataset split name")
    parser.add_argument("--start_line", type=int, default=1,
                       help="Start line in character file (1-indexed)")
    parser.add_argument("--end_line", type=int, default=None,
                       help="End line in character file")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    return parser.parse_args()


def main() -> None:
    """Main entry point"""
    args = parse_args()
    generator = None
    
    try:
        logger.info("=" * 60)
        logger.info("FontDiffuser Batch Generation")
        logger.info("=" * 60)
        
        # Load input data
        characters = load_characters(
            args.characters,
            args.start_line,
            args.end_line
        )
        style_images = load_style_images(args.style_images)
        
        # Setup and run generator
        generator = BatchGenerator(args)
        generator.setup()
        
        results = generator.generate_batch(characters, style_images)
        
        # Evaluate if requested
        if args.evaluate and args.ground_truth_dir:
            results = generator.evaluate(args.ground_truth_dir)
        
        # Log to WandB
        if args.use_wandb:
            generator.log_to_wandb()
        
        # Save final results
        if generator.generation_tracker:
            generator.generation_tracker.save_checkpoint(args.output_dir, results)
        
        logger.info("Generation completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\nGeneration interrupted by user")
        if generator:
            logger.info("Saving current progress...")
            generator.cleanup()
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        logger.error(traceback.format_exc())
        if generator:
            logger.info("Attempting to save current state...")
            try:
                generator.cleanup()
            except:
                pass
        sys.exit(1)
        
    finally:
        if generator:
            generator.cleanup()


if __name__ == "__main__":
    main()