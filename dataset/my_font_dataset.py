import hashlib
import random
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def compute_file_hash(char: str, style: str, font: str = "") -> str:
    """Compute deterministic hash for a (character, style, font) combination"""
    content = f"{char}_{style}_{font}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]


def get_content_filename(char: str, font: str = "") -> str:
    """
    Get content image filename
    Format: U+XXXX_[char]_hash.png or U+XXXX_hash.png
    """
    codepoint = f"U+{ord(char):04X}"
    hash_val = compute_file_hash(char, "", font)
    
    filesystem_unsafe = '<>:"/\\|?*'
    safe_char = char if char not in filesystem_unsafe else ""
    
    if safe_char:
        return f"{codepoint}_{safe_char}_{hash_val}.png"
    else:
        return f"{codepoint}_{hash_val}.png"


def get_target_filename(char: str, style: str, font: str = "") -> str:
    """
    Get target image filename
    Format: U+XXXX_[char]_style_hash.png or U+XXXX_style_hash.png
    """
    codepoint = f"U+{ord(char):04X}"
    hash_val = compute_file_hash(char, style, font)
    
    filesystem_unsafe = '<>:"/\\|?*'
    safe_char = char if char not in filesystem_unsafe else ""
    
    if safe_char:
        return f"{codepoint}_{safe_char}_{style}_{hash_val}.png"
    else:
        return f"{codepoint}_{style}_{hash_val}.png"


def get_nonorm_transform(resolution):
    """Get non-normalized transform for target image"""
    nonorm_transform = transforms.Compose(
        [
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
    )
    return nonorm_transform


class MyFontDataset(Dataset):
    """
    Font generation dataset
    ✅ Uses hash-based filenames (U+XXXX_[char]_style_hash.png)
    ✅ Loads from checkpoint.json as single source of truth
    ✅ Supports SCR (negative sampling)
    """

    def __init__(self, args, phase: str = "train", transforms=None, scr: bool = False):
        super().__init__()
        
        self.args = args
        self.root = args.data_root
        self.phase = phase
        self.scr = scr
        
        if self.scr:
            self.num_neg = args.num_neg

        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

        # Load checkpoint and build data structures
        self.target_images = []
        self.style_to_images = {}
        self.checkpoint_data = {}
        self.get_path()

    def get_path(self):
        """
        Load data from checkpoint.json and build index of all images
        ✅ Uses checkpoint as source of truth
        """
        data_root = Path(self.root) / self.phase
        checkpoint_path = data_root / "results_checkpoint.json"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with open(checkpoint_path, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        generations = checkpoint.get("generations", [])
        
        if not generations:
            raise ValueError(f"No generations in checkpoint: {checkpoint_path}")

        # Build target_images list and style_to_images dict from checkpoint
        for idx, gen in enumerate(generations):
            char = gen.get("character")
            style = gen.get("style")
            font = gen.get("font", "")

            if not char or not style:
                continue

            # Generate expected filename using hash
            target_filename = get_target_filename(char, style, font)
            target_image_path = f"{self.root}/{self.phase}/TargetImage/{style}/{target_filename}"

            # Verify file exists
            if not Path(target_image_path).exists():
                print(f"⚠️  File not found: {target_image_path}")
                continue

            # Store in target_images list
            self.target_images.append(target_image_path)

            # Store in style_to_images dict
            if style not in self.style_to_images:
                self.style_to_images[style] = []
            self.style_to_images[style].append(target_image_path)

            # Store checkpoint data for reference
            self.checkpoint_data[target_image_path] = gen

        print(f"✓ Loaded {len(self.target_images)} images from checkpoint")

    def __getitem__(self, index):
        """
        Load (content, style, target) triplet
        ✅ Handles hash-based filenames
        """
        target_image_path = self.target_images[index]
        gen = self.checkpoint_data[target_image_path]

        char = gen.get("character")
        style = gen.get("style")
        font = gen.get("font", "")

        # Generate content filename using hash
        content_filename = get_content_filename(char, font)
        content_image_path = f"{self.root}/{self.phase}/ContentImage/{content_filename}"

        # Read content image
        content_image = Image.open(content_image_path).convert("RGB")

        # Random sample used for style image (pick different target from same style)
        images_related_style = self.style_to_images[style].copy()
        images_related_style.remove(target_image_path)
        
        if not images_related_style:
            # If only one image per style, reuse the target as style image
            style_image_path = target_image_path
        else:
            style_image_path = random.choice(images_related_style)
        
        style_image = Image.open(style_image_path).convert("RGB")

        # Read target image
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        # Apply transforms
        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)

        sample = {
            "character": char,
            "style": style,
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image,
        }

        # Load negative samples if using SCR (Phase 2)
        if self.scr:
            style_list = list(self.style_to_images.keys())
            style_index = style_list.index(style)
            style_list.pop(style_index)

            if style_list:  # Only if other styles exist
                choose_neg_names = []
                
                for i in range(self.num_neg):
                    if not style_list:
                        break
                    
                    choose_style = random.choice(style_list)
                    choose_index = style_list.index(choose_style)
                    style_list.pop(choose_index)
                    
                    # Generate negative filename using hash
                    neg_filename = get_target_filename(char, choose_style, font)
                    choose_neg_name = f"{self.root}/{self.phase}/TargetImage/{choose_style}/{neg_filename}"
                    
                    if Path(choose_neg_name).exists():
                        choose_neg_names.append(choose_neg_name)

                # Load neg_images
                neg_images = None
                for i, neg_name in enumerate(choose_neg_names):
                    neg_image = Image.open(neg_name).convert("RGB")
                    if self.transforms is not None:
                        neg_image = self.transforms[2](neg_image)
                    
                    if neg_images is None:
                        neg_images = neg_image[None, :, :, :]
                    else:
                        neg_images = torch.cat(
                            [neg_images, neg_image[None, :, :, :]], dim=0
                        )

                if neg_images is not None:
                    sample["neg_images"] = neg_images

        return sample

    def __len__(self):
        return len(self.target_images)