"""
Enhanced collate function for FontDiffuserWithFST.
Handles both original and FST dataset modes.
"""

import torch
import logging
from typing import List, Dict, Any


class CollateFN(object):
    """
    Enhanced collate function supporting both original and FST modes.
    
    Handles:
    - Standard tensors (content_image, style_image, target_image)
    - FST tensors (style_source_image)
    - SCR tensors (neg_images with variable dimensions)
    - Variable-sized tensors with automatic resizing
    """
    
    def __init__(self, verbose: bool = False):
        """
        Args:
            verbose: If True, log detailed information about batching
        """
        self.verbose = verbose
        
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a list of samples into a batch.
        
        Args:
            batch: List of dictionaries from dataset __getitem__
            
        Returns:
            Dictionary with batched tensors
        """
        if not batch:
            return {}
        
        batched_data = {}
        
        for key in batch[0].keys():
            batch_key_data = [sample[key] for sample in batch]
            
            if isinstance(batch_key_data[0], torch.Tensor):
                batched_data[key] = self._collate_tensors(key, batch_key_data)
            else:
                # Non-tensor data (e.g., paths, metadata)
                batched_data[key] = batch_key_data
        
        return batched_data
    
    def _collate_tensors(
        self, 
        key: str, 
        tensors: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Collate a list of tensors, handling variable shapes.
        
        Args:
            key: Name of the tensor field (for logging)
            tensors: List of tensors to collate
            
        Returns:
            Batched tensor
        """
        first_shape = tensors[0].shape
        
        # Special handling for neg_images (already batched per sample)
        if key == "neg_images":
            return self._collate_neg_images(tensors)
        
        # Check if all tensors have the same shape
        if all(t.shape == first_shape for t in tensors):
            # All same shape - standard stacking
            return torch.stack(tensors)
        
        # Variable shapes detected - try to standardize
        if self.verbose:
            logging.warning(
                f"Variable shapes detected for key '{key}': "
                f"{[tuple(t.shape) for t in tensors]}"
            )
        
        return self._standardize_and_stack(key, tensors, first_shape)
    
    def _collate_neg_images(
        self, 
        neg_image_tensors: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Collate negative images from SCR.
        
        Each sample has shape (num_neg, C, H, W).
        Output shape: (batch_size, num_neg, C, H, W)
        
        Args:
            neg_image_tensors: List of negative image tensors
            
        Returns:
            Batched negative images
        """
        # Check if all have same num_neg
        first_shape = neg_image_tensors[0].shape
        
        if all(t.shape == first_shape for t in neg_image_tensors):
            # Standard case: stack directly
            return torch.stack(neg_image_tensors)
        
        # Variable num_neg - need to handle carefully
        logging.warning(
            f"Variable number of negative samples detected: "
            f"{[t.shape[0] for t in neg_image_tensors]}"
        )
        
        # Find max num_neg
        max_num_neg = max(t.shape[0] for t in neg_image_tensors)
        
        # Pad smaller batches by repeating last negative sample
        padded_negs = []
        for neg_tensor in neg_image_tensors:
            if neg_tensor.shape[0] < max_num_neg:
                # Pad by repeating the last sample
                num_to_pad = max_num_neg - neg_tensor.shape[0]
                padding = neg_tensor[-1:].repeat(num_to_pad, 1, 1, 1)
                neg_tensor = torch.cat([neg_tensor, padding], dim=0)
            padded_negs.append(neg_tensor)
        
        return torch.stack(padded_negs)
    
    def _standardize_and_stack(
        self,
        key: str,
        tensors: List[torch.Tensor],
        target_shape: torch.Size
    ) -> torch.Tensor:
        """
        Standardize tensor shapes and stack them.
        
        Args:
            key: Tensor field name
            tensors: List of tensors to standardize
            target_shape: Target shape to resize to
            
        Returns:
            Stacked tensor with standardized shapes
        """
        try:
            from torchvision.transforms import functional as TF
            
            standardized = []
            for tensor in tensors:
                if tensor.shape != target_shape:
                    # For images (C, H, W), resize spatial dimensions
                    if len(tensor.shape) == 3 and len(target_shape) == 3:
                        tensor = TF.resize(
                            tensor,
                            (target_shape[-2], target_shape[-1]),
                            interpolation=TF.InterpolationMode.BILINEAR,
                            antialias=True
                        )
                    elif len(tensor.shape) == 4 and len(target_shape) == 4:
                        # For batched images (N, C, H, W)
                        resized = []
                        for i in range(tensor.shape[0]):
                            img = TF.resize(
                                tensor[i],
                                (target_shape[-2], target_shape[-1]),
                                interpolation=TF.InterpolationMode.BILINEAR,
                                antialias=True
                            )
                            resized.append(img)
                        tensor = torch.stack(resized)
                    else:
                        # Fallback: warn and keep original
                        logging.warning(
                            f"Cannot resize tensor of shape {tensor.shape} "
                            f"to {target_shape} for key '{key}'"
                        )
                
                standardized.append(tensor)
            
            return torch.stack(standardized)
            
        except Exception as e:
            logging.error(
                f"Could not standardize shapes for '{key}': {e}. "
                f"Returning as list."
            )
            # Fallback: keep as list
            return tensors


class CollateFNDebug(CollateFN):
    """Debug version of CollateFN with detailed logging."""
    
    def __init__(self):
        super().__init__(verbose=True)
        self.call_count = 0
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate with debug logging."""
        self.call_count += 1
        
        print(f"\n{'='*80}")
        print(f"CollateFN Call #{self.call_count}")
        print(f"{'='*80}")
        print(f"Batch size: {len(batch)}")
        
        if batch:
            print(f"Keys in sample: {list(batch[0].keys())}")
            
            # Print shapes
            for key in batch[0].keys():
                if isinstance(batch[0][key], torch.Tensor):
                    shapes = [sample[key].shape for sample in batch]
                    print(f"  {key:25s}: {shapes}")
        
        # Call parent collate
        result = super().__call__(batch)
        
        # Print result shapes
        print(f"\nBatched tensor shapes:")
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key:25s}: {tuple(value.shape)}")
        
        print(f"{'='*80}\n")
        
        return result


def test_collate_fn():
    """Test the collate function with various scenarios."""
    print("Testing CollateFN...")
    
    # Test 1: Standard batch (original mode)
    print("\n" + "="*80)
    print("Test 1: Standard batch (original mode)")
    print("="*80)
    
    batch_original = [
        {
            'content_image': torch.randn(1, 96, 96),
            'style_image': torch.randn(1, 96, 96),
            'target_image': torch.randn(1, 128, 128),
            'nonorm_target_image': torch.randn(1, 128, 128),
            'target_image_path': 'path/to/image1.jpg',
        },
        {
            'content_image': torch.randn(1, 96, 96),
            'style_image': torch.randn(1, 96, 96),
            'target_image': torch.randn(1, 128, 128),
            'nonorm_target_image': torch.randn(1, 128, 128),
            'target_image_path': 'path/to/image2.jpg',
        },
    ]
    
    collate_fn = CollateFNDebug()
    batched = collate_fn(batch_original)
    
    # Test 2: FST batch
    print("\n" + "="*80)
    print("Test 2: FST batch with style_source_image")
    print("="*80)
    
    batch_fst = [
        {
            'content_image': torch.randn(1, 96, 96),
            'style_image': torch.randn(1, 96, 96),
            'style_source_image': torch.randn(1, 96, 96),
            'target_image': torch.randn(1, 128, 128),
            'nonorm_target_image': torch.randn(1, 128, 128),
        },
        {
            'content_image': torch.randn(1, 96, 96),
            'style_image': torch.randn(1, 96, 96),
            'style_source_image': torch.randn(1, 96, 96),
            'target_image': torch.randn(1, 128, 128),
            'nonorm_target_image': torch.randn(1, 128, 128),
        },
    ]
    
    batched_fst = collate_fn(batch_fst)
    
    # Test 3: SCR batch with neg_images
    print("\n" + "="*80)
    print("Test 3: SCR batch with neg_images")
    print("="*80)
    
    batch_scr = [
        {
            'content_image': torch.randn(1, 96, 96),
            'style_image': torch.randn(1, 96, 96),
            'target_image': torch.randn(1, 128, 128),
            'nonorm_target_image': torch.randn(1, 128, 128),
            'neg_images': torch.randn(3, 1, 128, 128),  # 3 negative samples
        },
        {
            'content_image': torch.randn(1, 96, 96),
            'style_image': torch.randn(1, 96, 96),
            'target_image': torch.randn(1, 128, 128),
            'nonorm_target_image': torch.randn(1, 128, 128),
            'neg_images': torch.randn(3, 1, 128, 128),
        },
    ]
    
    batched_scr = collate_fn(batch_scr)
    
    # Test 4: Variable shapes (should trigger warning)
    print("\n" + "="*80)
    print("Test 4: Variable shapes (should warn and resize)")
    print("="*80)
    
    batch_variable = [
        {
            'content_image': torch.randn(1, 96, 96),
            'style_image': torch.randn(1, 96, 96),
            'target_image': torch.randn(1, 128, 128),
        },
        {
            'content_image': torch.randn(1, 96, 96),
            'style_image': torch.randn(1, 100, 100),  # Different size
            'target_image': torch.randn(1, 128, 128),
        },
    ]
    
    batched_variable = collate_fn(batch_variable)
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)


if __name__ == "__main__":
    test_collate_fn()