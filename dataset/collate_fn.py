import torch
import logging
class CollateFN(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        batched_data = {}

        for k in batch[0].keys():
            batch_key_data = [ele[k] for ele in batch]
            
            if isinstance(batch_key_data[0], torch.Tensor):
                # ✅ FIX: Handle variable-size tensors safely
                first_shape = batch_key_data[0].shape
                
                # Check if all tensors have the same shape
                if all(tensor.shape == first_shape for tensor in batch_key_data):
                    # All same shape - use stack
                    batched_data[k] = torch.stack(batch_key_data)
                else:
                    # ✅ Variable shapes detected - log and try to resize
                    logging.warning(
                        f"Variable shapes detected for key '{k}': {[t.shape for t in batch_key_data]}"
                    )
                    
                    # Try to standardize shapes
                    try:
                        from torchvision.transforms import functional as TF
                        
                        # Resize all to first tensor's shape
                        resized = []
                        for tensor in batch_key_data:
                            if tensor.shape != first_shape:
                                tensor = TF.resize(
                                    tensor,
                                    (first_shape[-2], first_shape[-1]),
                                    interpolation=TF.InterpolationMode.LANCZOS
                                )
                            resized.append(tensor)
                        batched_data[k] = torch.stack(resized)
                    except Exception as e:
                        logging.error(f"Could not standardize shapes for '{k}': {e}")
                        # Fallback: keep as list
                        batched_data[k] = batch_key_data
            else:
                batched_data[k] = batch_key_data

        return batched_data