import torch


class CollateFN(object):
    def __init__(self):
        pass

    def __call__(self, batch):
        batched_data = {}

        for k in batch[0].keys():
            batch_key_data = [ele[k] for ele in batch]
            
            if isinstance(batch_key_data[0], torch.Tensor):
                # ✅ FIX: Handle variable-size tensors (like neg_images)
                # Check if all tensors have the same shape
                first_shape = batch_key_data[0].shape
                
                if all(tensor.shape == first_shape for tensor in batch_key_data):
                    # All same shape - use stack
                    batched_data[k] = torch.stack(batch_key_data)
                else:
                    # Variable shapes - use a list (for neg_images which is (num_neg, C, H, W))
                    # This happens when num_neg varies per sample
                    batched_data[k] = batch_key_data
            else:
                batched_data[k] = batch_key_data

        return batched_data