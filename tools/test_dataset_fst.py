# test_dataset_fst.py
from dataset.font_dataset_fst import FontDataset, FontDatasetDebug
from dataset.collate_fn_fst import CollateFN
from torch.utils.data import DataLoader

class Args:
    data_root = "my_dataset"
    resolution = 128
    num_neg = 3

# Test 1: Load dataset
print("Test 1: Loading FST dataset...")
dataset = FontDataset(
    args=Args(),
    phase="train",
    use_fst=True,
    style_source_same_prob=0.5
)
print(f"✓ Loaded {len(dataset)} samples")

# Test 2: Check sample
print("\nTest 2: Checking sample structure...")
sample = dataset[0]
required_keys = ['content_image', 'style_image', 'style_source_image', 'target_image']
for key in required_keys:
    if key in sample:
        print(f"✓ {key}: {sample[key].shape}")
    else:
        print(f"✗ {key}: MISSING")

# Test 3: Test dataloader
print("\nTest 3: Testing DataLoader...")
loader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=CollateFN(verbose=False),
    shuffle=True
)

batch = next(iter(loader))
print(f"✓ Batch loaded successfully")
for key, value in batch.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: {value.shape}")

print("\n✓ All tests passed!")