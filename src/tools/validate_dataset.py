# validate_dataset.py
import argparse
from pathlib import Path
from src.dataset.font_dataset_fst import FontDataset, FontDatasetDebug


def validate_dataset(data_root):
    """Validate dataset structure and compatibility."""

    print(f"Validating dataset at: {data_root}")
    print("=" * 80)

    # Check directory structure
    data_path = Path(data_root)
    required_dirs = [
        data_path / "train" / "ContentImage",
        data_path / "train" / "TargetImage",
    ]

    print("\n1. Checking directory structure...")
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"   ✓ {dir_path}")
        else:
            print(f"   ✗ {dir_path} (MISSING)")
            return False

    # Try loading dataset
    print("\n2. Loading dataset in original mode...")
    try:
        data_root = str(data_root)

        class Args:
            resolution = 128
            num_neg = 3

        dataset = FontDataset(args=Args(), phase="train", use_fst=False)
        print(f"   ✓ Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"   ✗ Failed to load: {e}")
        return False

    # Check dataset structure
    print("\n3. Analyzing dataset structure...")
    debug = FontDatasetDebug(dataset)
    stats = debug.check_structure()

    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Number of styles: {stats['num_styles']}")
    print(f"   Number of unique contents: {stats['num_contents']}")

    # Check sample
    print("\n4. Checking sample format...")
    if len(dataset) > 0:
        try:
            sample = dataset[0]
            required_keys = ["content_image", "style_image", "target_image"]
            for key in required_keys:
                if key in sample:
                    print(f"   ✓ {key}: {sample[key].shape}")
                else:
                    print(f"   ✗ {key}: MISSING")
        except Exception as e:
            print(f"   ✗ Failed to load sample: {e}")
            return False

    # Test FST mode
    print("\n5. Testing FST mode...")
    try:
        dataset_fst = FontDataset(
            args=Args(), phase="train", use_fst=True, style_source_same_prob=0.5
        )

        if len(dataset_fst) > 0:
            sample_fst = dataset_fst[0]
            if "style_source_image" in sample_fst:
                print(f"   ✓ FST mode working")
                print(
                    f"   ✓ style_source_image: {sample_fst['style_source_image'].shape}"
                )
            else:
                print(f"   ✗ style_source_image not found")
    except Exception as e:
        print(f"   ✗ FST mode failed: {e}")
        return False

    # Final verdict
    print("\n" + "=" * 80)
    print("✓ Dataset validation PASSED")
    print("=" * 80)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    args = parser.parse_args()

    validate_dataset(args.data_root)
