#!/usr/bin/env python
"""
Quick validation script to verify MRL configuration compatibility.
Run this before training to catch dimension errors early.
"""

import argparse
import sys


def validate_mrl_config(nesting_dims_str, freq_radii_str, embedding_dim=512):
    """Validate MRL configuration and print results."""
    
    # Parse dimensions
    nesting_dims = [int(x.strip()) for x in nesting_dims_str.split(",")]
    freq_radii = [float(x.strip()) for x in freq_radii_str.split(",")]
    
    print("\n" + "="*70)
    print("MRL CONFIGURATION VALIDATION")
    print("="*70)
    
    print(f"\nInput Configuration:")
    print(f"  nesting_dims: {nesting_dims}")
    print(f"  freq_radii:   {freq_radii}")
    print(f"  embedding_dim: {embedding_dim}")
    
    errors = []
    
    # Check 1: Length matching
    expected_radii_count = len(nesting_dims) - 1
    if len(freq_radii) != expected_radii_count:
        errors.append(
            f"❌ Length mismatch: len(freq_radii)={len(freq_radii)} "
            f"must equal len(nesting_dims)-1={expected_radii_count}"
        )
    else:
        print(f"\n✅ Length matching: {len(freq_radii)} == {expected_radii_count}")
    
    # Check 2: Ascending order for nesting_dims
    if nesting_dims != sorted(nesting_dims):
        errors.append(f"❌ nesting_dims not in ascending order: {nesting_dims}")
    else:
        print(f"✅ nesting_dims in ascending order")
    
    # Check 3: Ascending order for freq_radii
    if freq_radii != sorted(freq_radii):
        errors.append(f"❌ freq_radii not in ascending order: {freq_radii}")
    else:
        print(f"✅ freq_radii in ascending order")
    
    # Check 4: Embedding dimension constraint
    exceed = [d for d in nesting_dims if d > embedding_dim]
    if exceed:
        errors.append(
            f"❌ Some nesting_dims exceed embedding_dim={embedding_dim}: {exceed}"
        )
    else:
        print(f"✅ All nesting_dims <= embedding_dim ({embedding_dim})")
    
    # Check 5: Frequency radii range
    out_of_range = [r for r in freq_radii if r < 0 or r > 1]
    if out_of_range:
        errors.append(
            f"❌ Some freq_radii outside [0,1]: {out_of_range}"
        )
    else:
        print(f"✅ All freq_radii in range [0, 1]")
    
    # Print results
    print("\n" + "-"*70)
    if errors:
        print("VALIDATION FAILED:")
        for error in errors:
            print(f"  {error}")
        print("="*70 + "\n")
        return False
    else:
        print("✅ VALIDATION PASSED - Configuration is valid!")
        print("="*70 + "\n")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate MRL configuration before training"
    )
    parser.add_argument(
        "--mrl_nesting_dims",
        type=str,
        default="64,128,256,512",
        help="MRL nesting dimensions (comma-separated)"
    )
    parser.add_argument(
        "--mrl_freq_radii",
        type=str,
        default="0.1,0.3,0.5",
        help="MRL frequency radii (comma-separated)"
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=512,
        help="Content encoder embedding dimension"
    )
    
    args = parser.parse_args()
    
    is_valid = validate_mrl_config(
        args.mrl_nesting_dims,
        args.mrl_freq_radii,
        args.embedding_dim
    )
    
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
