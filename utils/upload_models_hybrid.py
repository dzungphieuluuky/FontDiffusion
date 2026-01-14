"""
Convert PyTorch .pth weights to SafeTensors format and upload to Hugging Face Hub.
Unified version combining simple and parallel approaches with backward compatibility.
"""

import os
import sys
from pathlib import Path
import argparse
import concurrent.futures
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

from safetensors.torch import save_file
from huggingface_hub import HfApi, create_repo, login
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utilities import (
    load_model_checkpoint, 
    save_model_checkpoint
)

logger = logging.getLogger("ModelsUploader")

# Default FontDiffusion weight files
DEFAULT_FILES = [
    "content_encoder.pth",
    "content_encoder.safetensors",
    "style_encoder.pth",
    "style_encoder.safetensors",
    "unet.pth",
    "unet.safetensors",
    "total_model.pth",
    "total_model.safetensors",
    "scr.pth",
    "scr.safetensors",
]


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments with backward compatibility."""
    parser = argparse.ArgumentParser(
        description="Convert PyTorch .pth weights to SafeTensors format and upload to HF Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple usage (sequential)
  python upload_models.py --weights_dir "ckpt" --repo_id "username/font-diffusion-weights"
  
  # Parallel processing
  python upload_models.py --weights_dir "ckpt" --repo_id "username/font-diffusion-weights" --parallel 4
  
  # Convert only, no upload
  python upload_models.py --weights_dir "ckpt" --no-upload
  
  # Upload only, skip conversion
  python upload_models.py --weights_dir "ckpt" --repo_id "username/font-diffusion-weights" --skip-conversion
  
  # Specific files only
  python upload_models.py --weights_dir "ckpt" --files "unet.pth" "style_encoder.pth"
  
  # Recursive search in subfolders
  python upload_models.py --weights_dir "outputs/FontDiffuser" --recursive
        """,
    )
    
    # Core arguments
    parser.add_argument(
        "--weights_dir",
        type=str,
        required=True,
        default="outputs/FontDiffuser",
        help="Directory with .pth/.safetensors files",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="dzungpham/font-diffusion-weights",
        help="Hugging Face repository ID",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (or set HF_TOKEN env var)",
    )
    
    # File selection
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help="Specific files to process (default: all standard FontDiffusion weights)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recursively search for files in subdirectories",
    )
    
    # Processing mode
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel workers (1 = sequential, >1 = parallel)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Number of files to process in each batch (parallel mode only)",
    )
    
    # Repository options
    parser.add_argument(
        "--repo_type",
        type=str,
        default="model",
        choices=["model", "dataset", "space"],
        help="Repository type (default: model)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Make repository private",
    )
    
    # Control flags
    parser.add_argument(
        "--no_upload",
        action="store_true",
        default=False,
        help="Convert only, do not upload",
    )
    parser.add_argument(
        "--skip_conversion",
        action="store_true",
        default=False,
        help="Skip conversion, only upload",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Skip files that already exist in the repository",
    )
    
    # Metadata
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Add converted safetensors and original pth weights",
        help="Commit message for upload",
    )
    
    # Output control
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose output with detailed progress",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Minimal output (overrides verbose)",
    )
    
    return parser.parse_args()


def get_token(token_arg: Optional[str]) -> Optional[str]:
    """Get Hugging Face token from argument, env var, or cache."""
    if token_arg:
        return token_arg
    
    # Check environment variables
    token_env = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token_env:
        logger.info("Using HF token from environment variable")
        return token_env
    
    # Check cache
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            logger.info("Using HF token from cache")
            return token
    except Exception:
        pass
    
    return None


def collect_files_recursive(weights_dir: Path, file_patterns: List[str], recursive: bool) -> List[Path]:
    """Collect files recursively or from top-level directory."""
    all_files = []
    
    if recursive:
        # Recursive search
        for pattern in file_patterns:
            for file_path in weights_dir.rglob(pattern):
                if file_path.is_file():
                    all_files.append(file_path)
    else:
        # Top-level only
        for pattern in file_patterns:
            for file_path in weights_dir.glob(pattern):
                if file_path.is_file():
                    all_files.append(file_path)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for f in all_files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)
    
    return unique_files


def validate_inputs(args: argparse.Namespace) -> Dict[str, Any]:
    """Validate inputs and collect file information."""
    weights_dir = Path(args.weights_dir)
    
    if not weights_dir.exists():
        raise ValueError(f"Weights directory not found: {weights_dir}")
    
    if not weights_dir.is_dir():
        raise ValueError(f"Not a directory: {weights_dir}")
    
    # Collect files
    all_files = []
    pth_files = []
    safe_files = []
    
    # Collect based on file patterns
    file_patterns = args.files.copy()
    
    # If using default files, also include any .pth/.safetensors files
    if not args.files or len(args.files) == len(DEFAULT_FILES):
        file_patterns.extend(["*.pth", "*.safetensors"])
    
    all_files = collect_files_recursive(weights_dir, file_patterns, args.recursive)
    
    # Categorize files
    for file_path in all_files:
        if file_path.suffix == '.pth':
            pth_files.append(file_path)
        elif file_path.suffix == '.safetensors':
            safe_files.append(file_path)
    
    # Calculate total size
    total_size_mb = sum(f.stat().st_size for f in all_files) / (1024 * 1024)
    
    # Validate upload requirements
    if not args.no_upload:
        if not args.repo_id:
            raise ValueError("--repo_id is required when uploading (use --no-upload to skip)")
        
        token = get_token(args.token)
        if not token:
            raise ValueError("HF token not found! Provide via --token or set HF_TOKEN env var")
    
    # Print file discovery summary
    if args.verbose and not args.quiet:
        print(f"\n🔍 File discovery:")
        print(f"   Search mode: {'Recursive' if args.recursive else 'Top-level only'}")
        print(f"   Patterns: {file_patterns}")
        print(f"   Found {len(all_files)} total files")
        if args.recursive:
            # Show directory structure
            dirs = set()
            for f in all_files:
                dirs.add(str(f.parent.relative_to(weights_dir)))
            if dirs:
                print(f"   Directories scanned: {sorted(dirs)}")
    
    return {
        'weights_dir': weights_dir,
        'all_files': all_files,
        'pth_files': pth_files,
        'safe_files': safe_files,
        'total_size_mb': total_size_mb,
    }


def convert_single_file(pth_path: Path, verbose: bool = False) -> Optional[Dict[str, Any]]:
    """Convert a single .pth file to .safetensors."""
    try:
        safe_path = pth_path.with_suffix('.safetensors')
        
        # Skip if already exists
        if safe_path.exists():
            if verbose:
                print(f"⚠ {pth_path.name}: Safetensors file already exists, skipping")
            return None
        
        # Load checkpoint
        state_dict = load_model_checkpoint(str(pth_path))
        
        if not isinstance(state_dict, dict):
            logger.error(f"Expected dict, got {type(state_dict)} for {pth_path.name}")
            return None
        
        # Save as safetensors
        save_model_checkpoint(state_dict, str(safe_path))
        
        # Calculate statistics
        pth_size_mb = pth_path.stat().st_size / (1024 * 1024)
        safe_size_mb = safe_path.stat().st_size / (1024 * 1024)
        compression = ((pth_size_mb - safe_size_mb) / pth_size_mb * 100) if pth_size_mb > 0 else 0
        
        if verbose:
            print(f"✓ {pth_path.name}: {pth_size_mb:.2f}MB → {safe_size_mb:.2f}MB ({compression:.1f}% compression)")
        
        return {
            'filename': pth_path.name,
            'pth_size_mb': pth_size_mb,
            'safe_size_mb': safe_size_mb,
            'compression': compression,
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Failed to convert {pth_path.name}: {str(e)}")
        return {'filename': pth_path.name, 'success': False, 'error': str(e)}


def convert_files_parallel(pth_files: List[Path], num_workers: int, verbose: bool) -> Dict[str, Any]:
    """Convert files in parallel."""
    results = {
        'converted': 0,
        'failed': 0,
        'skipped': 0,
        'total_size_mb': 0.0,
        'details': []
    }
    
    if not pth_files:
        return results
    
    print(f"Converting {len(pth_files)} .pth files with {num_workers} workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all conversion tasks
        future_to_file = {
            executor.submit(convert_single_file, pth_path, verbose): pth_path
            for pth_path in pth_files
        }
        
        # Process results
        for future in tqdm(
            concurrent.futures.as_completed(future_to_file),
            total=len(future_to_file),
            desc="Converting files",
            unit="file",
            disable=not verbose
        ):
            pth_path = future_to_file[future]
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                results['details'].append(result)
                
                if result is None:
                    results['skipped'] += 1
                elif result.get('success'):
                    results['converted'] += 1
                    results['total_size_mb'] += result.get('safe_size_mb', 0)
                else:
                    results['failed'] += 1
                    
            except concurrent.futures.TimeoutError:
                logger.error(f"Conversion timeout for {pth_path.name}")
                results['failed'] += 1
            except Exception as e:
                logger.error(f"Error processing {pth_path.name}: {e}")
                results['failed'] += 1
    
    return results


def convert_files_sequential(pth_files: List[Path], verbose: bool) -> Dict[str, Any]:
    """Convert files sequentially."""
    results = {
        'converted': 0,
        'failed': 0,
        'skipped': 0,
        'total_size_mb': 0.0,
        'details': []
    }
    
    if not pth_files:
        return results
    
    print(f"Converting {len(pth_files)} .pth files sequentially...")
    
    for pth_path in tqdm(pth_files, desc="Converting files", unit="file", disable=not verbose):
        try:
            result = convert_single_file(pth_path, verbose)
            results['details'].append(result)
            
            if result is None:
                results['skipped'] += 1
            elif result.get('success'):
                results['converted'] += 1
                results['total_size_mb'] += result.get('safe_size_mb', 0)
            else:
                results['failed'] += 1
                
        except Exception as e:
            logger.error(f"Error processing {pth_path.name}: {e}")
            results['failed'] += 1
    
    return results


def upload_files(all_files: List[Path], args: argparse.Namespace, file_info: Dict[str, Any]) -> bool:
    """Upload files to Hugging Face Hub."""
    if args.no_upload:
        print("⊘ Skipping upload (--no-upload)")
        return True
    
    print("\n" + "=" * 70)
    print("UPLOADING TO HUGGING FACE HUB")
    print("=" * 70)
    
    # Get token and login
    token = get_token(args.token)
    if not token:
        raise ValueError("HF token required for upload")
    
    login(token=token)
    api = HfApi()
    
    # Create repository
    print(f"📤 Creating/verifying repository: {args.repo_id}")
    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            exist_ok=True,
            private=args.private,
            token=token,
        )
        print("✓ Repository ready")
    except Exception as e:
        print(f"⚠ Repository creation/verification: {e}")
        # Continue anyway - repository might already exist
    
    # Upload files
    print(f"\n📤 Uploading {len(all_files)} files...")
    
    try:
        api.upload_folder(
            folder_path=str(file_info['weights_dir']),
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            token=token,
            commit_message=args.commit_message,
        )
        
        print(f"\n✓ Upload successful!")
        print(f"  Repository URL: https://huggingface.co/{args.repo_type}s/{args.repo_id}")
        return True
        
    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("FONTDIFFUSER WEIGHTS CONVERTER & UPLOADER")
    print("=" * 70)
    
    args = parse_arguments()
    start_time = time.time()
    
    try:
        # Parse arguments
        if args.verbose and not args.quiet:
            print("\n📋 Configuration:")
            for key, value in vars(args).items():
                if key == 'token' and value:
                    print(f"  {key}: {'*' * 20}")
                else:
                    print(f"  {key}: {value}")
        
        # Validate and collect files
        file_info = validate_inputs(args)
        if not file_info['all_files']:
            print("⚠️ No files found matching patterns!")
            return 1        
        print(f"\n📁 Found {len(file_info['all_files'])} files ({file_info['total_size_mb']:.2f} MB)")
        print(f"  • {len(file_info['pth_files'])} .pth files")
        print(f"  • {len(file_info['safe_files'])} .safetensors files")
        
        # Convert files
        if not args.skip_conversion and file_info['pth_files']:
            print("\n" + "=" * 70)
            print("CONVERTING .pth TO .safetensors")
            print("=" * 70)
            
            if args.parallel > 1:
                args.parallel = min(args.parallel, os.cpu_count())
                results = convert_files_parallel(
                    file_info['pth_files'], 
                    args.parallel, 
                    args.verbose and not args.quiet
                )
            else:
                results = convert_files_sequential(
                    file_info['pth_files'], 
                    args.verbose and not args.quiet
                )
            
            print(f"\n📊 Conversion complete:")
            print(f"  • Converted: {results['converted']}")
            print(f"  • Failed: {results['failed']}")
            print(f"  • Skipped: {results['skipped']}")
            
            if results['failed'] > 0 and not args.skip_conversion:
                print("⚠ Some conversions failed")
        
        # Upload files
        success = upload_files(file_info['all_files'], args, file_info)
        
        # Final summary
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print("✅ PROCESS COMPLETE")
        print("=" * 70)
        print(f"Total time: {total_time:.2f} seconds")
        
        if success and not args.no_upload:
            print(f"\n📦 Your weights are now available at:")
            print(f"   https://huggingface.co/{args.repo_type}s/{args.repo_id}")
            print(f"\n💾 Load them with:")
            print(f"   from safetensors.torch import load_file")
            print(f"   from huggingface_hub import hf_hub_download")
            print(f"   ")
            print(f"   # Download and load")
            print(f"   model_path = hf_hub_download(repo_id='{args.repo_id}', filename='model.safetensors')")
            print(f"   state_dict = load_file(model_path)")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        if args.verbose and not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())