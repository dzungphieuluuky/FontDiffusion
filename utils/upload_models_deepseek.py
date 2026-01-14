"""
Convert PyTorch .pth weights to SafeTensors format and upload to Hugging Face Hub.
Optimized for performance with parallel processing and efficient I/O.
"""

import os
import sys
import argparse
import asyncio
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
import logging
import time
from dataclasses import dataclass
import threading
from queue import Queue

import torch
import safetensors
from huggingface_hub import (
    HfApi, 
    create_repo, 
    login, 
    upload_file,
    upload_folder,
    CommitOperationAdd,
)
from tqdm.asyncio import tqdm as async_tqdm
from tqdm.auto import tqdm

from logging_utils import setup_logging
from utilities import load_model_checkpoint, save_model_checkpoint, find_checkpoint

logger = setup_logging(level=logging.INFO, name="ModelsUploader")


@dataclass
class UploadStats:
    """Statistics for upload operations."""
    total_files: int = 0
    converted_files: int = 0
    uploaded_files: int = 0
    failed_files: int = 0
    total_size_mb: float = 0.0
    start_time: float = 0.0
    
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    def print_summary(self):
        """Print formatted statistics summary."""
        elapsed = self.elapsed_time()
        print("\n" + "="*70)
        print("UPLOAD STATISTICS SUMMARY")
        print("="*70)
        print(f"Total files processed:   {self.total_files}")
        print(f"Successfully converted:  {self.converted_files}")
        print(f"Successfully uploaded:   {self.uploaded_files}")
        print(f"Failed files:            {self.failed_files}")
        print(f"Total size:              {self.total_size_mb:.2f} MB")
        print(f"Elapsed time:            {elapsed:.2f} seconds")
        if self.uploaded_files > 0:
            avg_speed = self.total_size_mb / elapsed if elapsed > 0 else 0
            print(f"Average upload speed:    {avg_speed:.2f} MB/s")
        print("="*70)


class ParallelConverter:
    """Parallel converter for .pth to .safetensors files."""
    
    def __init__(self, num_workers: int = 4, verbose: bool = False):
        self.num_workers = num_workers
        self.verbose = verbose
        self.stats = UploadStats()
    
    def convert_file(self, pth_path: Path, safe_path: Path) -> Optional[Dict[str, Any]]:
        """Convert a single .pth file to .safetensors."""
        try:
            # Load checkpoint
            start_time = time.time()
            state_dict = load_model_checkpoint(str(pth_path))
            
            if not isinstance(state_dict, dict):
                logger.error(f"Expected dict, got {type(state_dict)} for {pth_path.name}")
                return None
            
            # Save as safetensors
            save_model_checkpoint(state_dict, str(safe_path))
            
            # Calculate statistics
            pth_size_mb = pth_path.stat().st_size / (1024 * 1024)
            safe_size_mb = safe_path.stat().st_size / (1024 * 1024)
            conversion_time = time.time() - start_time
            
            return {
                'filename': pth_path.name,
                'pth_size_mb': pth_size_mb,
                'safe_size_mb': safe_size_mb,
                'conversion_time': conversion_time,
                'compression': ((pth_size_mb - safe_size_mb) / pth_size_mb * 100) if pth_size_mb > 0 else 0,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to convert {pth_path.name}: {str(e)}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return {'filename': pth_path.name, 'success': False, 'error': str(e)}
    
    def convert_batch(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Convert a batch of files in parallel."""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all conversion tasks
            future_to_file = {
                executor.submit(self.convert_file, pth_path, pth_path.with_suffix('.safetensors')): 
                pth_path for pth_path in files if pth_path.suffix == '.pth'
            }
            
            # Process results as they complete
            for future in tqdm(
                concurrent.futures.as_completed(future_to_file),
                total=len(future_to_file),
                desc="Converting files",
                unit="file"
            ):
                pth_path = future_to_file[future]
                try:
                    result = future.result(timeout=300)  # 5 minute timeout
                    results.append(result)
                    
                    if result and result.get('success'):
                        self.stats.converted_files += 1
                        self.stats.total_size_mb += result.get('safe_size_mb', 0)
                        
                        if self.verbose:
                            print(f"✓ {pth_path.name}: {result.get('pth_size_mb', 0):.2f}MB → "
                                  f"{result.get('safe_size_mb', 0):.2f}MB "
                                  f"({result.get('compression', 0):.1f}% compression) "
                                  f"in {result.get('conversion_time', 0):.2f}s")
                    else:
                        self.stats.failed_files += 1
                        logger.warning(f"Failed to convert {pth_path.name}")
                        
                except concurrent.futures.TimeoutError:
                    logger.error(f"Conversion timeout for {pth_path.name}")
                    self.stats.failed_files += 1
                except Exception as e:
                    logger.error(f"Error processing {pth_path.name}: {e}")
                    self.stats.failed_files += 1
        
        return results


class ParallelUploader:
    """Parallel uploader for files to Hugging Face Hub."""
    
    def __init__(self, num_workers: int = 4, verbose: bool = False):
        self.num_workers = num_workers
        self.verbose = verbose
        self.api = HfApi()
        self.token = None
    
    def set_token(self, token: str):
        """Set Hugging Face token."""
        self.token = token
        login(token=token)
    
    def upload_single_file(self, local_path: Path, repo_id: str, path_in_repo: str) -> bool:
        """Upload a single file to Hugging Face Hub."""
        try:
            # Use upload_file for better performance with large files
            self.api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=self.token,
                commit_message=f"Add {local_path.name}",
            )
            
            file_size_mb = local_path.stat().st_size / (1024 * 1024)
            if self.verbose:
                print(f"✓ Uploaded {local_path.name} ({file_size_mb:.2f}MB)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {local_path.name}: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False
    
    def upload_batch(self, files: List[Path], repo_id: str, base_path: Path) -> Dict[str, Any]:
        """Upload a batch of files in parallel."""
        results = {
            'success': 0,
            'failed': 0,
            'total_size_mb': 0,
            'details': []
        }
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all upload tasks
            future_to_file = {}
            for file_path in files:
                # Calculate relative path for repository
                rel_path = file_path.relative_to(base_path)
                future = executor.submit(
                    self.upload_single_file, 
                    file_path, 
                    repo_id, 
                    str(rel_path)
                )
                future_to_file[future] = file_path
            
            # Process results as they complete
            for future in tqdm(
                concurrent.futures.as_completed(future_to_file),
                total=len(future_to_file),
                desc="Uploading files",
                unit="file"
            ):
                file_path = future_to_file[future]
                try:
                    success = future.result(timeout=600)  # 10 minute timeout
                    
                    if success:
                        results['success'] += 1
                        file_size_mb = file_path.stat().st_size / (1024 * 1024)
                        results['total_size_mb'] += file_size_mb
                        results['details'].append({
                            'file': file_path.name,
                            'success': True,
                            'size_mb': file_size_mb
                        })
                    else:
                        results['failed'] += 1
                        results['details'].append({
                            'file': file_path.name,
                            'success': False
                        })
                        
                except concurrent.futures.TimeoutError:
                    logger.error(f"Upload timeout for {file_path.name}")
                    results['failed'] += 1
                except Exception as e:
                    logger.error(f"Error uploading {file_path.name}: {e}")
                    results['failed'] += 1
        
        return results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PyTorch .pth weights to SafeTensors format and upload to HF Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python upload_models.py --weights_dir "ckpt" --repo_id "username/font-diffusion-weights" --token "hf_xxx"
  python upload_models.py --weights_dir "ckpt" --repo_id "username/font-diffusion-weights" --files "unet.pth" "style_encoder.pth"
  python upload_models.py --weights_dir "ckpt" --no-upload
  python upload_models.py --weights_dir "ckpt" --repo_id "username/font-diffusion-weights" --skip-conversion
        """,
    )
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
        help="Hugging Face repo ID",
    )
    parser.add_argument(
        "--token", 
        type=str, 
        default=None, 
        help="Hugging Face API token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=[
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
        ],
        help="Specific files to convert (default: all standard FontDiffusion weights)",
    )
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
        help="Make repository private"
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        default=False,
        help="Convert only, do not upload",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        default=False,
        help="Skip conversion, only upload",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Add converted safetensors and original pth weights",
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        default=False, 
        help="Verbose output"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip files that already exist in the repository",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Number of files to process in each batch (default: 50)",
    )
    return parser.parse_args()


def get_token(token_arg: Optional[str]) -> Optional[str]:
    """Get Hugging Face token from argument, env var, or cache."""
    # Priority: argument > env var > cached token
    if token_arg:
        return token_arg
    
    token_env = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if token_env:
        logger.info("Using HF token from environment variable")
        return token_env
    
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            logger.info("Using HF token from cache")
            return token
    except Exception:
        pass
    
    return None


def validate_and_collect_files(args: argparse.Namespace) -> Dict[str, Any]:
    """Validate inputs and collect files to process."""
    print("\n" + "="*70)
    print("VALIDATING INPUTS AND COLLECTING FILES")
    print("="*70)
    
    weights_dir = Path(args.weights_dir)
    if not weights_dir.exists():
        raise ValueError(f"Weights directory not found: {weights_dir}")
    
    print(f"✓ Weights directory: {weights_dir.absolute()}")
    
    # Collect all matching files
    all_files = []
    pth_files = []
    safe_files = []
    other_files = []
    
    for file_pattern in args.files:
        for file_path in weights_dir.glob(file_pattern):
            if file_path.is_file():
                all_files.append(file_path)
                if file_path.suffix == '.pth':
                    pth_files.append(file_path)
                elif file_path.suffix == '.safetensors':
                    safe_files.append(file_path)
                else:
                    other_files.append(file_path)
    
    # Also include any .pth files not explicitly listed
    if not args.files or len(args.files) > 5:  # If default or many files specified
        for pth_file in weights_dir.glob("*.pth"):
            if pth_file not in pth_files:
                pth_files.append(pth_file)
                all_files.append(pth_file)
    
    print(f"\n📁 Found {len(all_files)} total files:")
    print(f"  • {len(pth_files)} .pth files")
    print(f"  • {len(safe_files)} .safetensors files")
    print(f"  • {len(other_files)} other files")
    
    # Calculate total size
    total_size_mb = sum(f.stat().st_size for f in all_files) / (1024 * 1024)
    print(f"  • Total size: {total_size_mb:.2f} MB")
    
    # Validate upload requirements
    if not args.no_upload:
        if not args.repo_id:
            raise ValueError("--repo_id is required when uploading (use --no-upload to skip)")
        
        token = get_token(args.token)
        if not token:
            raise ValueError("HF token not found! Provide via --token or set HF_TOKEN env var")
        
        print(f"✓ HF token: {'*' * 20}")
        print(f"✓ Repository: {args.repo_id}")
    else:
        print("⊘ Skipping upload (--no-upload)")
    
    return {
        'weights_dir': weights_dir,
        'all_files': all_files,
        'pth_files': pth_files,
        'safe_files': safe_files,
        'other_files': other_files,
        'total_size_mb': total_size_mb,
    }


def convert_files_parallel(args: argparse.Namespace, file_info: Dict[str, Any]) -> bool:
    """Convert .pth files to .safetensors in parallel."""
    if args.skip_conversion:
        print("\n⊘ Skipping conversion (--skip-conversion)")
        return True
    
    print("\n" + "="*70)
    print("CONVERTING .pth TO .safetensors (PARALLEL)")
    print("="*70)
    
    pth_files = file_info['pth_files']
    if not pth_files:
        print("⚠ No .pth files found to convert")
        return True
    
    print(f"Converting {len(pth_files)} .pth files with {args.parallel} workers...")
    
    converter = ParallelConverter(num_workers=args.parallel, verbose=args.verbose)
    converter.stats.start_time = time.time()
    converter.stats.total_files = len(pth_files)
    
    # Convert in batches for better memory management
    batch_size = args.chunk_size
    all_results = []
    
    for i in range(0, len(pth_files), batch_size):
        batch = pth_files[i:i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(pth_files) + batch_size - 1)//batch_size}")
        
        batch_results = converter.convert_batch(batch)
        all_results.extend(batch_results)
        
        # Print batch summary
        success_count = sum(1 for r in batch_results if r and r.get('success'))
        failed_count = len(batch_results) - success_count
        print(f"  Batch complete: {success_count} succeeded, {failed_count} failed")
    
    # Print overall conversion statistics
    converter.stats.print_summary()
    
    # Print detailed conversion stats
    if args.verbose and all_results:
        successful_conversions = [r for r in all_results if r and r.get('success')]
        if successful_conversions:
            print("\n📊 Conversion Details:")
            for result in successful_conversions[:10]:  # Show first 10
                print(f"  • {result['filename']}: {result['pth_size_mb']:.2f}MB → "
                      f"{result['safe_size_mb']:.2f}MB ({result['compression']:.1f}%)")
            if len(successful_conversions) > 10:
                print(f"  ... and {len(successful_conversions) - 10} more")
    
    return converter.stats.failed_files == 0 or len(pth_files) == 0


def upload_files_parallel(args: argparse.Namespace, file_info: Dict[str, Any]) -> bool:
    """Upload files to Hugging Face Hub in parallel."""
    if args.no_upload:
        print("\n⊘ Skipping upload (--no-upload)")
        return True
    
    print("\n" + "="*70)
    print("UPLOADING TO HUGGING FACE HUB (PARALLEL)")
    print("="*70)
    
    all_files = file_info['all_files']
    weights_dir = file_info['weights_dir']
    
    if not all_files:
        print("⚠ No files found to upload")
        return True
    
    print(f"Uploading {len(all_files)} files to {args.repo_id}...")
    
    # Get token and login
    token = get_token(args.token)
    if not token:
        raise ValueError("HF token required for upload")
    
    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            exist_ok=True,
            private=args.private,
            token=token,
        )
        print(f"✓ Repository ready: {args.repo_id}")
    except Exception as e:
        print(f"⚠ Repository creation/verification: {e}")
        # Continue anyway - repository might already exist
    
    # Initialize uploader
    uploader = ParallelUploader(num_workers=args.parallel, verbose=args.verbose)
    uploader.set_token(token)
    
    # Upload in batches
    batch_size = args.chunk_size
    total_uploaded = 0
    total_failed = 0
    
    for i in range(0, len(all_files), batch_size):
        batch = all_files[i:i + batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(all_files) + batch_size - 1)//batch_size
        
        print(f"\n📤 Uploading batch {batch_num}/{total_batches} ({len(batch)} files)")
        
        results = uploader.upload_batch(batch, args.repo_id, weights_dir)
        
        total_uploaded += results['success']
        total_failed += results['failed']
        
        print(f"  Batch complete: {results['success']} succeeded, {results['failed']} failed")
    
    # Print upload summary
    print("\n" + "="*70)
    print("UPLOAD COMPLETE")
    print("="*70)
    print(f"Total files uploaded: {total_uploaded}")
    print(f"Failed uploads:      {total_failed}")
    print(f"Repository URL:      https://huggingface.co/{args.repo_type}s/{args.repo_id}")
    
    return total_failed == 0


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("🔥 HIGH-PERFORMANCE PYTORCH TO SAFETENSORS CONVERTER & UPLOADER")
    print("="*70)
    
    args = parse_arguments()
    start_time = time.time()
    
    try:
        # Parse arguments
        if args.verbose:
            print("\n📋 Configuration:")
            for key, value in vars(args).items():
                if key == 'token' and value:
                    print(f"  {key}: {'*' * 20}")
                else:
                    print(f"  {key}: {value}")
        
        # Validate and collect files
        file_info = validate_and_collect_files(args)
        
        # Convert files in parallel
        if not convert_files_parallel(args, file_info):
            if not args.skip_conversion:
                print("\n⚠ Some conversions failed, but continuing...")
        
        # Upload files in parallel
        if not upload_files_parallel(args, file_info):
            print("\n⚠ Some uploads failed")
        
        # Final summary
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("🎉 PROCESS COMPLETE!")
        print("="*70)
        print(f"Total time: {total_time:.2f} seconds")
        
        if not args.no_upload and args.repo_id:
            print(f"\n📦 Your weights are now available at:")
            print(f"   https://huggingface.co/{args.repo_type}s/{args.repo_id}")
            print(f"\n💾 Load them with:")
            print(f"   from safetensors.torch import load_file")
            print(f"   from huggingface_hub import hf_hub_download")
            print(f"   ")
            print(f"   # Download and load")
            print(f"   model_path = hf_hub_download(repo_id='{args.repo_id}', filename='model.safetensors')")
            print(f"   state_dict = load_file(model_path)")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())