#!/usr/bin/env python3
"""
Script to download MAE and DINOv2 models for offline use.
Run this on a machine with internet access, then transfer the cache to your cluster.

Usage:
    python download_models_offline.py --cache_dir ./offline_models
"""

import argparse
import os
import timm
import torch
from pathlib import Path

def download_mae_model(cache_dir):
    """Download MAE model and save to local directory"""
    print("Downloading MAE model...")
    
    # Set cache directory
    os.environ['HF_HOME'] = cache_dir
    os.environ['HF_HUB_CACHE'] = cache_dir
    
    # Download the model (this will cache it)
    model = timm.create_model("hf-hub:timm/vit_large_patch16_224.mae", pretrained=True)
    
    # Save model weights explicitly
    model_path = Path(cache_dir) / "mae_vit_large_patch16_224"
    model_path.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.default_cfg,
    }, model_path / "pytorch_model.bin")
    
    print(f"MAE model saved to {model_path}")
    return model_path

def download_dinov2_model(cache_dir):
    """Download DINOv2 model and save to local directory"""
    print("Downloading DINOv2 model...")
    
    # Set cache directory
    os.environ['HF_HOME'] = cache_dir
    os.environ['HF_HUB_CACHE'] = cache_dir
    
    # Download the model (this will cache it)
    model = timm.create_model("hf-hub:timm/vit_large_patch14_dinov2.lvd142m", pretrained=True)
    
    # Save model weights explicitly
    model_path = Path(cache_dir) / "dinov2_vit_large_patch14"
    model_path.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.default_cfg,
    }, model_path / "pytorch_model.bin")
    
    print(f"DINOv2 model saved to {model_path}")
    return model_path

def main():
    parser = argparse.ArgumentParser(description="Download models for offline use")
    parser.add_argument("--cache_dir", type=str, default="./offline_models", 
                      help="Directory to store downloaded models")
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir).absolute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading models to: {cache_dir}")
    
    # Download both models
    mae_path = download_mae_model(str(cache_dir))
    dinov2_path = download_dinov2_model(str(cache_dir))
    
    # Create a summary file
    summary_file = cache_dir / "models_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Downloaded Models Summary\n")
        f.write("========================\n\n")
        f.write(f"MAE Model: {mae_path}\n")
        f.write(f"DINOv2 Model: {dinov2_path}\n")
        f.write(f"\nHugging Face Cache: {cache_dir}\n")
        f.write("\nTo use offline, set these environment variables:\n")
        f.write(f"export HF_HOME={cache_dir}\n")
        f.write("export HF_HUB_OFFLINE=1\n")
    
    print(f"\nDownload complete! Summary saved to {summary_file}")
    print(f"\nTo transfer to cluster:")
    print(f"1. Copy the entire '{cache_dir}' directory to your cluster")
    print(f"2. Set environment variables: HF_HOME={cache_dir} and HF_HUB_OFFLINE=1")

if __name__ == "__main__":
    main() 