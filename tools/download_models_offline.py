#!/usr/bin/env python3
"""
Download and cache foundation models (MAE, DINOv2, VGG) for offline use.
This script downloads all required models to local cache directories.
"""

import os
import sys
import requests
import hashlib
from pathlib import Path
import torch
import torchvision
from tqdm import tqdm

def download_file(url, local_path, expected_md5=None):
    """Download a file with progress bar and MD5 verification"""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    print(f"Downloading {url} -> {local_path}")
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=8192):
                    if data:
                        f.write(data)
                        pbar.update(len(data))
    
    # Verify MD5 if provided
    if expected_md5:
        print(f"Verifying MD5 hash...")
        with open(local_path, "rb") as f:
            content = f.read()
        actual_md5 = hashlib.md5(content).hexdigest()
        if actual_md5 != expected_md5:
            raise ValueError(f"MD5 mismatch! Expected: {expected_md5}, Got: {actual_md5}")
        print("✓ MD5 verification passed")
    
    print(f"✓ Downloaded: {local_path}")

def setup_offline_cache():
    """Setup offline model cache directories"""
    # Get HuggingFace cache directory
    hf_home = os.environ.get('HF_HOME', 
                            os.path.expanduser('~/.cache/huggingface'))
    
    # Create cache directories
    cache_dirs = {
        'hf_hub': os.path.join(hf_home, 'hub'),
        'torch_hub': os.path.join(hf_home, 'torch_hub'),
        'lpips': os.path.join(hf_home, 'lpips'),
        'offline_models': './offline_models'
    }
    
    for cache_dir in cache_dirs.values():
        os.makedirs(cache_dir, exist_ok=True)
    
    return cache_dirs

def download_mae_models(cache_dirs):
    """Download MAE models using timm"""
    print("\n=== Downloading MAE Models ===")
    
    try:
        import timm
        
        mae_models = [
            'vit_large_patch16_224.mae',
            'vit_base_patch16_224.mae',
            'vit_huge_patch14_224.mae'
        ]
        
        for model_name in mae_models:
            print(f"\nDownloading MAE model: {model_name}")
            try:
                # Load model to trigger download
                model = timm.create_model(model_name, pretrained=True, num_classes=0)
                print(f"✓ Downloaded and cached: {model_name}")
            except Exception as e:
                print(f"✗ Failed to download {model_name}: {e}")
                
    except ImportError:
        print("✗ timm not installed. Install with: pip install timm")

def download_dinov2_models(cache_dirs):
    """Download DINOv2 models using timm"""
    print("\n=== Downloading DINOv2 Models ===")
    
    try:
        import timm
        
        dinov2_models = [
            'vit_base_patch14_dinov2.lvd142m',
            'vit_large_patch14_dinov2.lvd142m',
            'vit_giant_patch14_dinov2.lvd142m'
        ]
        
        for model_name in dinov2_models:
            print(f"\nDownloading DINOv2 model: {model_name}")
            try:
                # Load model to trigger download
                model = timm.create_model(model_name, pretrained=True, num_classes=0)
                print(f"✓ Downloaded and cached: {model_name}")
            except Exception as e:
                print(f"✗ Failed to download {model_name}: {e}")
                
    except ImportError:
        print("✗ timm not installed. Install with: pip install timm")

def download_vgg_models(cache_dirs):
    """Download VGG models for LPIPS"""
    print("\n=== Downloading VGG Models ===")
    
    # 1. Download VGG16 from torchvision
    print("\nDownloading VGG16 from torchvision...")
    try:
        # This will download to torch cache
        model = torchvision.models.vgg16(pretrained=True)
        print("✓ VGG16 from torchvision downloaded and cached")
    except Exception as e:
        print(f"✗ Failed to download VGG16: {e}")
    
    # 2. Download custom LPIPS VGG checkpoint
    print("\nDownloading LPIPS VGG checkpoint...")
    lpips_url = "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
    lpips_path = os.path.join(cache_dirs['lpips'], 'vgg.pth')
    lpips_md5 = "d507d7349b931f0638a25a48a722f98a"
    
    try:
        if not os.path.exists(lpips_path):
            download_file(lpips_url, lpips_path, lpips_md5)
        else:
            print(f"✓ LPIPS VGG checkpoint already exists: {lpips_path}")
    except Exception as e:
        print(f"✗ Failed to download LPIPS VGG checkpoint: {e}")
    
    # 3. Create alternative paths for offline access
    alternative_paths = [
        os.path.join(cache_dirs['offline_models'], 'lpips', 'vgg.pth'),
        os.path.join(cache_dirs['offline_models'], 'movqgan/modules/losses/lpips', 'vgg.pth'),
        './vgg.pth'
    ]
    
    for alt_path in alternative_paths:
        if os.path.exists(lpips_path) and not os.path.exists(alt_path):
            os.makedirs(os.path.dirname(alt_path), exist_ok=True)
            import shutil
            shutil.copy2(lpips_path, alt_path)
            print(f"✓ Copied LPIPS checkpoint to: {alt_path}")

def create_offline_env_script(cache_dirs):
    """Create environment setup script for offline usage"""
    env_script = """#!/bin/bash
# Offline model environment setup
# Source this script before running training: source offline_env.sh

export TORCH_HOME="{torch_home}"
export HF_HOME="{hf_home}"
export HF_HUB_CACHE="{hf_hub_cache}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Set torch hub to offline mode
export TORCH_HUB_OFFLINE=1

echo "✓ Offline model environment configured"
echo "  TORCH_HOME: $TORCH_HOME"
echo "  HF_HOME: $HF_HOME" 
echo "  HF_HUB_CACHE: $HF_HUB_CACHE"
echo "  HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
echo "  TORCH_HUB_OFFLINE: $TORCH_HUB_OFFLINE"
""".format(
        torch_home=os.path.dirname(cache_dirs['torch_hub']),
        hf_home=os.path.dirname(cache_dirs['hf_hub']),
        hf_hub_cache=cache_dirs['hf_hub']
    )
    
    with open('offline_env.sh', 'w') as f:
        f.write(env_script)
    
    print(f"\n✓ Created offline environment script: offline_env.sh")
    print("  Usage: source offline_env.sh")

def main():
    print("🚀 Foundation Models Offline Downloader")
    print("=====================================")
    
    # Setup cache directories
    cache_dirs = setup_offline_cache()
    print("\n📁 Cache directories:")
    for name, path in cache_dirs.items():
        print(f"  {name}: {path}")
    
    # Download all models
    download_mae_models(cache_dirs)
    download_dinov2_models(cache_dirs)
    download_vgg_models(cache_dirs)
    
    # Create environment setup script
    create_offline_env_script(cache_dirs)
    
    print("\n🎉 All models downloaded successfully!")
    print("\n📋 Next steps:")
    print("1. Copy this directory to your offline cluster")
    print("2. Run: source offline_env.sh")
    print("3. Start training with offline model support")

if __name__ == "__main__":
    main() 