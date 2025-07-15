"""
We use the file to instantiate the vision foundation models.
They serves as the auxiliary regularizer for the autoencoder.

by Jingfeng Yao
from HUST-VL
"""

import os
import timm
import torch
import torch.nn as nn
from pathlib import Path

def get_offline_model_path(model_type):
    """
    Get the path to offline model cache.
    
    Args:
        model_type: 'mae' or 'dinov2'
    
    Returns:
        Path to the offline model directory if it exists, None otherwise
    """
    # Check common cache locations
    cache_locations = [
        os.environ.get('HF_HOME'),
        os.environ.get('HF_HUB_CACHE'),
        './offline_models',
        './models',
        '~/.cache/huggingface/hub'
    ]
    
    model_names = {
        'mae': ['mae_vit_large_patch16_224', 'models--timm--vit_large_patch16_224.mae'],
        'dinov2': ['dinov2_vit_large_patch14', 'models--timm--vit_large_patch14_dinov2.lvd142m']
    }
    
    for cache_dir in cache_locations:
        if cache_dir is None:
            continue
        
        cache_path = Path(cache_dir).expanduser()
        if not cache_path.exists():
            continue
            
        for model_name in model_names[model_type]:
            model_path = cache_path / model_name
            if model_path.exists():
                # Check if it contains model files
                if (model_path / "pytorch_model.bin").exists():
                    return model_path
                # Check for HF hub cache structure
                for snapshot_dir in model_path.glob("snapshots/*"):
                    if snapshot_dir.is_dir():
                        return snapshot_dir
    
    return None

def load_model_from_local_cache(model_path, model_name):
    """
    Load model from local cache directory.
    
    Args:
        model_path: Path to the cached model
        model_name: Original model name for timm.create_model
    
    Returns:
        Loaded model
    """
    pytorch_model_path = model_path / "pytorch_model.bin"
    
    if pytorch_model_path.exists():
        # Load from our custom saved format
        checkpoint = torch.load(pytorch_model_path, map_location='cpu')
        
        # Create model with default config
        model = timm.create_model(model_name, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from local cache: {pytorch_model_path}")
        return model
    else:
        # Try to load using timm's cache mechanism
        # This works if the HF cache is properly set up
        try:
            model = timm.create_model(model_name, pretrained=True)
            print(f"Loaded model from HF cache: {model_path}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

def get_mae_encoder():
    """
    Load the MAE pretrained ViT-L encoder from the timm library.
    Supports both online and offline loading.
    """
    model_name = "hf-hub:timm/vit_large_patch16_224.mae"
    
    # Check if we're in offline mode or if local cache exists
    offline_mode = os.environ.get('HF_HUB_OFFLINE', '0') == '1'
    offline_path = get_offline_model_path('mae')
    
    if offline_mode or offline_path:
        if offline_path:
            print(f"Loading MAE model from offline cache: {offline_path}")
            model = load_model_from_local_cache(offline_path, model_name)
        else:
            raise RuntimeError(
                "HF_HUB_OFFLINE=1 but no offline model cache found. "
                "Please run download_models_offline.py first or set correct HF_HOME path."
            )
    else:
        print("Loading MAE model from online source...")
        try:
            model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True)
        except Exception as e:
            print(f"Failed to load online, trying offline cache: {e}")
            offline_path = get_offline_model_path('mae')
            if offline_path:
                model = load_model_from_local_cache(offline_path, model_name)
            else:
                raise RuntimeError(f"Failed to load MAE model both online and offline: {e}")
    
    model.requires_grad_(False)
    return model

def get_dinov2_encoder():
    """
    Load the DINOv2 pretrained ViT-L encoder from the timm library.
    Supports both online and offline loading.
    """
    model_name = "hf-hub:timm/vit_large_patch14_dinov2.lvd142m"
    
    # Check if we're in offline mode or if local cache exists
    offline_mode = os.environ.get('HF_HUB_OFFLINE', '0') == '1'
    offline_path = get_offline_model_path('dinov2')
    
    if offline_mode or offline_path:
        if offline_path:
            print(f"Loading DINOv2 model from offline cache: {offline_path}")
            model = load_model_from_local_cache(offline_path, model_name)
        else:
            raise RuntimeError(
                "HF_HUB_OFFLINE=1 but no offline model cache found. "
                "Please run download_models_offline.py first or set correct HF_HOME path."
            )
    else:
        print("Loading DINOv2 model from online source...")
        try:
            model = timm.create_model(model_name, pretrained=True, dynamic_img_size=True)
        except Exception as e:
            print(f"Failed to load online, trying offline cache: {e}")
            offline_path = get_offline_model_path('dinov2')
            if offline_path:
                model = load_model_from_local_cache(offline_path, model_name)
            else:
                raise RuntimeError(f"Failed to load DINOv2 model both online and offline: {e}")
    
    model.requires_grad_(False)
    return model

def create_foundation_model(
    type,
):
    assert type in ['mae', 'dinov2'], f"Unsupported foundation model type: {type}"

    if type == 'mae':
        return get_mae_encoder(), 1024
    elif type == 'dinov2':
        return get_dinov2_encoder(), 1024

class aux_foundation_model(nn.Module):
    """
    Load the foundation model and forward the input image to get 
    the feature maps.
    """
    def __init__(self, type):
        super().__init__()
        self.model, feature_dim = create_foundation_model(type)
        self.type = type
        self.feature_dim = feature_dim

    def forward_mae(self, x):
        b, c, h, w = x.shape
        return self.model.forward_features(x)[:, 1:].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2)
    
    def forward_dinov2(self, x):
        b, c, h, w = x.shape
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model.forward_features(x)[:, 1:].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2)
        
    def forward(self, x):
        with torch.no_grad():
            if self.type == 'mae':
                return self.forward_mae(x)
            elif self.type == 'dinov2':
                return self.forward_dinov2(x)