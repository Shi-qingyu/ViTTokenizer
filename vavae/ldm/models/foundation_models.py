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
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # vavae_dir = os.path.dirname(os.path.dirname(script_dir))

    cache_locations = [
        os.path.join('./offline_models'),
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
        checkpoint = torch.load(pytorch_model_path, map_location='cpu', weights_only=False)
        
        # Try multiple approaches to create the correct model architecture
        model = None
        
        # Approach 1: Try to recreate the exact model using saved config
        try:
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                print(f"Using saved model config: {config}")
                # Try to recreate model with the saved config
                # This might work for some timm models
                pass
        except:
            pass
        
        # Approach 2: Try with the original model name (this may work if HF cache exists)
        if model is None:
            try:
                # Set offline mode temporarily to use local cache
                original_offline = os.environ.get('HF_HUB_OFFLINE')
                os.environ['HF_HUB_OFFLINE'] = '1'
                
                # Try creating with original name (works if model def is available locally)
                model = timm.create_model(model_name, pretrained=False, dynamic_img_size=True)
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print(f"Loaded model using original name with strict=False: {pytorch_model_path}")
                
                # Restore offline setting
                if original_offline is not None:
                    os.environ['HF_HUB_OFFLINE'] = original_offline
                else:
                    del os.environ['HF_HUB_OFFLINE']
                    
            except Exception as e:
                # Restore offline setting
                if original_offline is not None:
                    os.environ['HF_HUB_OFFLINE'] = original_offline
                elif 'HF_HUB_OFFLINE' in os.environ:
                    del os.environ['HF_HUB_OFFLINE']
                print(f"Failed to load with original name: {e}")
        
        # Approach 3: Try with base architecture names if previous approaches failed
        if model is None:
            try:
                # Extract the base model name without hf-hub prefix for offline creation
                if "mae" in model_name.lower():
                    base_model_name = "vit_large_patch16_224"
                elif "dinov2" in model_name.lower():
                    # For DINOv2, try a few different base architectures
                    base_names = [
                        "vit_large_patch14_dinov2.lvd142m",
                    ]
                    for base_name in base_names:
                        try:
                            model = timm.create_model(base_name, pretrained=False, num_classes=0, dynamic_img_size=True)
                            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                            print(f"Loaded model using base name {base_name} with strict=False: {pytorch_model_path}")
                            break
                        except Exception as e:
                            print(f"Failed with base name {base_name}: {e}")
                            continue
                else:
                    # Fallback: try to extract from the model_name
                    base_model_name = model_name.split("/")[-1].replace(".mae", "").replace(".lvd142m", "")
                    model = timm.create_model(base_model_name, pretrained=False, num_classes=0, dynamic_img_size=True)
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    
                if model is None and "dinov2" not in model_name.lower():
                    # For MAE, try the base name
                    model = timm.create_model(base_model_name, pretrained=False, num_classes=0, dynamic_img_size=True)
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print(f"Loaded model using base name {base_model_name} with strict=False: {pytorch_model_path}")
                    
            except Exception as e:
                print(f"Failed to load with base architectures: {e}")
        
        if model is not None:
            return model
        else:
            raise RuntimeError(f"Failed to load model from {pytorch_model_path} - all loading approaches failed")
    else:
        # Try to load using timm's cache mechanism with offline environment
        # This works if the HF cache is properly set up
        try:
            # Set offline environment temporarily
            original_offline = os.environ.get('HF_HUB_OFFLINE')
            os.environ['HF_HUB_OFFLINE'] = '1'
            
            model = timm.create_model(model_name, pretrained=True)
            print(f"Loaded model from HF cache: {model_path}")
            
            # Restore original offline setting
            if original_offline is not None:
                os.environ['HF_HUB_OFFLINE'] = original_offline
            
            return model
        except Exception as e:
            # Restore original offline setting
            if original_offline is not None:
                os.environ['HF_HUB_OFFLINE'] = original_offline
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

def get_mae_encoder():
    """
    Load the MAE pretrained ViT-L encoder from the timm library.
    Supports both online and offline loading.
    """
    model_name = "hf-hub:timm/vit_large_patch16_224.mae"
    
    # Check if we're in offline mode or if local cache exists
    offline_mode = os.environ.get('HF_HUB_OFFLINE', '1') == '1'
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
    offline_mode = os.environ.get('HF_HUB_OFFLINE', '1') == '1'
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
        model = get_mae_encoder()
        if model is None:
            raise RuntimeError("Failed to load MAE encoder - model is None")
        return model, 1024
    elif type == 'dinov2':
        model = get_dinov2_encoder()
        if model is None:
            raise RuntimeError("Failed to load DINOv2 encoder - model is None")
        return model, 1024
    
    raise ValueError(f"Unknown model type: {type}")

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
        # print(f"DINOv2 input shape: {x.shape}")
        # print(f"DINOv2 output shape: {self.model.forward_features(x)[:, 1:].shape}")
        return self.model.forward_features(x)[:, 1:].reshape(b, h//16, w//16, -1).permute(0, 3, 1, 2)

    def forward(self, x):
        with torch.no_grad():
            if self.type == 'mae':
                return self.forward_mae(x)
            elif self.type == 'dinov2':
                return self.forward_dinov2(x)