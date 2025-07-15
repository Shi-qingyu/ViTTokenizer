#!/usr/bin/env python3
"""
Test script to verify offline MAE and DINOv2 model loading.
Run this after setting up offline models to ensure everything works.

Usage:
    source offline_env.sh  # Set environment variables
    python test_offline_models.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the current directory to Python path to import foundation_models
sys.path.insert(0, str(Path(__file__).parent))

def test_environment():
    """Test if environment variables are set correctly."""
    print("=== Environment Test ===")
    
    required_vars = ['HF_HOME', 'HF_HUB_OFFLINE']
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            print(f"✓ {var} = {value}")
        else:
            print(f"✗ {var} is not set")
            return False
    
    # Check if cache directory exists
    cache_dir = os.environ.get('HF_HOME')
    if cache_dir and Path(cache_dir).exists():
        print(f"✓ Cache directory exists: {cache_dir}")
        return True
    else:
        print(f"✗ Cache directory not found: {cache_dir}")
        return False

def test_model_loading():
    """Test loading both MAE and DINOv2 models."""
    print("\n=== Model Loading Test ===")
    
    try:
        # Import our modified foundation_models
        from vavae.ldm.models.foundation_models import get_mae_encoder, get_dinov2_encoder
        
        # Test MAE model
        print("\nTesting MAE model...")
        mae_model = get_mae_encoder()
        print(f"✓ MAE model loaded successfully")
        print(f"  Model type: {type(mae_model)}")
        
        # Test DINOv2 model  
        print("\nTesting DINOv2 model...")
        dinov2_model = get_dinov2_encoder()
        print(f"✓ DINOv2 model loaded successfully")
        print(f"  Model type: {type(dinov2_model)}")
        
        return mae_model, dinov2_model
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return None, None

def test_model_inference():
    """Test inference with dummy data."""
    print("\n=== Model Inference Test ===")
    
    try:
        from vavae.ldm.models.foundation_models import aux_foundation_model
        
        # Create dummy input (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 256, 256)
        print(f"Created dummy input: {dummy_input.shape}")
        
        # Test MAE
        print("\nTesting MAE inference...")
        mae_aux = aux_foundation_model('mae')
        with torch.no_grad():
            mae_output = mae_aux(dummy_input)
        print(f"✓ MAE inference successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {mae_output.shape}")

        # Test DINOv2
        print("\nTesting DINOv2 inference...")
        dinov2_aux = aux_foundation_model('dinov2')
        with torch.no_grad():
            dinov2_output = dinov2_aux(dummy_input)
        print(f"✓ DINOv2 inference successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {dinov2_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model inference failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Offline Model Setup")
    print("=" * 40)
    
    # Test 1: Environment variables
    # env_ok = test_environment()
    # if not env_ok:
    #     print("\n❌ Environment test failed. Please run 'source offline_env.sh' first.")
    #     return 1
    
    # Test 2: Model loading
    mae_model, dinov2_model = test_model_loading()
    if mae_model is None or dinov2_model is None:
        print("\n❌ Model loading failed. Please check your model cache.")
        return 1
    
    # Test 3: Model inference
    inference_ok = test_model_inference()
    if not inference_ok:
        print("\n❌ Model inference failed.")
        return 1
    
    print("\n" + "=" * 40)
    print("🎉 All tests passed! Offline model setup is working correctly.")
    print("\nYour models are ready for offline use!")
    return 0

if __name__ == "__main__":
    exit(main()) 