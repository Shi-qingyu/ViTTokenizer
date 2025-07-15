# Offline Model Setup for MAE and DINOv2

This guide explains how to use MAE and DINOv2 models in your cluster environment without internet access.

## Overview

Since your cluster cannot connect to the internet, we need to:
1. Download the models on a machine with internet access
2. Transfer the model cache to your cluster
3. Configure your code to use the offline models

## Step-by-Step Setup

### 1. Download Models (On Machine with Internet)

On a machine with internet access, run the download script:

```bash
# Install required packages
pip install timm torch huggingface_hub

# Download models to local cache
python download_models_offline.py --cache_dir ./offline_models
```

This will create an `offline_models` directory containing:
- MAE ViT-Large model
- DINOv2 ViT-Large model  
- All necessary model files and configurations

### 2. Transfer to Cluster

Copy the entire `offline_models` directory to your cluster:

```bash
# Example using scp
scp -r offline_models/ user@cluster:/path/to/your/project/

# Or using rsync
rsync -av offline_models/ user@cluster:/path/to/your/project/offline_models/
```

### 3. Setup Environment on Cluster

Run the setup script to configure environment variables:

```bash
# Make script executable
chmod +x setup_offline_models.sh

# Run setup (specify cache directory if different)
./setup_offline_models.sh ./offline_models
```

This creates an `offline_env.sh` file with the correct environment variables.

### 4. Activate Offline Environment

Before running your code, activate the offline environment:

```bash
source offline_env.sh
```

This sets:
- `HF_HOME`: Points to your offline model cache
- `HF_HUB_OFFLINE=1`: Forces offline mode
- `HF_HUB_CACHE`: Cache directory for Hugging Face Hub

### 5. Test the Setup

Verify everything works:

```bash
python test_offline_models.py
```

This will test:
- Environment variable configuration
- Model loading from cache
- Model inference with dummy data

## Usage in Your Code

After setting up offline models, your existing code should work without changes:

```python
from vavae.ldm.models.foundation_models import aux_foundation_model

# This will automatically use offline models
mae_model = aux_foundation_model('mae')
dinov2_model = aux_foundation_model('dinov2')

# Use models normally
with torch.no_grad():
    features = mae_model(input_images)
```

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `HF_HOME` | Hugging Face cache directory | `/path/to/offline_models` |
| `HF_HUB_OFFLINE` | Force offline mode | `1` |
| `HF_HUB_CACHE` | Hub cache location | `/path/to/offline_models` |

## Troubleshooting

### Model Not Found Error

If you get "model not found" errors:

1. Check environment variables:
   ```bash
   echo $HF_HOME
   echo $HF_HUB_OFFLINE
   ```

2. Verify model files exist:
   ```bash
   find $HF_HOME -name "*mae*" -o -name "*dinov2*"
   ```

3. Check permissions:
   ```bash
   ls -la $HF_HOME
   ```

### Cache Directory Issues

The code looks for models in these locations (in order):
1. `$HF_HOME` (environment variable)
2. `$HF_HUB_CACHE` (environment variable)  
3. `./offline_models` (current directory)
4. `./models` (current directory)
5. `~/.cache/huggingface/hub` (default cache)

### Internet Connection Errors

If you still see internet connection attempts:

1. Ensure `HF_HUB_OFFLINE=1` is set:
   ```bash
   export HF_HUB_OFFLINE=1
   ```

2. Check that model files are in the correct format
3. Verify the cache directory structure

### Model Loading Fallback

The code implements automatic fallback:
1. First tries offline cache if `HF_HUB_OFFLINE=1` or cache exists
2. Falls back to online download if offline fails
3. Gives clear error messages if both fail

## File Structure

After setup, your directory should look like:

```
your_project/
├── offline_models/           # Model cache directory
│   ├── mae_vit_large_patch16_224/
│   │   └── pytorch_model.bin
│   ├── dinov2_vit_large_patch14/
│   │   └── pytorch_model.bin
│   └── models_summary.txt
├── download_models_offline.py  # Download script
├── setup_offline_models.sh     # Setup script  
├── offline_env.sh              # Environment variables
├── test_offline_models.py      # Test script
└── vavae/ldm/models/foundation_models.py  # Modified model loading
```

## Alternative Methods

### Method 1: Manual Download with `huggingface-cli`

```bash
# Download models manually
huggingface-cli download timm/vit_large_patch16_224.mae --cache-dir ./offline_models
huggingface-cli download timm/vit_large_patch14_dinov2.lvd142m --cache-dir ./offline_models
```

### Method 2: Using Python huggingface_hub

```python
from huggingface_hub import snapshot_download

# Download MAE
snapshot_download(
    repo_id="timm/vit_large_patch16_224.mae",
    cache_dir="./offline_models"
)

# Download DINOv2  
snapshot_download(
    repo_id="timm/vit_large_patch14_dinov2.lvd142m", 
    cache_dir="./offline_models"
)
```

## Advanced Configuration

### Custom Cache Location

To use a different cache location:

```bash
export HF_HOME="/custom/path/to/models"
export HF_HUB_CACHE="/custom/path/to/models"
```

### Multiple Model Versions

To cache multiple versions of models:

```python
# Download specific revisions
model = timm.create_model("hf-hub:timm/vit_large_patch16_224.mae", 
                         pretrained=True, revision="main")
```

### Disk Space Considerations

Model sizes:
- MAE ViT-Large: ~1.2 GB
- DINOv2 ViT-Large: ~1.1 GB
- Total: ~2.3 GB + cache overhead

## Support

If you encounter issues:

1. Run the test script: `python test_offline_models.py`
2. Check environment variables: `env | grep HF_`
3. Verify model files: `find $HF_HOME -name "*.bin" -o -name "*.safetensors"`
4. Check the model loading logs for specific error messages

The modified `foundation_models.py` provides detailed logging to help diagnose issues. 