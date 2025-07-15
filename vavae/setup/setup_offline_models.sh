#!/bin/bash

# Setup script for offline MAE and DINOv2 models
# Usage: ./setup_offline_models.sh [cache_directory]

set -e

CACHE_DIR=${1:-"./offline_models"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

echo "Setting up offline models in: $CACHE_DIR"

# Create cache directory
mkdir -p "$CACHE_DIR"
CACHE_DIR=$(realpath "$CACHE_DIR")

# Set environment variables for current session
export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"
export HF_HUB_OFFLINE=1

echo "Environment variables set:"
echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"
echo "  HF_HUB_OFFLINE=$HF_HUB_OFFLINE"

# Create environment file for future use
ENV_FILE="$SCRIPT_DIR/offline_env.sh"
cat > "$ENV_FILE" << EOF
#!/bin/bash
# Source this file to set up offline model environment
# Usage: source offline_env.sh

export HF_HOME="$CACHE_DIR"
export HF_HUB_CACHE="$CACHE_DIR"
export HF_HUB_OFFLINE=1

echo "Offline model environment activated:"
echo "  HF_HOME=\$HF_HOME"
echo "  HF_HUB_CACHE=\$HF_HUB_CACHE"
echo "  HF_HUB_OFFLINE=\$HF_HUB_OFFLINE"
EOF

chmod +x "$ENV_FILE"

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. If you haven't already, download models on a machine with internet:"
echo "   python download_models_offline.py --cache_dir $CACHE_DIR"
echo ""
echo "2. Copy the '$CACHE_DIR' directory to your cluster"
echo ""
echo "3. On your cluster, activate the offline environment:"
echo "   source $ENV_FILE"
echo ""
echo "4. Run your code normally - it will automatically use offline models"

# Check if models exist
if [ -d "$CACHE_DIR" ]; then
    MAE_COUNT=$(find "$CACHE_DIR" -name "*mae*" -type d | wc -l)
    DINOV2_COUNT=$(find "$CACHE_DIR" -name "*dinov2*" -type d | wc -l)
    
    echo ""
    echo "Model cache status:"
    echo "  MAE models found: $MAE_COUNT"
    echo "  DINOv2 models found: $DINOV2_COUNT"
    
    if [ "$MAE_COUNT" -gt 0 ] && [ "$DINOV2_COUNT" -gt 0 ]; then
        echo "  ✓ Both models appear to be cached"
    else
        echo "  ⚠ Some models may be missing - run download script first"
    fi
fi 