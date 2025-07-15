#!/bin/bash
# Source this file to set up offline model environment
# Usage: source offline_env.sh

export HF_HOME="/home/sqy/projects/tokenizer/ViTTokenizer/vavae/setup/offline_models"
export HF_HUB_CACHE="/home/sqy/projects/tokenizer/ViTTokenizer/vavae/setup/offline_models"
export HF_HUB_OFFLINE=1

echo "Offline model environment activated:"
echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"
echo "  HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
