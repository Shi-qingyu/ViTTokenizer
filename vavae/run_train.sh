#!/bin/bash

# Get config path from argument or environment variable
config_path=${1:-$CONFIG_PATH}

# Check if config path is provided
if [ -z "$config_path" ]; then
    echo "Usage: $0 <config_path>"
    echo "Example: $0 vavae/configs/f16d32_vfdinov2.yaml"
    exit 1
fi

# Set default values for distributed training if not provided
export WORLD_SIZE=${WORLD_SIZE:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-"12355"}

# Number of GPUs per node (adjust based on your hardware)
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

echo "Starting training with:"
echo "  Config: $config_path"
echo "  World Size: $WORLD_SIZE"
echo "  Rank: $RANK"
echo "  Master Addr: $MASTER_ADDR"
echo "  Master Port: $MASTER_PORT"
echo "  GPUs per node: $NPROC_PER_NODE"

# Check if config file exists
if [ ! -f "$config_path" ]; then
    echo "Error: Config file '$config_path' not found!"
    exit 1
fi

# For single node training (most common case)
if [ "$WORLD_SIZE" -eq 1 ]; then
    echo "Running single-node training..."
    if [ "$NPROC_PER_NODE" -gt 1 ]; then
        echo "Using torchrun for multi-GPU training..."
        torchrun --nproc_per_node=$NPROC_PER_NODE \
            --standalone \
            main.py \
            --base "$config_path" \
            --train
    else
        echo "Using single GPU training..."
        python main.py --base "$config_path" --train
    fi
else
    echo "Running multi-node training..."
    torchrun --nproc_per_node=$NPROC_PER_NODE \
        --nnodes=$WORLD_SIZE \
        --node_rank=$RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        main.py \
        --base "$config_path" \
        --train
fi