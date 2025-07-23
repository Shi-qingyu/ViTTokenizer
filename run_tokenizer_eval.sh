CONFIG_PATH=$1
MODEL_TYPE=${2:-'vavae'}
DATA_PATH=${3:-'/data02/sqy/datasets/imageNet1K/ILSVRC2012_validation'}
OUTPUT_PATH=${4:-'eval_results'}

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-1235}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    evaluate_tokenizer.py \
    --config_path $CONFIG_PATH \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --model_type $MODEL_TYPE