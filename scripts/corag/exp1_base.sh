#!/bin/bash
# Corag Experiment 1: Base (Clean)
# Usage: ./scripts/corag/exp1_base.sh [GPU_ID]

GPU_ID=${1:-0}
INDEX_DIR="datasets/hotpotqa/e5_index"
CORPUS_PATH="datasets/hotpotqa/corpus.jsonl"
QID_TO_IDX_PATH="datasets/hotpotqa/qid_to_idx.json"
ADV_DATA_PATH="datasets/hotpotqa/hotpotqa100.json"
VLLM_HOST="localhost"
VLLM_PORT="8005"

# Note: Corag Agent requires VLLM server running at $VLLM_HOST:$VLLM_PORT
# And you must have launched it with the Agent Model.

export CUDA_VISIBLE_DEVICES=$GPU_ID
export VLLM_HOST=$VLLM_HOST
export VLLM_PORT=$VLLM_PORT

echo "Running Corag Exp 1: Base"
python models/corag/src/inference/attack_corag.py \
    --attack_mode base \
    --base \
    --index_dir $INDEX_DIR \
    --corpus_path $CORPUS_PATH \
    --adv_data_path $ADV_DATA_PATH \
    --qid_to_idx_path $QID_TO_IDX_PATH \
    --output_dir results/trajectory_results/corag
