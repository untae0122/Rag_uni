#!/bin/bash
# Corag Experiment 5: Dynamic Retrieval (Oracle-Soft)
# Usage: ./scripts/corag/exp5_dynamic_retrieval.sh [GPU_ID]

GPU_ID=${1:-0}
INDEX_DIR="datasets/hotpotqa/e5_index"
CORPUS_PATH="datasets/hotpotqa/corpus.jsonl"
DATA_PATH="datasets/hotpotqa/hotpotqa100.json"
QID_TO_IDX_PATH="datasets/hotpotqa/qid_to_idx.json"
ADV_DATA_PATH="datasets/hotpotqa/hotpotqa.json"
VLLM_HOST="localhost"
VLLM_PORT="8000"
ATTACKER_API_BASE="http://localhost:8001/v1"
ATTACKER_MODEL_NAME="Qwen/Qwen2.5-32B-Instruct" # Change if needed

export CUDA_VISIBLE_DEVICES=$GPU_ID
export VLLM_HOST=$VLLM_HOST
export VLLM_PORT=$VLLM_PORT

echo "Running Corag Exp 5: Dynamic Retrieval"
python models/corag/src/inference/attack_corag.py \
    --attack_mode dynamic_retrieval \
    --index_dir $INDEX_DIR \
    --corpus_path $CORPUS_PATH \
    --adv_data_path $ADV_DATA_PATH \
    --qid_to_idx_path $QID_TO_IDX_PATH \
    --output_dir results/trajectory_results/corag \
    --attacker_api_base $ATTACKER_API_BASE \
    --attacker_model_name $ATTACKER_MODEL_NAME
