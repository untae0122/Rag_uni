#!/bin/bash
# Corag Experiment 6: Oracle Injection (Oracle-Hard)
# Usage: ./scripts/corag/exp6_oracle_injection.sh [GPU_ID]

GPU_ID=${1:-0}
INDEX_DIR="datasets/hotpotqa/e5_index"
CORPUS_PATH="datasets/hotpotqa/corpus.jsonl"
VLLM_HOST="localhost"
VLLM_PORT="8000"
ATTACKER_API_BASE="http://localhost:8001/v1"
ATTACKER_MODEL_NAME="Qwen/Qwen2.5-32B-Instruct"

export CUDA_VISIBLE_DEVICES=$GPU_ID
export VLLM_HOST=$VLLM_HOST
export VLLM_PORT=$VLLM_PORT

echo "Running Corag Exp 6: Oracle Injection"
python models/corag/src/inference/attack_corag.py \
    --attack_mode oracle_injection \
    --index_dir $INDEX_DIR \
    --corpus_path $CORPUS_PATH \
    --output_dir results/trajectory_results/corag \
    --attacker_api_base $ATTACKER_API_BASE \
    --attacker_model_name $ATTACKER_MODEL_NAME
