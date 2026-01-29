#!/bin/bash
# ReAct Experiment 5: Dynamic Retrieval (Oracle-Soft)
# Usage: ./scripts/react/exp5_dynamic_retrieval.sh [GPU_ID]

GPU_ID=${1:-0}
MODEL_PATH="/home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
INDEX_DIR="datasets/hotpotqa/e5_index"
CORPUS_PATH="datasets/hotpotqa/corpus.jsonl"
DATA_PATH="datasets/hotpotqa/hotpotqa100.json"
ATTACKER_API_BASE="http://localhost:8001/v1"

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Running ReAct Exp 5: Dynamic Retrieval"
python models/react/attack_react.py \
    --attack_mode dynamic_retrieval \
    --model_path $MODEL_PATH \
    --index_dir $INDEX_DIR \
    --corpus_path $CORPUS_PATH \
    --data_path $DATA_PATH \
    --output_dir results/trajectory_results/react \
    --attacker_api_base $ATTACKER_API_BASE \
    --attacker_model_name $MODEL_PATH
