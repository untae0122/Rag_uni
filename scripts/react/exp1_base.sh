#!/bin/bash
# ReAct Experiment 1: Base (Clean)
# Usage: ./scripts/react/exp1_base.sh [GPU_ID]

GPU_ID=${1:-0}
MODEL_PATH="/home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
INDEX_DIR="datasets/hotpotqa/e5_index"
CORPUS_PATH="datasets/hotpotqa/corpus.jsonl"
DATA_PATH="datasets/hotpotqa/hotpotqa100.json"

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Running ReAct Exp 1: Base"
python models/react/attack_react.py \
    --attack_mode base \
    --model_path $MODEL_PATH \
    --index_dir $INDEX_DIR \
    --corpus_path $CORPUS_PATH \
    --data_path $DATA_PATH \
    --output_dir results/trajectory_results/react
