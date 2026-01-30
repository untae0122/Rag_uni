#!/bin/bash
# ReAct Experiment 2: Static Main-Main
# Usage: ./scripts/react/exp2_static_main_main.sh [GPU_ID]

GPU_ID=${1:-0}
MODEL_PATH="/home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
INDEX_DIR="datasets/hotpotqa/e5_index"
CORPUS_PATH="datasets/hotpotqa/corpus.jsonl"
DATA_PATH="datasets/hotpotqa/hotpotqa100_x3.json"
QID_TO_IDX_PATH="datasets/hotpotqa/qid_to_idx.json"
POISONED_INDEX_DIR="datasets/hotpotqa/e5_index_poisoned_main_main_x3"
POISONED_CORPUS_PATH="datasets/hotpotqa/poisoned_corpus_main_main_x3.jsonl"

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Running ReAct Exp 2: Static Main-Main"
python models/react/attack_react.py \
    --attack_mode static_main_main \
    --model_path $MODEL_PATH \
    --index_dir $INDEX_DIR \
    --corpus_path $CORPUS_PATH \
    --poisoned_index_dir $POISONED_INDEX_DIR \
    --poisoned_corpus_path $POISONED_CORPUS_PATH \
    --data_path $DATA_PATH \
    --qid_to_idx_path $QID_TO_IDX_PATH \
    --output_dir results/trajectory_results/react
