#!/bin/bash
# Corag Experiment 2: Static Main-Main
# Usage: ./scripts/corag/exp2_static_main_main.sh [GPU_ID]

GPU_ID=${1:-0}
INDEX_DIR="datasets/hotpotqa/e5_index"
CORPUS_PATH="datasets/hotpotqa/corpus.jsonl"
POISONED_INDEX_DIR="datasets/hotpotqa/poisoned_index_exp2"
POISONED_CORPUS_PATH="datasets/hotpotqa/poisoned_corpus_exp2.jsonl"
VLLM_HOST="localhost"
VLLM_PORT="8000"

export CUDA_VISIBLE_DEVICES=$GPU_ID
export VLLM_HOST=$VLLM_HOST
export VLLM_PORT=$VLLM_PORT

echo "Running Corag Exp 2: Static Main-Main"
python models/corag/src/inference/attack_corag.py \
    --attack_mode static_main_main \
    --index_dir $INDEX_DIR \
    --corpus_path $CORPUS_PATH \
    --poisoned_index_dir $POISONED_INDEX_DIR \
    --poisoned_corpus_path $POISONED_CORPUS_PATH \
    --output_dir results/trajectory_results/corag
