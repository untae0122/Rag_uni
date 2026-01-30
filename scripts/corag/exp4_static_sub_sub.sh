#!/bin/bash
# Corag Experiment 4: Static Sub-Sub
# Usage: ./scripts/corag/exp4_static_sub_sub.sh [GPU_ID]

GPU_ID=${1:-0}
INDEX_DIR="datasets/hotpotqa/e5_index"
CORPUS_PATH="datasets/hotpotqa/corpus.jsonl"
QID_TO_IDX_PATH="datasets/hotpotqa/qid_to_idx.json"
ADV_DATA_PATH="datasets/hotpotqa/hotpotqa100.json"
POISONED_INDEX_DIR="datasets/hotpotqa/poisoned_index_exp4"
POISONED_CORPUS_PATH="datasets/hotpotqa/poisoned_corpus_exp4.jsonl"
VLLM_HOST="localhost"
VLLM_PORT="8000"

export CUDA_VISIBLE_DEVICES=$GPU_ID
export VLLM_HOST=$VLLM_HOST
export VLLM_PORT=$VLLM_PORT

echo "Running Corag Exp 4: Static Sub-Sub"
python models/corag/src/inference/attack_corag.py \
    --attack_mode static_sub_sub \
    --index_dir $INDEX_DIR \
    --corpus_path $CORPUS_PATH \
    --poisoned_index_dir $POISONED_INDEX_DIR \
    --poisoned_corpus_path $POISONED_CORPUS_PATH \
    --adv_data_path $ADV_DATA_PATH \
    --qid_to_idx_path $QID_TO_IDX_PATH \
    --output_dir results/trajectory_results/corag
