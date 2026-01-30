#!/bin/bash
# Corag Experiment 3: Static Main-Sub
# Usage: ./scripts/corag/exp3_static_main_sub.sh [GPU_ID]

GPU_ID=${1:-0}
INDEX_DIR="datasets/hotpotqa/e5_index"
CORPUS_PATH="datasets/hotpotqa/corpus.jsonl"
QID_TO_IDX_PATH="datasets/hotpotqa/qid_to_idx.json"
ADV_DATA_PATH="datasets/hotpotqa/hotpotqa100_x3.json"
POISONED_INDEX_DIR="datasets/hotpotqa/poisoned_corpus_main_sub_corag"
POISONED_CORPUS_PATH="datasets/hotpotqa/poisoned_corpus_main_sub_corag.jsonl"
VLLM_HOST="localhost"
VLLM_PORT="8005"

export CUDA_VISIBLE_DEVICES=$GPU_ID
export VLLM_HOST=$VLLM_HOST
export VLLM_PORT=$VLLM_PORT

echo "Running Corag Exp 3: Static Main-Sub"
python models/corag/src/inference/attack_corag.py \
    --attack_mode static_main_sub \
    --index_dir $INDEX_DIR \
    --corpus_path $CORPUS_PATH \
    --poisoned_index_dir $POISONED_INDEX_DIR \
    --poisoned_corpus_path $POISONED_CORPUS_PATH \
    --adv_data_path $ADV_DATA_PATH \
    --qid_to_idx_path $QID_TO_IDX_PATH \
    --output_dir results/trajectory_results/corag
