#!/bin/bash
# WebThinker Experiment 2: Static Main-Main
# Usage: ./scripts/webthinker/exp2_static_main_main.sh [GPU_ID]

GPU_ID=${1:-0}
INDEX_DIR="datasets/hotpotqa/e5_index"
CORPUS_PATH="datasets/hotpotqa/corpus.jsonl"
POISONED_INDEX_DIR="datasets/hotpotqa/poisoned_index_exp2"
POISONED_CORPUS_PATH="datasets/hotpotqa/poisoned_corpus_exp2.jsonl"
AGENT_API_BASE="http://localhost:8000/v1"
AGENT_API_KEY="EMPTY"
AGENT_MODEL_NAME="webthinker" 
# [Unified] Define ADV_DATA_PATH
ADV_DATA_PATH="datasets/hotpotqa/hotpotqa100_x3.json"
QID_TO_IDX_PATH="datasets/hotpotqa/qid_to_idx.json" 

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Running WebThinker Exp 2: Static Main-Main"
python models/webthinker/scripts/run_web_thinker_for_attack.py \
    --attack_mode static_main_main \
    --dataset_name hotpotqa_attack \
    --api_base_url $AGENT_API_BASE \
    --api_key $AGENT_API_KEY \
    --model_name $AGENT_MODEL_NAME \
    --search_engine e5 \
    --index_dir $INDEX_DIR \
    --corpus_path $CORPUS_PATH \
    --poisoned_index_dir $POISONED_INDEX_DIR \
    --poisoned_corpus_path $POISONED_CORPUS_PATH \
    --top_k 5
    --qid_to_idx_path $QID_TO_IDX_PATH \
    --adv_data_path $ADV_DATA_PATH \
