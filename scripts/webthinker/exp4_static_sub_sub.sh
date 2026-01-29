#!/bin/bash
# WebThinker Experiment 4: Static Sub-Sub
# Usage: ./scripts/webthinker/exp4_static_sub_sub.sh [GPU_ID]

GPU_ID=${1:-0}
INDEX_DIR="datasets/hotpotqa/e5_index"
CORPUS_PATH="datasets/hotpotqa/corpus.jsonl"
POISONED_INDEX_DIR="datasets/hotpotqa/poisoned_index_exp4"
POISONED_CORPUS_PATH="datasets/hotpotqa/poisoned_corpus_exp4.jsonl"
AGENT_API_BASE="http://localhost:8000/v1"
AGENT_API_KEY="EMPTY"
AGENT_MODEL_NAME="webthinker" 

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Running WebThinker Exp 4: Static Sub-Sub"
python models/webthinker/scripts/run_web_thinker_for_attack.py \
    --attack_mode static_sub_sub \
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
