#!/bin/bash
# run_webthinker_experiments.sh
# Usage: ./scripts/run_webthinker_experiments.sh [MODE] [GPU_ID]

MODE=$1
GPU_ID=${2:-0}

# WebThinker requires OpenAI API style config for Agent (run via vLLM)
AGENT_API_BASE="http://localhost:8000/v1"
AGENT_API_KEY="EMPTY"
AGENT_MODEL_NAME="webthinker" # Check your vLLM serving name

# Attacker Config (WebThinker uses 'aux' logic or remote config)
ATTACKER_API_BASE="http://localhost:8001/v1"
ATTACKER_API_KEY="EMPTY"
ATTACKER_MODEL_NAME="attacker"

# Paths
INDEX_DIR="datasets/hotpotqa/e5_index"
CORPUS_PATH="datasets/hotpotqa/corpus.jsonl"
DATA_PATH="datasets/hotpotqa/hotpotqa100.json"

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Running WebThinker Experiment..."
echo "Mode: $MODE"

if [ "$MODE" == "base" ]; then
    python models/webthinker/scripts/run_web_thinker_for_attack.py \
        --attack_mode base \
        --dataset_name hotpotqa_attack \
        --api_base_url $AGENT_API_BASE \
        --api_key $AGENT_API_KEY \
        --model_name $AGENT_MODEL_NAME \
        --search_engine e5 \
        --index_dir $INDEX_DIR \
        --corpus_path $CORPUS_PATH \
        --top_k 5
        
elif [ "$MODE" == "dynamic" ]; then
    CMD="python models/webthinker/scripts/run_web_thinker_for_attack.py \
        --attack_mode dynamic_retrieval \
        --dataset_name hotpotqa_attack \
        --api_base_url $AGENT_API_BASE \
        --api_key $AGENT_API_KEY \
        --model_name $AGENT_MODEL_NAME \
        --search_engine e5 \
        --index_dir $INDEX_DIR \
        --corpus_path $CORPUS_PATH \
        --top_k 5"
        
    if [ ! -z "$ATTACKER_API_BASE" ]; then
        CMD="$CMD --attacker_api_base $ATTACKER_API_BASE --attacker_model_name $ATTACKER_MODEL_NAME"
    fi
    $CMD

else
    echo "Unknown mode: $MODE"
fi
