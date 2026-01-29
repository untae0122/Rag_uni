#!/bin/bash
# run_corag_experiments.sh
# Usage: ./scripts/run_corag_experiments.sh [MODE] [GPU_ID]

MODE=$1
GPU_ID=${2:-0}

# Common Configs
# NOTE: Update paths to match your environment
MODEL_PATH="/home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
INDEX_DIR="datasets/hotpotqa/e5_index"
CORPUS_PATH="datasets/hotpotqa/corpus.jsonl"
POISONED_INDEX_DIR="datasets/hotpotqa/poisoned_index" 
POISONED_CORPUS_PATH="datasets/hotpotqa/poisoned_corpus.jsonl"

# VLLM Client Config (Corag Agent uses this)
VLLM_HOST="localhost"
VLLM_PORT="8000"

# Remote Attacker Config (For Dynamic/Oracle Modes)
ATTACKER_API_BASE="http://localhost:8001/v1"
ATTACKER_MODEL_NAME=$MODEL_PATH
# ATTACKER_API_BASE="" # Uncomment for local load (Caution: OOM risk if VLLM Client is also local)

export CUDA_VISIBLE_DEVICES=$GPU_ID
export VLLM_HOST=$VLLM_HOST
export VLLM_PORT=$VLLM_PORT

echo "Running Corag Experiment..."
echo "Mode: $MODE"
echo "GPU: $GPU_ID"
echo "Agent VLLM: $VLLM_HOST:$VLLM_PORT"

if [ "$MODE" == "base" ]; then
    python models/corag/src/inference/attack_corag.py \
        --attack_mode base \
        --base \
        --index_dir $INDEX_DIR \
        --corpus_path $CORPUS_PATH \
        --output_dir results/trajectory_results/corag
        
elif [ "$MODE" == "static" ]; then
    python models/corag/src/inference/attack_corag.py \
        --attack_mode static_main_main \
        --index_dir $INDEX_DIR \
        --corpus_path $CORPUS_PATH \
        --poisoned_index_dir $POISONED_INDEX_DIR \
        --poisoned_corpus_path $POISONED_CORPUS_PATH \
        --output_dir results/trajectory_results/corag

elif [ "$MODE" == "dynamic" ]; then
    CMD="python models/corag/src/inference/attack_corag.py \
        --attack_mode dynamic_retrieval \
        --index_dir $INDEX_DIR \
        --corpus_path $CORPUS_PATH \
        --output_dir results/trajectory_results/corag"
        
    if [ ! -z "$ATTACKER_API_BASE" ]; then
        CMD="$CMD --attacker_api_base $ATTACKER_API_BASE --attacker_model_name $ATTACKER_MODEL_NAME"
    fi
    $CMD

elif [ "$MODE" == "oracle" ]; then
    CMD="python models/corag/src/inference/attack_corag.py \
        --attack_mode oracle_injection \
        --index_dir $INDEX_DIR \
        --corpus_path $CORPUS_PATH \
        --output_dir results/trajectory_results/corag"
        
    if [ ! -z "$ATTACKER_API_BASE" ]; then
        CMD="$CMD --attacker_api_base $ATTACKER_API_BASE --attacker_model_name $ATTACKER_MODEL_NAME"
    fi
    $CMD

else
    echo "Unknown mode: $MODE"
fi
