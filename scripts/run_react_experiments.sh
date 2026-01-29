#!/bin/bash
# run_react_experiments.sh
# Usage: ./scripts/run_react_experiments.sh [MODE] [GPU_ID]
# Modes: base, static, dynamic, oracle, surrogate
# Example: ./scripts/run_react_experiments.sh base 0

MODE=$1
GPU_ID=${2:-0}

# Common Configs
# NOTE: Update paths to match your environment
MODEL_PATH="/home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
RETRIEVER_MODEL="/home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6"
INDEX_DIR="datasets/hotpotqa/e5_index"
CORPUS_PATH="datasets/hotpotqa/corpus.jsonl"
DATA_PATH="datasets/hotpotqa/hotpotqa100.json"
POISONED_INDEX_DIR="datasets/hotpotqa/poisoned_index" # Example
POISONED_CORPUS_PATH="datasets/hotpotqa/poisoned_corpus.jsonl" # Example

# Remote Attacker Config (For Dynamic/Oracle Modes)
# If using remote vLLM, set these:
ATTACKER_API_BASE="http://localhost:8001/v1"
ATTACKER_MODEL_NAME=$MODEL_PATH
# If using local reuse (1 GPU), unset API base:
# ATTACKER_API_BASE=""

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Running ReAct Experiment..."
echo "Mode: $MODE"
echo "GPU: $GPU_ID"

if [ "$MODE" == "base" ]; then
    python models/react/attack_react.py \
        --attack_mode base \
        --model_path $MODEL_PATH \
        --index_dir $INDEX_DIR \
        --corpus_path $CORPUS_PATH \
        --data_path $DATA_PATH \
        --output_dir results/trajectory_results/react
        
elif [ "$MODE" == "static" ]; then
    # Runs Experiment 2 (Static Main-Main) as example
    python models/react/attack_react.py \
        --attack_mode static_main_main \
        --model_path $MODEL_PATH \
        --index_dir $INDEX_DIR \
        --corpus_path $CORPUS_PATH \
        --poisoned_index_dir $POISONED_INDEX_DIR \
        --poisoned_corpus_path $POISONED_CORPUS_PATH \
        --data_path $DATA_PATH \
        --output_dir results/trajectory_results/react

elif [ "$MODE" == "dynamic" ]; then
    # Experiment 5: Dynamic Retrieval (Oracle-Soft)
    # Requires Attacker
    CMD="python models/react/attack_react.py \
        --attack_mode dynamic_retrieval \
        --model_path $MODEL_PATH \
        --index_dir $INDEX_DIR \
        --corpus_path $CORPUS_PATH \
        --data_path $DATA_PATH \
        --output_dir results/trajectory_results/react"
        
    if [ ! -z "$ATTACKER_API_BASE" ]; then
        CMD="$CMD --attacker_api_base $ATTACKER_API_BASE --attacker_model_name $ATTACKER_MODEL_NAME"
    fi
    
    $CMD

elif [ "$MODE" == "oracle" ]; then
    # Experiment 6: Oracle Injection (Oracle-Hard)
    CMD="python models/react/attack_react.py \
        --attack_mode oracle_injection \
        --model_path $MODEL_PATH \
        --index_dir $INDEX_DIR \
        --corpus_path $CORPUS_PATH \
        --data_path $DATA_PATH \
        --output_dir results/trajectory_results/react"
        
    if [ ! -z "$ATTACKER_API_BASE" ]; then
        CMD="$CMD --attacker_api_base $ATTACKER_API_BASE --attacker_model_name $ATTACKER_MODEL_NAME"
    fi
    
    $CMD

else
    echo "Unknown mode: $MODE"
    echo "Available: base, static, dynamic, oracle"
fi
