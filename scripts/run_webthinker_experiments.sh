#!/bin/bash
# run_webthinker_experiments.sh
# Usage: ./scripts/run_webthinker_experiments.sh [MODE] [GPU_ID]

MODE=$1
GPU_ID=${2:-0}

# WebThinker requires OpenAI API style config for Agent (run via vLLM)
AGENT_API_BASE="http://localhost:8000/v1"
AGENT_API_KEY="EMPTY"
AGENT_MODEL_NAME="webthinker" # Check your vLLM serving name
TOKENIZER_PATH="/share/project/llm/QwQ-32B" # Replace with actual path or handle by name

# Aux Model Config (For Search Intent, etc.)
AUX_API_BASE="http://localhost:8000/v1" # Can be same as Agent
AUX_API_KEY="EMPTY"
AUX_MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" # Usually a strong generalist model
AUX_TOKENIZER_PATH="/share/project/llm/Qwen2.5-72B-Instruct"

# Attacker Config (WebThinker uses 'aux' logic or remote config)
ATTACKER_API_BASE="http://localhost:8001/v1"
ATTACKER_API_KEY="EMPTY"
ATTACKER_MODEL_NAME="attacker"

# Paths
INDEX_DIR="datasets/hotpotqa/e5_index"
CORPUS_PATH="datasets/hotpotqa/corpus.jsonl"
ADV_DATA_PATH="datasets/hotpotqa/hotpotqa_100.json" # Env var or arg

export CUDA_VISIBLE_DEVICES=$GPU_ID
export ADV_DATA_PATH=$ADV_DATA_PATH # ReAct style fallback

echo "Running WebThinker Experiment..."
echo "Mode: $MODE"

# Common Args
COMMON_ARGS="--dataset_name hotpotqa_attack \
    --api_base_url $AGENT_API_BASE \
    --api_key $AGENT_API_KEY \
    --model_name $AGENT_MODEL_NAME \
    --tokenizer_path $TOKENIZER_PATH \
    --aux_api_base_url $AUX_API_BASE \
    --aux_api_key $AUX_API_KEY \
    --aux_model_name $AUX_MODEL_NAME \
    --aux_tokenizer_path $AUX_TOKENIZER_PATH \
    --search_engine e5 \
    --index_dir $INDEX_DIR \
    --corpus_path $CORPUS_PATH \
    --adv_data_path $ADV_DATA_PATH \
    --top_k 5"

if [ "$MODE" == "base" ]; then
    python models/webthinker/scripts/run_web_thinker_for_attack.py \
        --attack_mode base \
        $COMMON_ARGS
        
elif [ "$MODE" == "dynamic" ]; then
    CMD="python models/webthinker/scripts/run_web_thinker_for_attack.py \
        --attack_mode dynamic_retrieval \
        $COMMON_ARGS"
        
    if [ ! -z "$ATTACKER_API_BASE" ]; then
        CMD="$CMD --attacker_api_base $ATTACKER_API_BASE --attacker_model_name $ATTACKER_MODEL_NAME"
    fi
    $CMD

elif [ "$MODE" == "oracle" ]; then
    CMD="python models/webthinker/scripts/run_web_thinker_for_attack.py \
        --attack_mode oracle_injection \
        $COMMON_ARGS"
        
    if [ ! -z "$ATTACKER_API_BASE" ]; then
        CMD="$CMD --attacker_api_base $ATTACKER_API_BASE --attacker_model_name $ATTACKER_MODEL_NAME"
    fi
    $CMD

else
    echo "Unknown mode: $MODE"
fi
