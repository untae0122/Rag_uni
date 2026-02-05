#!/bin/bash
# Example script to run WebThinker agent
# Note: Requires a running vLLM server compatible with OpenAI API
# Usage: ./scripts/run_webthinker.sh

# Environment Variables
# export OPENAI_API_KEY="EMPTY" # If needed

# Paths
DATA_PATH="./datasets/hotpotqa/test.jsonl"
OUTPUT_PATH="./results/webthinker/results.json"
CORPUS_PATH="./datasets/hotpotqa/corpus.jsonl"
RETRIEVAL_MODEL="intfloat/e5-large-v2" # Can be local path

# VLLM Server Config
API_BASE="http://localhost:8000/v1"
MODEL_NAME="QwQ-32B-Preview" # Main reasoning model
AUX_MODEL_NAME="Qwen2.5-72B-Instruct" # Aux model from same or different endpoint

# Run
python3 scripts/run_attack.py \
    --model webthinker \
    --api_base_url "$API_BASE" \
    --model_name "$MODEL_NAME" \
    --aux_api_base_url "$API_BASE" \
    --aux_model_name "$AUX_MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --retrieval_model_name "$RETRIEVAL_MODEL" \
    --max_search_limit 5 \
    --concurrent_limit 10 \
    # --dry_run # Uncomment for testing
