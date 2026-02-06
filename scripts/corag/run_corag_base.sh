#!/bin/bash
# Example script to run CoRag model
# Note: Requires a VLLM server
# Usage: ./scripts/run_corag.sh

# Paths
DATA_PATH="./datasets/hotpotqa/test.jsonl"
CORPUS_PATH="./datasets/hotpotqa/corpus.jsonl"
RETRIEVAL_MODEL="intfloat/e5-large-v2" # Can be local path

# VLLM Config
VLLM_HOST="localhost"
VLLM_PORT=8000
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct" # Model served by VLLM

OUTPUT_PATH="./results/corag/base_results.json"

# Run
python3 scripts/run_attack.py \
    --model corag \
    --vllm_host "$VLLM_HOST" \
    --vllm_port $VLLM_PORT \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --retrieval_model_name "$RETRIEVAL_MODEL" \
    --decode_strategy "best_of_n" \
    --best_n 4 \
    --max_path_length 3 \
    # --dry_run # Uncomment for testing
