#!/bin/bash
# Example script to run ReAct agent
# Usage: ./scripts/run_react.sh

# Environment Variables
export CUDA_VISIBLE_DEVICES=0

# Paths
MODEL_PATH="Qwen/Qwen2.5-32B-Instruct" # Replace with your local model path if needed
DATA_PATH="./datasets/hotpotqa/test.jsonl"
OUTPUT_PATH="./results/react/results.json"
CORPUS_PATH="./datasets/hotpotqa/corpus.jsonl"
RETRIEVAL_MODEL="intfloat/e5-large-v2" # Can be local path

# Run
python3 scripts/run_attack.py \
    --model react \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --retrieval_model_name "$RETRIEVAL_MODEL" \
    --max_new_tokens 512 \
    # --dry_run # Uncomment for testing
