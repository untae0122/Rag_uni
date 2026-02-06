#!/bin/bash
# Example script to run ReAct agent
# Usage: ./scripts/run_react.sh

# Environment Variables
export CUDA_VISIBLE_DEVICES=3

# Paths
MODEL_PATH="/home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe" # Replace with your local model path if needed
RETRIEVAL_MODEL="/home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6" # Can be local path

DATA_PATH="./datasets/hotpotqa/hotpotqa.json"
CORPUS_PATH="./datasets/hotpotqa_corpus/hotpotqa_corpus.jsonl"

POISON_CORPUS_PATH="./datasets/hotpotqa_poisoned_corpus/REACT_CHANWOO_ours_vllm_납치3회.jsonl"
OUTPUT_PATH="./results/react/base_results.json"
# Run
python3 scripts/run_attack.py \
    --model react \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --retrieval_model_name "$RETRIEVAL_MODEL" \
    --max_new_tokens 100 \
    # --poisoned_corpus_path "$POISON_CORPUS_PATH" \
    # --dry_run # Uncomment for testing
