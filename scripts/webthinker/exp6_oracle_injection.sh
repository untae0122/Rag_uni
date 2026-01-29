#!/bin/bash
# WebThinker Experiment 6: Oracle Injection (Oracle-Hard)
# Usage: ./scripts/webthinker/exp6_oracle_injection.sh [GPU_ID]

GPU_ID=${1:-0}
INDEX_DIR="datasets/hotpotqa/e5_index"
CORPUS_PATH="datasets/hotpotqa/corpus.jsonl"
AGENT_API_BASE="http://localhost:8000/v1"
AGENT_API_KEY="EMPTY"
AGENT_MODEL_NAME="webthinker" 
ATTACKER_API_BASE="http://localhost:8001/v1"
ATTACKER_MODEL_NAME="attacker"

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Running WebThinker Exp 6: Oracle Injection"
python models/webthinker/scripts/run_web_thinker_for_attack.py \
    --attack_mode oracle_injection \
    --dataset_name hotpotqa_attack \
    --api_base_url $AGENT_API_BASE \
    --api_key $AGENT_API_KEY \
    --model_name $AGENT_MODEL_NAME \
    --search_engine e5 \
    --index_dir $INDEX_DIR \
    --corpus_path $CORPUS_PATH \
    --top_k 5 \
    --attacker_api_base $ATTACKER_API_BASE \
    --attacker_model_name $ATTACKER_MODEL_NAME
