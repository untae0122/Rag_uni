#!/bin/bash
# Example script to run WebThinker agent
# Note: Requires a running vLLM server compatible with OpenAI API
# Usage: ./scripts/run_webthinker.sh


# Set Trap to kill background processes (vLLM launchers) on exit
trap 'kill $(jobs -p)' EXIT

# Paths
DATA_PATH="./datasets/hotpotqa/hotpotqa.json"
CORPUS_PATH="./datasets/hotpotqa_corpus/hotpotqa_corpus.jsonl"
RETRIEVAL_MODEL="/home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6" # Can be local path

# vLLM Configuration as provided by user
# Main Model: WebThinker (GPU 0)
MAIN_MODEL_PATH="/home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--lixiaoxi45--WebThinker-R1-7B/snapshots/e9bfc92a6dd649a6f4130972b0fd9aceb707e145"
MAIN_MODEL_NAME="WebThinker-R1-7B"
MAIN_PORT=8000
MAIN_GPU="0"

# Aux Model: Qwen (GPU 1)
AUX_MODEL_PATH="/home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
AUX_MODEL_NAME="Qwen2.5-7B-Instruct"
AUX_PORT=8001
AUX_GPU="1"

OUTPUT_PATH="./results/webthinker/base_results.json"

echo "========================================================"
echo "Starting vLLM Orchestration for WebThinker"
echo "========================================================"

# Launch Main Model
python3 scripts/launch_vllm.py \
    --model "$MAIN_MODEL_PATH" \
    --served_model_name "$MAIN_MODEL_NAME" \
    --port "$MAIN_PORT" \
    --gpu_device "$MAIN_GPU" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.8 \
    > logs/vllm_main.log 2>&1 &
PID_MAIN=$!
echo "[Script] Launched Main Model (PID $PID_MAIN). Logs: logs/vllm_main.log"

# Launch Aux Model
python3 scripts/launch_vllm.py \
    --model "$AUX_MODEL_PATH" \
    --served_model_name "$AUX_MODEL_NAME" \
    --port "$AUX_PORT" \
    --gpu_device "$AUX_GPU" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.8 \
    > logs/vllm_aux.log 2>&1 &
PID_AUX=$!
echo "[Script] Launched Aux Model (PID $PID_AUX). Logs: logs/vllm_aux.log"

# Wait for Servers
echo "[Script] Waiting for servers to be ready..."
python3 scripts/wait_for_servers.py \
    --urls "http://localhost:$MAIN_PORT/v1" "http://localhost:$AUX_PORT/v1" \
    --timeout 600

if [ $? -ne 0 ]; then
    echo "[Script] Failed to start servers. Exiting."
    exit 1
fi

echo "[Script] Servers Ready! Starting Attack..."


# Run Attack
# User requested Retriever to be on the same GPU as Main Model
export CUDA_VISIBLE_DEVICES="$MAIN_GPU"

python3 scripts/run_attack.py \
    --model webthinker \
    --api_base_url "http://localhost:$MAIN_PORT/v1" \
    --model_name "$MAIN_MODEL_NAME" \
    --aux_api_base_url "http://localhost:$AUX_PORT/v1" \
    --aux_model_name "$AUX_MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --retrieval_model_name "$RETRIEVAL_MODEL" \
    --max_search_limit 3 \
    --concurrent_limit 10

echo "[Script] Finished. Cleaning up..."
