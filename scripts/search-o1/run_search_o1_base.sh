#!/bin/bash
# Example script to run Search-o1 agent
# Note: Requires a running vLLM server compatible with OpenAI API
# Usage: ./scripts/run_search_o1_base.sh


# Set Trap to kill background processes (vLLM launchers) on exit
trap 'kill $(jobs -p)' EXIT

# Paths
DATA_PATH="./datasets/hotpotqa/hotpotqa_poisonedrag.json"
CORPUS_PATH="./datasets/hotpotqa_corpus/hotpotqa_corpus.jsonl"
RETRIEVAL_MODEL="intfloat/e5-large-v2" # Can be local path

# vLLM Configuration as provided by user
# Main Model: Search-o1 (GPU 0)
MAIN_MODEL_PATH="Qwen/Qwen2.5-32B-Instruct" # Change this to the specific model you want to test
MAIN_MODEL_NAME="Qwen2.5-32B-Instruct"
MAIN_PORT=8000
MAIN_GPU="0"

OUTPUT_PATH="./results/search_o1/base_results.json"

echo "========================================================"
echo "Starting vLLM Orchestration for Search-o1"
echo "========================================================"

# Launch Main Model
python3 scripts/launch_vllm.py \
    --model "$MAIN_MODEL_PATH" \
    --served_model_name "$MAIN_MODEL_NAME" \
    --port "$MAIN_PORT" \
    --gpu_device "$MAIN_GPU" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    > logs/vllm_search_o1.log 2>&1 &
PID_MAIN=$!
echo "[Script] Launched Main Model (PID $PID_MAIN). Logs: logs/vllm_search_o1.log"

# Wait for Servers
echo "[Script] Waiting for servers to be ready..."
python3 scripts/wait_for_servers.py \
    --urls "http://localhost:$MAIN_PORT/v1" \
    --timeout 600

if [ $? -ne 0 ]; then
    echo "[Script] Failed to start servers. Exiting."
    exit 1
fi

echo "[Script] Servers Ready! Starting Attack..."

# Run Attack
export CUDA_VISIBLE_DEVICES="$MAIN_GPU"

python3 scripts/run_attack.py \
    --model search_o1 \
    --api_base_url "http://localhost:$MAIN_PORT/v1" \
    --model_path "$MAIN_MODEL_PATH" \
    --model_name "$MAIN_MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --corpus_path "$CORPUS_PATH" \
    --retrieval_model_name "$RETRIEVAL_MODEL" \
    --max_search_limit 5 \
    --concurrent_limit 10

echo "[Script] Finished. Cleaning up..."
