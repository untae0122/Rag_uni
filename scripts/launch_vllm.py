
import os
import sys
import subprocess
import signal
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="vLLM Launcher Wrapper")
    parser.add_argument("--model", type=str, required=True, help="Path to model or model name")
    parser.add_argument("--served_model_name", type=str, required=True, help="Name for the served model")
    parser.add_argument("--port", type=int, required=True, help="Port to run on")
    parser.add_argument("--gpu_device", type=str, required=True, help="CUDA_VISIBLE_DEVICES value (e.g. '0')")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.80, help="GPU memory utilization")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--dtype", type=str, default="auto", help="Data type")
    
    args = parser.parse_args()

    # Set Environment Variable for correct GPU visibility
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    
    # Construct Command
    # python -m vllm.entrypoints.openai.api_server
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--served-model-name", args.served_model_name,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--dtype", args.dtype,
        "--trust-remote-code"
    ]
    
    print(f"[Launcher] Starting vLLM on Port {args.port} (GPU {args.gpu_device})")
    print(f"[Launcher] Command: {' '.join(cmd)}")
    
    # Start Subprocess
    process = subprocess.Popen(cmd, env=env)
    
    # Signal Helper
    def signal_handler(sig, frame):
        print(f"\n[Launcher] Received Signal {sig}. Terminating vLLM process...")
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("[Launcher] Process did not terminate, killing...")
                process.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Wait for process
    try:
        process.wait()
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)
        
    print(f"[Launcher] vLLM Process exited with code {process.returncode}")
    sys.exit(process.returncode)

if __name__ == "__main__":
    main()
