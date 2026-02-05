
import argparse
import os
import sys
import json
import asyncio
import torch
import numpy as np
import random
from typing import List, Dict

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from models.react import ReActAgent
from models.webthinker import WebThinkerAgent
from models.corag import CoRagModel

def parse_args():
    parser = argparse.ArgumentParser(description="Unified Attack Runner for RAG Experiments")
    
    # Common Arguments
    parser.add_argument("--model", type=str, required=True, choices=["react", "webthinker", "corag"], help="Model to run")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input dataset JSON/JSONL")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_samples", type=int, default=-1, help="Max samples to process (for debugging)")
    parser.add_argument("--dry_run", action="store_true", help="Run on a small subset")

    # Retrieval Arguments
    parser.add_argument("--search_engine", type=str, default="e5", choices=["e5", "bing", "serper"], help="Search engine to use")
    parser.add_argument("--index_dir", type=str, default=None, help="Path to E5 index directory")
    parser.add_argument("--corpus_path", type=str, default=None, help="Path to corpus JSONL")
    parser.add_argument("--poisoned_index_dir", type=str, default=None, help="Path to poisoned E5 index")
    parser.add_argument("--poisoned_corpus_path", type=str, default=None, help="Path to poisoned corpus")

    # Model Specific - ReAct (vLLM In-Process)
    parser.add_argument("--model_path", type=str, default=None, help="Path to model weights for ReAct (vLLM)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for generation")
    
    # Model Specific - WebThinker/CoRag (Server Based)
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API Key for vLLM/OpenAI")
    parser.add_argument("--api_base_url", type=str, default="http://localhost:8000/v1", help="Base URL for vLLM/OpenAI")
    parser.add_argument("--model_name", type=str, default="default", help="Model name for API requests")
    parser.add_argument("--aux_model_name", type=str, default="default", help="Auxiliary model name for WebThinker")
    parser.add_argument("--aux_api_key", type=str, default="EMPTY", help="Auxiliary API Key")
    parser.add_argument("--aux_api_base_url", type=str, default="http://localhost:8000/v1", help="Auxiliary Base URL")
    
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Sampling top_p")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens")
    parser.add_argument("--concurrent_limit", type=int, default=10, help="Concurrency limit for async requests")
    
    # WebThinker Specific
    parser.add_argument("--max_search_limit", type=int, default=5, help="Max search steps for WebThinker")
    parser.add_argument("--top_k_sampling", type=int, default=1, help="Top k sampling") # For WebThinker extra_body
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--min_p", type=float, default=0.0, help="Min p")
    parser.add_argument("--top_k", type=int, default=5, help="Retrieval Top K")

    # CoRag Specific
    parser.add_argument("--vllm_host", type=str, default="localhost", help="VLLM Host for CoRag")
    parser.add_argument("--vllm_port", type=int, default=8000, help="VLLM Port for CoRag")
    parser.add_argument("--decode_strategy", type=str, default="greedy", help="CoRag decode strategy")
    parser.add_argument("--max_path_length", type=int, default=3, help="CoRag max path length")
    parser.add_argument("--best_n", type=int, default=4, help="CoRag best of n")
    parser.add_argument("--max_len", type=int, default=4096, help="CoRag max message length")

    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(path, max_samples=-1):
    if path.endswith(".jsonl"):
        with open(path, 'r') as f:
            data = [json.loads(line) for line in f]
    elif path.endswith(".json"):
        with open(path, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                 # Handle cases where json is wrapped or dict
                 data = list(data.values()) if 'data' not in data else data['data']
    else:
        raise ValueError("Unsupported file format")

    if max_samples > 0:
        data = data[:max_samples]
    return data

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup Output Directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load Data
    print(f"Loading data from {args.data_path}")
    data = load_data(args.data_path, args.max_samples if not args.dry_run else 2)
    
    # Preprocessing Input Data specific to Models?
    # Usually standardized to {'question': ..., 'id': ...}
    # ReAct wraps data logic. WebThinker expects 'prompt'.
    # We might need to adapt data to expected keys.
    
    # Standardize data keys locally
    standardized_data = []
    for item in data:
        q = item.get('question', item.get('query', ''))
        # Prompt formation?
        # WebThinker and ReAct have internal prompt logic usually.
        # But WebThinker code expects 'prompt' in input dict in process_single_sequence!
        # ReActAgent.run_batch expects list of questions.
        
        # Let's align on 'question' key.
        new_item = item.copy()
        new_item['question'] = q
        standardized_data.append(new_item)
        
    data = standardized_data

    # Instantiate Model
    print(f"Initializing Model: {args.model}")
    if args.model == "react":
        # ReAct Agent (Runs vLLM locally)
        agent = ReActAgent(
            model_path=args.model_path,
            index_dir=args.index_dir,
            corpus_path=args.corpus_path,
            poisoned_index_dir=args.poisoned_index_dir,
            poisoned_corpus_path=args.poisoned_corpus_path
        )
        # ReAct run_batch takes list of questions/tasks
        # We need to adapt data to ReAct input format if necessary
        # ReActAgent.run_batch seems to take just list of strings? No, let's check.
        # models/react.py: run_batch(self, tasks: List[str])
        
        tasks = [d['question'] for d in data]
        results = agent.run_batch(
            tasks=tasks, 
            max_new_tokens=args.max_new_tokens
        )
        # Results are list of dicts/strings? ReAct returns list of dicts with 'history', 'answer'.
        # Merging with original data
        final_results = []
        for d, r in zip(data, results):
            d.update(r)
            final_results.append(d)
        results = final_results

    elif args.model == "webthinker":
        # WebThinker Agent
        # Requires 'prompt' in input. 
        # WebThinkerAgent.run_batch expects inputs list of dicts with 'prompt'.
        # We need to Construct the prompt! 
        # But wait, models/webthinker.py uses 'prompts' module to get instruction.
        # But process_single_sequence uses seq['prompt'].
        # The prompt construction logic was usually in the MAIN script of WebThinker.
        # I moved `get_gpqa_web_thinker_instruction` etc to `webthinker_utils.prompts`.
        # WE NEED TO CONSTRUCT PROMPT HERE or inside Agent.
        # Ideally Agent should handle it.
        # But `process_single_sequence` takes `seq` which has `prompt`.
        
        # Let's construct prompts here using `webthinker_utils.prompts`.
        from models.webthinker_utils.prompts import get_gpqa_web_thinker_instruction
        instruction = get_gpqa_web_thinker_instruction(MAX_SEARCH_LIMIT=args.max_search_limit)
        
        for d in data:
            # Simple prompt construction: Instruction + Question
            # Or formatted?
            # get_gpqa_web_thinker_instruction returns just instruction text.
            # We append question.
            q = d['question']
            d['prompt'] = f"{instruction}\nQuestion: {q}\n"

        agent = WebThinkerAgent(args)
        # WebThinkerAgent.run_batch is async
        results = asyncio.run(agent.run_batch(data))
        # results is list of dicts (seq)
        
    elif args.model == "corag":
        # CoRag Model
        agent = CoRagModel(args)
        results = agent.run_batch(data)

    # Save Results
    print(f"Saving results to {args.output_path}")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print("Done.")

if __name__ == "__main__":
    main()
