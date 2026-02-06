
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
from models.corag import CoRagModel
from src.retrieval import indexer
from src.evaluation.metrics import f1_score, exact_match_score

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
    parser.add_argument("--retrieval_model_name", type=str, default="intfloat/e5-large-v2", help="Path or name of retrieval model (E5)")
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
    # Setup Output Directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # [NEW] Handle Auto-Indexing Logic Here
    if args.search_engine == "e5":
        # 1. Clean Index
        if args.corpus_path:
            # If index_dir not provided, derive it
            if not args.index_dir:
                args.index_dir = indexer.derive_index_path(args.corpus_path, args.retrieval_model_name)
                print(f"Auto-derived index_dir: {args.index_dir}")
            
            # Build index if not exists
            indexer.build_index(
                corpus_path=args.corpus_path, 
                index_dir=args.index_dir, 
                model_name=args.retrieval_model_name
            )
        
        # 2. Poisoned Index (if applicable)
        if args.poisoned_corpus_path:
            if not args.poisoned_index_dir:
                args.poisoned_index_dir = indexer.derive_index_path(args.poisoned_corpus_path, args.retrieval_model_name)
                print(f"Auto-derived poisoned_index_dir: {args.poisoned_index_dir}")
                
            indexer.build_index(
                corpus_path=args.poisoned_corpus_path, 
                index_dir=args.poisoned_index_dir, 
                model_name=args.retrieval_model_name
            )

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
        
        # Let's align on 'question' key and preserve 'answer' as 'ground_truth_answer'
        new_item = item.copy()
        new_item['question'] = q
        new_item['ground_truth_answer'] = item.get('answer', '') # Preserve GS
        standardized_data.append(new_item)
        
    data = standardized_data

    # Instantiate Model
    print(f"Initializing Model: {args.model}")
    if args.model == "react":
        # ReAct Agent (Runs vLLM locally)
        agent = ReActAgent(
            args,
            model_name=args.model_path
        )
        # ReAct run_batch takes list of dicts/questions
        results = agent.run_batch(
            inputs=data, 
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

    # Calculate Metrics and Update Results with Correct Scores
    print(f"Calculating Metrics over {len(results)} samples...")
    ems = []
    f1s = []
    
    for item in results:
        # Determine prediction and ground truth keys
        # ReAct: 'answer' (from info dict) - output from model
        # WebThinker: likely 'answer' or 'pred'
        # CoRag: 'pred'
        
        # Heuristic to find prediction
        prediction = item.get('answer', item.get('pred', item.get('prediction', '')))
        
        # Ground Truth preserved in 'ground_truth_answer'
        ground_truth = item.get('ground_truth_answer', item.get('golden_answer', ''))
        
        if prediction is None:
             prediction = ""
             
        # Calculate Metrics
        em = exact_match_score(prediction, ground_truth)
        f1, prec, recall = f1_score(prediction, ground_truth)
        
        # Update item with correct metrics (overwrite potentially wrong wrapper metrics)
        # Update item with correct metrics (overwrite potentially wrong wrapper metrics)
        # item['em'] = check_accuracy(prediction, ground_truth)  <-- Caused NameError
        # Since we already calculated em above, just use it.
        # actually check_accuracy calls exact_match_score.
        # But 'em' key in wrapper was boolean or int? Wrapper: int(score).
        item['em'] = int(em)
        item['f1'] = f1
        item['reward'] = int(em) # usually reward is EM
        
        ems.append(int(em))
        f1s.append(f1)

    # Calculate Poisoning Metrics
    total_poisoned_docs = 0
    total_docs = 0
    samples_with_poison = 0
    
    for item in results:
        sample_has_poison = False
        
        # 0. Check for pre-aggregated stats (ReAct / E5WikiEnv)
        if 'cumulative_poisoned_count' in item and 'cumulative_total_count' in item:
            p_count = item['cumulative_poisoned_count']
            t_count = item['cumulative_total_count']
            total_poisoned_docs += p_count
            total_docs += t_count
            
            if item.get('any_poisoned', False) or p_count > 0:
                samples_with_poison += 1
            if item.get('any_poisoned', False): # Double check flag
                samples_with_poison = samples_with_poison # Already incremented
            
            continue # Skip trace extraction for this item

        # Extract retrieval trace based on model structure
        retrieved_docs_lists = []
        
        if 'step_stats' in item: # WebThinker
            for step in item['step_stats']:
                if step.get('is_search', False) and step.get('search_documents'):
                     retrieved_docs_lists.append(step['search_documents'])
        elif 'steps' in item: # CoRag
             for step in item['steps']:
                 if step.get('retrieved_results'):
                     retrieved_docs_lists.append(step['retrieved_results'])
        elif 'retrieval_history' in item: # ReAct (New)
             # ReAct might return a list of lists (steps) or flat list. 
             # Let's assume list of search results per step.
             for step_result in item['retrieval_history']:
                 retrieved_docs_lists.append(step_result)

        # Calculate stats for this sample
        for docs in retrieved_docs_lists:
            for doc in docs:
                total_docs += 1
                if doc.get('is_poisoned', False):
                    total_poisoned_docs += 1
                    sample_has_poison = True
        
        if sample_has_poison:
            samples_with_poison += 1

    avg_poisoned_ratio = (total_poisoned_docs / total_docs) if total_docs > 0 else 0.0
    poisoned_retrieval_rate = (samples_with_poison / len(results)) if len(results) > 0 else 0.0

    avg_em = np.mean(ems) if ems else 0.0
    avg_f1 = np.mean(f1s) if f1s else 0.0
    
    print(f"Results Summary:")
    print(f"Samples: {len(results)}")
    print(f"Average EM: {avg_em:.4f}")
    print(f"Average F1: {avg_f1:.4f}")
    print(f"Avg Poisoned Ratio: {avg_poisoned_ratio:.4f}")
    print(f"Poisoned Retrieval Rate: {poisoned_retrieval_rate:.4f}")

    # Construct final output with summary
    final_output = {
        "metrics": {
            "average_em": float(f"{avg_em:.4f}"),
            "average_f1": float(f"{avg_f1:.4f}"),
            "avg_poisoned_ratio": float(f"{avg_poisoned_ratio:.4f}"),
            "poisoned_retrieval_rate": float(f"{poisoned_retrieval_rate:.4f}"),
            "total_samples": len(results)
        },
        "data": results
    }

    # Save Results
    print(f"Saving results to {args.output_path}")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    print("Done.")

if __name__ == "__main__":
    main()
