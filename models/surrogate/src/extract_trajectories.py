import json
import os
import glob
import argparse
from typing import List, Dict

def extract_trajectories(log_paths: List[str], output_path: str):
    """
    Extracts (Prompt -> Subquery) pairs from interaction logs.
    """
    data_pairs = []
    total_files = 0
    total_examples = 0
    
    print(f"Searching for logs in: {log_paths}")
    
    files = []
    for pattern in log_paths:
        files.extend(glob.glob(pattern))
        
    print(f"Found {len(files)} log files.")
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                
            # Log standard format: { "results": [ ... ] }
            results = content.get('results', [])
            # Also support direct list format
            if isinstance(content, list):
                results = content
                
            for res in results:
                # We need the full history to construct the prompt
                # ReAct: Question -> Thought -> Action -> Obs -> Thought -> ...
                # We want to predict 'Action' given history.
                
                # However, ReAct data in Logger has 'steps'.
                # We can reconstruct the "Prompt at step K" from steps 0..K-1.
                
                question = res.get('query', '') or res.get('question', '')
                steps = res.get('steps', [])
                
                # Build context step by step
                # Format depends on the Model's prompt template.
                # Here we use a generic "Question: ...\nThought: ..." format
                # strictly for the Surrogate Model which we train ourselves.
                
                history_str = f"Question: {question}\n"
                
                for step in steps:
                    thought = step.get('thought', '')
                    action = step.get('action', '')
                    
                    if 'Search[' in action:
                        # This is a training example!
                        # Input: history + thought
                        # Target: action (subquery)
                        
                        prompt = history_str + f"Thought: {thought}\nAction:"
                        target = f" {action}" # " Search[query]"
                        
                        data_pairs.append({
                            "prompt": prompt,
                            "completion": target,
                            "subquery": step.get('subquery', '')
                        })
                        
                        total_examples += 1
                        
                    # Update history for next step
                    # We assume we also feed the observation? 
                    # Yes, to predict step 2, we need step 1's obs.
                    observation = step.get('observation', '')
                    history_str += f"Thought: {thought}\nAction: {action}\nObservation: {observation}\n"
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Save to JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data_pairs:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
            
    print(f"Saved {total_examples} training examples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # accept multiple patterns
    parser.add_argument("--log_patterns", nargs='+', default=["../../results/trajectory_results/**/*.json"], help="Glob patterns for log files")
    parser.add_argument("--output_path", default="../data/surrogate_training_data.jsonl", help="Output JSONL path")
    args = parser.parse_args()
    
    extract_trajectories(args.log_patterns, args.output_path)
