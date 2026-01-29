
import os
import wikienv, wrappers
import json
import sys
import time
import requests
import argparse
from vllm import LLM, SamplingParams

# [변경됨] 1. Argument Parsing을 최상단으로 이동 (Retriever 초기화 전에 args가 필요함)
parser = argparse.ArgumentParser()
parser.add_argument("--dry_run", action="store_true", help="Run only the first sample")
args = parser.parse_args()

# Initialize vLLM (limit memory to leave room for retriever index)
print("Initializing vLLM model...")
llm_model = LLM(
    model="/home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590", 
    dtype="half", 
    trust_remote_code=True,
    gpu_memory_utilization=0.6 # A6000(48GB) 기준, 인덱스(~11GB)를 고려하여 60%만 사용
)

def llm(prompt, stop=["\n"]):
    # Add extra stop tokens to prevent the model from hallucinating next steps or questions
    custom_stop = stop + ["\nQuestion:", "\nThought", "Observation"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=100, stop=custom_stop)
    outputs = llm_model.generate([prompt], sampling_params, use_tqdm=False)
    return outputs[0].outputs[0].text

# Environment Setup
import sys
# Get the base directory (rag-exp) - go up one level from ReAct directory
base_dir = os.path.dirname(os.path.dirname(__file__))

# Add src to path to import E5_Retriever
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from e5_retriever import E5_Retriever
from e5_env import E5WikiEnv

print(">>> MODE: TRAJECTORY COLLECTION (Clean Corpus)")
trajectory_results_dir = '/home/work/Redteaming/rag-exp/results/trajectory_results/react'
results_path = os.path.join(trajectory_results_dir, 'test.json')

# Initialize Retriever
retriever = E5_Retriever(
    index_dir=os.path.join(base_dir, "datasets/hotpotqa/e5_index"), 
    corpus_path=os.path.join(base_dir, "datasets/hotpotqa/corpus.jsonl"),
    poisoned_index_dir=None, 
    poisoned_corpus_path=None,
    model_name="/home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6",
    device="cuda"
)

env = E5WikiEnv(retriever, k=5)
env = wrappers.HotPotQAWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

# ReAct Logic
folder = './prompts/'
# prompt_file = 'prompts_naive.json' # Adjusted path handling below in case running from different dir
prompt_file = os.path.join(os.path.dirname(__file__), 'prompts', 'prompts_naive.json')

with open(prompt_file, 'r') as f:
    prompt_dict = json.load(f)

webthink_examples = prompt_dict['webthink_simple6']
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_examples

def webthink(idx=None, prompt=webthink_prompt, to_print=True):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    step_stats = []  # Step별 통계 수집
    
    for i in range(1, 8):
        n_calls += 1
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
        
        # Robust action parsing: take the first line and truncate after ] to handle hallucinations
        action = action.split('\n')[0].strip()
        if ']' in action:
            action = action[:action.find(']')+1]
        
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        
        step_stats.append({
            'step': i,
            'thought': thought,
            'action': action,
            'observation': obs
        })
        
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        if to_print:
            print(step_str)
        if done:
            break
    
    if not done:
        obs, r, done, info = step(env, "finish[]")
    
    if to_print:
        print(info, '\n')
    info.update({
        'n_calls': n_calls, 
        'n_badcalls': n_badcalls, 
        'traj': prompt,
        'step_stats': step_stats  # Step별 통계 추가
    })
    return r, info

# Main Loop
if __name__ == "__main__":
    # Load HotpotQA dev data
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'hotpot_dev_v1_simplified.json')
    
    with open(data_path, 'r') as f:
        dev_data = json.load(f)

    # Limit number of samples if dry_run
    if args.dry_run:
        dev_data = dev_data[:10]
        print("Dry run enabled: only the first 50 samples will be processed.")

    results = []
    total_count = 0
    start_time = time.time()

    for idx, item in enumerate(dev_data):
        total_count += 1
        question_text = item['question']
        correct_answer = item['answer']
        
        print(f"\n[{total_count}/{len(dev_data)}] Index: {idx}")
        print(f"Question: {question_text}")
        print(f"Correct Answer: {correct_answer}")
        
        try:
            r, info = webthink(idx=idx, to_print=True)
            llm_answer = info.get('answer', '')
            
            # Save thought, observation, action path for this question
            step_stats = info.get('step_stats', [])
            
            # Build steps array
            steps = []
            for step_stat in step_stats:
                step_data = {
                    'thought': step_stat.get('thought', ''),
                    'action': step_stat.get('action', ''),
                    'observation': step_stat.get('observation', '')
                }
                steps.append(step_data)
            
            os.makedirs(trajectory_results_dir, exist_ok=True)
            
            question_path_data = {
                'id': idx,
                'query': question_text,
                'correct_answer': correct_answer,
                'steps': steps,
                'final_answer': llm_answer
            }
            
            # Load existing results, append new data, and overwrite file
            all_results = []
            if os.path.exists(results_path):
                try:
                    with open(results_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            all_results = data
                except (json.JSONDecodeError, ValueError):
                    all_results = []
            
            all_results.append(question_path_data)
            
            # Overwrite file with updated list
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
            
            results.append({
                "idx": idx,
                "question": question_text,
                "correct_answer": correct_answer,
                "llm_answer": llm_answer,
                "step_stats": info.get('step_stats', [])
            })
            
            print(f"LLM Answer: {llm_answer}")
            print("-" * 20)
            
        except Exception as e:
            print(f"Error processing question {idx}: {e}")
            continue

    # Final Summary
    end_time = time.time()
    
    print("\n" + "="*50)
    print("TRAJECTORY COLLECTION COMPLETED")
    print(f"Total Questions: {total_count}")
    print(f"Total Time: {end_time - start_time:.2f}s")
    print("="*50)
    print(f"Results saved to {results_path}")
