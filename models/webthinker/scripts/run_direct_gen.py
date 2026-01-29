import csv
import json
import random
import re
import os, time
import asyncio
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from evaluate.evaluate import run_evaluation
from prompts.prompts import (
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code,
)
import argparse
from openai import AsyncOpenAI
from typing import List, Dict
import aiohttp

def parse_args():
    parser = argparse.ArgumentParser(description="Run direct generation for various datasets and models.")
    
    parser.add_argument(
        '--dataset_name', 
        type=str, 
        required=True, 
        help="Name of the dataset to use."
    )
    
    parser.add_argument(
        '--split', 
        type=str, 
        required=True, 
        help="Dataset split to use."
    )
    
    parser.add_argument(
        '--subset_num', 
        type=int, 
        default=-1, 
        help="Number of examples to process. Defaults to all if not specified."
    )
    
    parser.add_argument(
        '--api_base_url',
        type=str,
        required=True,
        help="Base URL for the API endpoint"
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        default="QwQ-32B",
        help="Name of the model to use"
    )
    
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.7, 
        help="Sampling temperature."
    )
    
    parser.add_argument(
        '--top_p', 
        type=float, 
        default=0.8, 
        help="Top-p sampling parameter."
    )
    
    parser.add_argument(
        '--top_k_sampling', 
        type=int, 
        default=20, 
        help="Top-k sampling parameter."
    )
    
    parser.add_argument(
        '--repetition_penalty', 
        type=float, 
        default=None, 
        help="Repetition penalty. If not set, defaults based on the model."
    )
    
    parser.add_argument(
        '--max_tokens', 
        type=int, 
        default=32768, 
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    parser.add_argument(
        '--eval',
        action='store_true',
        help="Whether to run evaluation after generation."
    )

    parser.add_argument(
        '--concurrent_limit',
        type=int,
        default=50,
        help="Maximum number of concurrent API calls"
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Add new arguments for document processing
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help="Number of top search results to retrieve."
    )
    
    parser.add_argument(
        '--max_doc_len',
        type=int,
        default=3000,
        help="Maximum length of each searched document."
    )

    parser.add_argument(
        '--api_key',
        type=str,
        default="empty",
        help="API key for authentication"
    )

    return parser.parse_args()

async def generate_response(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model_name: str,
    top_k_sampling: int = 20,
    repetition_penalty: float = None,
    retry_limit: int = 3,
) -> str:
    for attempt in range(retry_limit):
        try:
            async with semaphore:
                messages = [{"role": "user", "content": prompt}]
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    extra_body={
                        'top_k': top_k_sampling,
                        'include_stop_str_in_output': True,
                        'repetition_penalty': repetition_penalty,
                        # 'min_p': min_p
                    },
                    timeout=2500,
                )
                return response.choices[0].message.content
        except Exception as e:
            if attempt == retry_limit - 1:
                print(f"Failed after {retry_limit} attempts: {e}")
                return ""
            if "maximum context length" in str(e):
                max_tokens = min(max_tokens, 32768 - 1000 * (attempt + 1))
                continue
            await asyncio.sleep(1 * (attempt + 1))
    return ""

async def generate_all_responses(
    client: AsyncOpenAI,
    prompts: List[str],
    concurrent_limit: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model_name: str,
    top_k_sampling: int = 20,
    repetition_penalty: float = None,
) -> List[str]:
    """Generate all responses concurrently with a limit"""
    semaphore = asyncio.Semaphore(concurrent_limit)
    
    # Create tasks with their index to maintain order
    tasks = [
        generate_response(
            client, prompt, semaphore, temperature, top_p, max_tokens, model_name,
            top_k_sampling=top_k_sampling,
            repetition_penalty=repetition_penalty,
        )
        for prompt in prompts
    ]
    
    # Use asyncio.gather to maintain order of results
    with tqdm(total=len(tasks)) as pbar:
        # Create a progress tracking callback
        async def track_progress(task):
            result = await task
            pbar.update(1)
            return result
            
        # Wrap each task with the progress tracker
        tracked_tasks = [track_progress(task) for task in tasks]
        
        # Gather all results in order
        responses = await asyncio.gather(*tracked_tasks)
    
    return responses

async def main_async():
    args = parse_args()
    
    # Set random seed
    if args.seed is None:
        args.seed = int(time.time())
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base_url,
    )
    
    dataset_name = args.dataset_name.lower()
    split = args.split
    subset_num = args.subset_num
    model_name = args.model_name
    temperature = args.temperature
    top_p = args.top_p
    max_tokens = args.max_tokens

    # Paths to datasets
    if dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'gaia', 'hle', 'limo', 'nr']:
        data_path = f'./data/{dataset_name.upper()}/{split}.json'
    elif dataset_name == 'supergpqa':
        data_path = f'./data/SuperGPQA/{split}.json'
    elif dataset_name == 'livecode':
        data_path = f'./data/LiveCodeBench/{split}.json'
    elif dataset_name == 'openthoughts':
        data_path = f'./data/OpenThoughts/{split}.json'
    elif dataset_name == 'aimo-math':
        data_path = f'./data/AIMO-Math/{split}.json'
    elif dataset_name == 'webwalker':
        data_path = f'./data/WebWalkerQA/{split}.json'    
    elif dataset_name == 'bigmath':
        data_path = f'./data/BigMath/{split}.json'
    elif dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'medmcqa', 'pubhealth']:
        data_path = f'./data/QA_Datasets/{dataset_name}.json'
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    # Load data
    with open(data_path, mode='r', encoding='utf-8') as json_file:
        filtered_data = json.load(json_file)

    # Set model short name for output directory
    if 'qwq' in model_name.lower():
        model_short_name = 'qwq'
    elif 'deepseek' in model_name.lower():
        if 'llama-8b' in model_name.lower():
            model_short_name = 'dpsk-llama-8b'
        elif 'qwen-1.5b' in model_name.lower():
            model_short_name = 'dpsk-qwen-1.5b'
        elif 'qwen-7b' in model_name.lower():
            model_short_name = 'dpsk-qwen-7b'
        elif 'qwen-32b' in model_name.lower():
            model_short_name = 'dpsk-qwen-32b'
        elif 'reasoner' in model_name.lower():
            model_short_name = 'dpsk-r1'
    elif 'sky-t1' in model_name.lower():
        model_short_name = 'sky-t1'
    else:
        model_short_name = model_name.split('/')[-1].lower().replace('-instruct', '')

    # Set output directory
    if model_short_name in ['qwq', 'dpsk-llama-8b', 'dpsk-qwen-1.5b', 'dpsk-qwen-7b', 'dpsk-qwen-32b', 'dpsk-r1', 'sky-t1']:
        if dataset_name in ['math500', 'gpqa', 'supergpqa', 'aime', 'amc', 'livecode', 'openthoughts', 'webwalker', 'supergpqa', 'aimo-math', 'bigmath', 'nr']:
            output_dir = f'./outputs/{dataset_name}.{model_short_name}.direct'
        else:
            output_dir = f'./outputs/runs.qa/{dataset_name}.{model_short_name}.direct'
    else:
        output_dir = f'./outputs/runs.baselines/{dataset_name}.{model_short_name}.direct'
    
    os.makedirs(output_dir, exist_ok=True)
    # Prepare prompts and filter data
    prompts = []
    filtered_data_new = []
    for item in filtered_data:
        question = item['Question']
        user_prompt = ""  # Default value
        
        if dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'webwalker', 'gaia', 'hle', 'webwalker', 'nr']:
            if 'qwq' in model_name.lower() or 'sky-t1' in model_name.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')
            elif 'deepseek' in model_name.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='dpsk')
            else:
                user_prompt = get_task_instruction_openqa(question)
        elif dataset_name in ['math500', 'aime', 'amc', 'aimo-math', 'bigmath']:
            if 'qwq' in model_name.lower() or 'sky-t1' in model_name.lower() or 'deepseek' in model_name.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_math(question)
        elif dataset_name in ['gpqa']:
            if 'qwq' in model_name.lower() or 'sky-t1' in model_name.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
            elif 'deepseek' in model_name.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='dpsk')
            elif 'llama' in model_name.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(question)
        elif dataset_name == 'livecode':
            question_title = item.get('question_title', '')
            if 'qwq' in model_name.lower() or 'deepseek' in model_name.lower() or 'sky-t1' in model_name.lower():
                user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
            else:
                user_prompt = get_task_instruction_code(question)
        elif dataset_name == 'openthoughts':
            domain = item['domain']
            if domain == 'math':
                if 'qwq' in model_name.lower() or 'sky-t1' in model_name.lower() or 'deepseek' in model_name.lower():
                    user_prompt = get_task_instruction_math(question, model_name='qwq')
                else:
                    user_prompt = get_task_instruction_math(question)
            elif domain == 'code':
                question_title = item.get('question_title', '')
                if 'qwq' in model_name.lower() or 'sky-t1' in model_name.lower() or 'deepseek' in model_name.lower():
                    user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
                else:
                    user_prompt = get_task_instruction_code(question)
            elif domain == 'puzzle':
                if 'qwq' in model_name.lower() or 'sky-t1' in model_name.lower():
                    user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
                elif 'deepseek' in model_name.lower():
                    user_prompt = get_task_instruction_multi_choice(question, model_name='dpsk')
                elif 'llama' in model_name.lower():
                    user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
                else:
                    user_prompt = get_task_instruction_multi_choice(question)
        elif dataset_name == 'supergpqa':
            question_type = item['question_type']
            if question_type == 'generation':
                if 'qwq' in model_name.lower() or 'sky-t1' in model_name.lower():
                    user_prompt = get_task_instruction_openqa(question, model_name='qwq')
                elif 'deepseek' in model_name.lower():
                    user_prompt = get_task_instruction_openqa(question, model_name='dpsk')
                elif 'llama' in model_name.lower():
                    user_prompt = get_task_instruction_openqa(question, model_name='llama')
                else:
                    user_prompt = get_task_instruction_openqa(question)
            elif question_type == 'multi-choice':
                if 'qwq' in model_name.lower() or 'sky-t1' in model_name.lower():
                    user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
                elif 'deepseek' in model_name.lower():
                    user_prompt = get_task_instruction_multi_choice(question, model_name='dpsk')
                else:
                    user_prompt = get_task_instruction_multi_choice(question)

        # Add prompt and item to lists
        prompts.append(user_prompt)
        filtered_data_new.append(item)
        item['input'] = user_prompt

    # Replace filtered_data with the new filtered version
    filtered_data = filtered_data_new

    if args.subset_num != -1:
        prompts = prompts[:args.subset_num]
        filtered_data = filtered_data[:args.subset_num]
    
    # Generate outputs using async client
    t_start = time.time()
    output_list = await generate_all_responses(
        client,
        prompts,
        args.concurrent_limit,
        args.temperature,
        args.top_p,
        args.max_tokens,
        args.model_name,
        top_k_sampling=args.top_k_sampling,
        repetition_penalty=args.repetition_penalty,
    )
    total_time = time.time() - t_start
    
    # Run evaluation if --eval flag is set
    if args.eval:
        run_evaluation(
            filtered_data,
            prompts,
            output_list,
            args.dataset_name,
            output_dir,
            total_time,
            args.split,
        )
    else:
        for item, result in zip(filtered_data, output_list):
            item['Output'] = result
        
        t = time.localtime()
        result_json_name = f'{args.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
        # Save prediction results
        with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
            json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()