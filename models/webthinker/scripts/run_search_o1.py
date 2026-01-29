# run_search_o1.py
import os
import json
import time
import re
from tqdm import tqdm
import numpy as np
import torch
import string
from typing import Optional, Tuple, List, Dict
import argparse
import random
import asyncio

from openai import AsyncOpenAI

from search.bing_search import (
    bing_web_search, 
    extract_relevant_info, 
    fetch_page_content, 
    extract_snippet_with_context,
    google_serper_search,
    extract_relevant_info_serper
)
from evaluate.evaluate import (
    run_evaluation, 
    extract_answer_fn
)
from prompts.prompts import (
    get_gpqa_search_o1_instruction, 
    get_math_search_o1_instruction, 
    get_code_search_o1_instruction, 
    get_singleqa_search_o1_instruction, 
    get_multiqa_search_o1_instruction, 
    get_webpage_to_reasonchain_instruction,
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
)

# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

def parse_args():
    parser = argparse.ArgumentParser(description="Run Search-o1 for various datasets and models.")

    # Dataset and split configuration
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

    # Search and document retrieval configuration
    parser.add_argument(
        '--max_search_limit',
        type=int,
        default=10,
        help="Maximum number of searches per question."
    )

    parser.add_argument(
        '--max_turn',
        type=int,
        default=15,
        help="Maximum number of turns."
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help="Maximum number of search documents to return."
    )

    parser.add_argument(
        '--max_doc_len',
        type=int,
        default=3000,
        help="Maximum length of each searched document."
    )

    parser.add_argument(
        '--use_jina',
        type=bool,
        default=False,
        help="Whether to use Jina API for document fetching."
    )

    parser.add_argument(
        '--jina_api_key',
        type=str,
        default='None',
        help="Your Jina API Key to Fetch URL Content."
    )

    # Sampling parameters
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
        '--min_p',
        type=float,
        default=0.05,
        help="Minimum p sampling parameter."
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
        default=1.0,
        help="Repetition penalty. If not set, defaults based on the model."
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=32768,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    # Bing API Configuration
    parser.add_argument(
        '--bing_subscription_key',
        type=str,
        default=None,
        help="Bing Search API subscription key."
    )

    parser.add_argument(
        '--bing_endpoint',
        type=str,
        default="https://api.bing.microsoft.com/v7.0/search",
        help="Bing Search API endpoint."
    )

    # Serper API Configuration (New)
    parser.add_argument(
        '--serper_api_key',
        type=str,
        default=None,
        help="Google Serper API key."
    )

    # Search Engine Choice (New)
    parser.add_argument(
        '--search_engine',
        type=str,
        default="bing",
        choices=["bing", "serper"],
        help="Search engine to use (bing or serper). Default: bing"
    )

    # Add new eval and seed arguments
    parser.add_argument(
        '--eval',
        action='store_true',
        help="Whether to run evaluation after generation."
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help="Random seed for generation. If not set, will use current timestamp as seed."
    )
    
    # Add new arguments to parser
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
        '--concurrent_limit',
        type=int,
        default=200,
        help="Maximum number of concurrent API calls"
    )

    return parser.parse_args()

async def generate_response(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
    top_k: int,
    min_p: float,
    model_name: str,
    retry_limit: int = 3,
) -> str:
    """Generate a single response with retry logic"""
    for attempt in range(retry_limit):
        try:
            async with semaphore:
                messages = [{"role": "user", "content": prompt}]
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=min(max_tokens, 32768),  # Reserve 1000 tokens for prompt
                    stop=[END_SEARCH_QUERY],
                    extra_body={
                        'top_k': top_k,
                        'include_stop_str_in_output': True,
                        'repetition_penalty': repetition_penalty,
                        # 'min_p': min_p
                    },
                    timeout=1500,
                )
                # print('---\n', response.choices[0].message.content)
                return response.choices[0].message.content
        except Exception as e:
            print(f"Generate Response Error occurred: {e}, Starting retry attempt {attempt + 1}")
            if attempt == retry_limit - 1:
                print(f"Failed after {retry_limit} attempts: {e}")
                return ""
            await asyncio.sleep(1 * (attempt + 1))
    return ""

async def generate_webpage_to_reasonchain(
    client: AsyncOpenAI,
    original_question: str,
    prev_reasoning: str,
    search_query: str,
    document: str,
    dataset_name: str,
    batch_output_records: List[Dict],
    max_tokens: int = 32768,
    temperature: float = 0.7,
    top_p: float = 0.8,
    repetition_penalty: float = 1.05,
    top_k: int = 20,
    min_p: float = 0.05,
    model_name: str = "QwQ-32B",
    semaphore: asyncio.Semaphore = None,
) -> str:
    user_prompt = get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document)

    raw_output = await generate_response(
        client=client,
        prompt=user_prompt,
        semaphore=semaphore,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        min_p=min_p,
        model_name=model_name,
    )
    
    extracted_info = extract_answer_fn(raw_output, mode='infogen')

    batch_output_records.append({
        'prompt': user_prompt,
        'raw_output': raw_output,
        'extracted_info': extracted_info
    })

    return extracted_info


def extract_between(text, start_marker, end_marker):
    """
    Extracts text between two markers in a string.

    Parameters:
    - text (str): The source text to extract from
    - start_marker (str): The starting marker/tag
    - end_marker (str): The ending marker/tag

    Returns:
    - Optional[str]: The extracted text between markers, or None if not found
    """
    pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None

def replace_recent_steps(origin_str, replace_str):
    """
    Replaces specific steps in the original reasoning steps with new steps.
    If a replacement step contains "DELETE THIS STEP", that step is removed.

    Parameters:
    - origin_str (str): The original reasoning steps.
    - replace_str (str): The steps to replace or delete.

    Returns:
    - str: The updated reasoning steps after applying replacements.
    """

    def parse_steps(text):
        """
        Parses the reasoning steps from a given text.

        Parameters:
        - text (str): The text containing reasoning steps.

        Returns:
        - dict: A dictionary mapping step numbers to their content.
        """
        step_pattern = re.compile(r"Step\s+(\d+):\s*")
        steps = {}
        current_step_num = None
        current_content = []

        for line in text.splitlines():
            step_match = step_pattern.match(line)
            if step_match:
                # If there's an ongoing step, save its content
                if current_step_num is not None:
                    steps[current_step_num] = "\n".join(current_content).strip()
                current_step_num = int(step_match.group(1))
                content = line[step_match.end():].strip()
                current_content = [content] if content else []
            else:
                if current_step_num is not None:
                    current_content.append(line)
        
        # Save the last step if any
        if current_step_num is not None:
            steps[current_step_num] = "\n".join(current_content).strip()
        
        return steps

    # Parse the original and replacement steps
    origin_steps = parse_steps(origin_str)
    replace_steps = parse_steps(replace_str)

    # Apply replacements
    for step_num, content in replace_steps.items():
        if "DELETE THIS STEP" in content:
            # Remove the step if it exists
            if step_num in origin_steps:
                del origin_steps[step_num]
        else:
            # Replace or add the step
            origin_steps[step_num] = content

    # Sort the steps by step number
    sorted_steps = sorted(origin_steps.items())

    # Reconstruct the reasoning steps as a single string
    new_reasoning_steps = "\n\n".join([f"{content}" for num, content in sorted_steps])

    return new_reasoning_steps

async def process_single_sequence(
    seq: Dict,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    search_cache: Dict,
    url_cache: Dict,
    batch_output_records: List[Dict],
    turn: int = 0,
) -> Dict:
    """Process a single sequence through its entire reasoning chain"""
    
    while not seq['finished'] and turn < args.max_turn:
        # Generate next step in reasoning
        text = await generate_response(
            client=client,
            prompt=seq['prompt'],
            semaphore=semaphore,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k_sampling,
            min_p=args.min_p,
            model_name=args.model_name,
        )
        
        seq['history'].append(text)
        seq['prompt'] += text
        seq['output'] += text

        # Extract search query
        search_query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

        if search_query and seq['output'].rstrip().endswith(END_SEARCH_QUERY):
            # Remove the </think> tag from the prompt and output
            seq['prompt'] = seq['prompt'].replace('</think>\n','')
            seq['output'] = seq['output'].replace('</think>\n','')
            if seq['search_count'] < args.max_search_limit and search_query not in seq['executed_search_queries']:
                # Execute search
                results = {}
                if search_query in search_cache:
                    results = search_cache[search_query]
                else:
                    try:
                        if args.search_engine == "bing":
                            results = bing_web_search(search_query, args.bing_subscription_key, args.bing_endpoint)
                        elif args.search_engine == "serper":
                            results = google_serper_search(search_query, args.serper_api_key)
                        search_cache[search_query] = results
                    except Exception as e:
                        print(f"Error during search query '{search_query}' using {args.search_engine}: {e}")
                        search_cache[search_query] = {}
                        results = {}
                
                if args.search_engine == "bing":
                    relevant_info = extract_relevant_info(results)[:args.top_k]
                elif args.search_engine == "serper":
                    relevant_info = extract_relevant_info_serper(results)[:args.top_k]
                else: # Should not happen
                    relevant_info = []
                seq['relevant_info'] = relevant_info

                # Process documents
                formatted_documents = ""
                urls_to_fetch = []
                for doc_info in relevant_info:
                    url = doc_info['url']
                    if url not in url_cache:
                        urls_to_fetch.append(url)

                if urls_to_fetch:
                    try:
                        contents = fetch_page_content(urls_to_fetch, use_jina=args.use_jina, jina_api_key=args.jina_api_key)
                        for url, content in contents.items():
                            url_cache[url] = content
                    except Exception as e:
                        print(f"Error fetching URLs: {e}")
                        for url in urls_to_fetch:
                            url_cache[url] = ""

                for i, doc_info in enumerate(relevant_info):
                    url = doc_info['url']
                    raw_context = url_cache[url]
                    doc_info['snippet'] = doc_info['snippet'].replace('<b>','').replace('</b>','')
                    success, filtered_context = extract_snippet_with_context(raw_context, doc_info['snippet'], context_chars=args.max_doc_len)
                    context = filtered_context if success else raw_context[:args.max_doc_len*2]
                    
                    doc_info['context'] = context
                    formatted_documents += f"**Web Page {i + 1}:**\n"
                    formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"

                # Process reasoning steps
                all_reasoning_steps = seq['output'].replace('\n\n', '\n').split("\n")
                truncated_prev_reasoning = ""
                for i, step in enumerate(all_reasoning_steps):
                    truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"

                prev_steps = truncated_prev_reasoning.split('\n\n')
                if len(prev_steps) > 5:
                    truncated_prev_reasoning = ''
                    for i, step in enumerate(prev_steps):
                        if i == 0 or i >= len(prev_steps) - 4 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                            truncated_prev_reasoning += step + '\n\n'
                        else:
                            if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                                truncated_prev_reasoning += '...\n\n'
                truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')

                # Generate webpage analysis
                analysis = await generate_webpage_to_reasonchain(
                    client=client,
                    original_question=seq['item']['Question'],
                    prev_reasoning=truncated_prev_reasoning,
                    search_query=search_query,
                    document=formatted_documents,
                    dataset_name=args.dataset_name,
                    batch_output_records=batch_output_records,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    top_k=args.top_k_sampling,
                    min_p=args.min_p,
                    model_name=args.model_name,
                    semaphore=semaphore,
                )

                # Update sequence with analysis
                append_text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"
                
                seq['prompt'] += append_text
                seq['output'] += append_text
                seq['history'].append(append_text)
                
                seq['search_count'] += 1
                seq['executed_search_queries'].add(search_query)

            elif seq['search_count'] >= args.max_search_limit:
                limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                seq['prompt'] += limit_message
                seq['output'] += limit_message
                seq['history'].append(limit_message)

            elif search_query in seq['executed_search_queries']:
                limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                seq['prompt'] += limit_message
                seq['output'] += limit_message
                seq['history'].append(limit_message)

        else:
            seq['finished'] = True

        turn += 1

    return seq

async def main_async():
    args = parse_args()

    # Validate API keys based on selected search engine
    if args.search_engine == "bing" and not args.bing_subscription_key:
        print("Error: Bing search engine is selected, but --bing_subscription_key is not provided.")
        return
    elif args.search_engine == "serper" and not args.serper_api_key:
        print("Error: Serper search engine is selected, but --serper_api_key is not provided.")
        return
    elif args.search_engine not in ["bing", "serper"]: # Should be caught by choices, but good to have
        print(f"Error: Invalid search engine '{args.search_engine}'. Choose 'bing' or 'serper'.")
        return

    # Set random seed
    if args.seed is None:
        args.seed = int(time.time())
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.jina_api_key == 'None':
        jina_api_key = None

    # Data paths based on dataset
    if args.dataset_name == 'livecode':
        data_path = f'./data/LiveCodeBench/{args.split}.json'
    elif args.dataset_name == 'webwalker':
        data_path = f'./data/WebWalkerQA/{args.split}.json'
    elif args.dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'gaia', 'hle']:
        data_path = f'./data/{args.dataset_name.upper()}/{args.split}.json'
    else:
        data_path = f'./data/QA_Datasets/{args.dataset_name}.json'

    print('-----------------------')
    print(f'Using {args.dataset_name} {args.split} set.')
    print('-----------------------')

    # ---------------------- Caching Mechanism ----------------------
    cache_dir = './cache'
    search_cache_path = os.path.join(cache_dir, f'{args.search_engine}_search_cache.json')
    url_cache_path = os.path.join(cache_dir, 'url_cache.json')

    os.makedirs(cache_dir, exist_ok=True)

    # Load existing caches
    search_cache = json.load(open(search_cache_path)) if os.path.exists(search_cache_path) else {}
    url_cache = json.load(open(url_cache_path)) if os.path.exists(url_cache_path) else {}

    def save_caches():
        with open(search_cache_path, 'w', encoding='utf-8') as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)
        with open(url_cache_path, 'w', encoding='utf-8') as f:
            json.dump(url_cache, f, ensure_ascii=False, indent=2)

    # Define output directory
    if 'qwq' in args.model_name.lower():
        model_short_name = 'qwq'
    elif 'deepseek' in args.model_name.lower():
        if 'llama-8b' in args.model_name.lower():
            model_short_name = 'dpsk-llama-8b'
        elif 'llama-70b' in args.model_name.lower():
            model_short_name = 'dpsk-llama-70b'
        elif 'qwen-1.5b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-1.5b'
        elif 'qwen-7b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-7b'
        elif 'qwen-32b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-32b'
    elif 'sky-t1' in args.model_name.lower():
        model_short_name = 'sky-t1'
    else:
        model_short_name = args.model_name.split('/')[-1].lower().replace('-instruct', '')

    if model_short_name in ['qwq', 'dpsk-llama-8b', 'dpsk-llama-70b', 'dpsk-qwen-1.5b', 'dpsk-qwen-7b', 'dpsk-qwen-32b', 'sky-t1']:
        if args.dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode']:
            output_dir = f'./outputs/{args.dataset_name}.{model_short_name}.search_o1'
            if args.dataset_name == 'gpqa' and (args.max_search_limit != 5 or args.top_k != 10):
                output_dir = f'./outputs/runs.analysis/{args.dataset_name}.{model_short_name}.search_o1.{args.max_search_limit}.{args.top_k}'
        else:
            output_dir = f'./outputs/runs.qa/{args.dataset_name}.{model_short_name}.search_o1'
    else:
        output_dir = f'./outputs/runs.baselines/{args.dataset_name}.{model_short_name}.search_o1'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the OpenAI client
    client = AsyncOpenAI(
        api_key="empty",
        base_url=args.api_base_url,
    )

    # Load and prepare data
    with open(data_path, 'r', encoding='utf-8') as json_file:
        filtered_data = json.load(json_file)

    if args.subset_num != -1:
        indices = list(range(len(filtered_data)))
        selected_indices = random.sample(indices, min(args.subset_num, len(indices)))
        filtered_data = [filtered_data[i] for i in selected_indices]

    # Prepare sequences
    active_sequences = []
    for item in filtered_data:
        question = item['Question']
        
        # Get appropriate instruction and user prompt based on dataset
        if args.dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'gaia', 'hle', 'webwalker']:
            if args.dataset_name in ['nq', 'triviaqa']:
                instruction = get_singleqa_search_o1_instruction(args.max_search_limit)
            else:
                instruction = get_multiqa_search_o1_instruction(args.max_search_limit)
            
            if 'qwq' in args.model_name.lower() or 'sky-t1' in args.model_name.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')
            elif 'deepseek' in args.model_name.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='dpsk')
            else:
                user_prompt = get_task_instruction_openqa(question)

        elif args.dataset_name in ['math500', 'aime', 'amc']:
            instruction = get_math_search_o1_instruction(args.max_search_limit)
            if 'qwq' in args.model_name.lower() or 'sky-t1' in args.model_name.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')
            elif 'deepseek' in args.model_name.lower():
                user_prompt = get_task_instruction_math(question, model_name='dpsk')
            else:
                user_prompt = get_task_instruction_math(question)

        elif args.dataset_name in ['gpqa']:
            instruction = get_gpqa_search_o1_instruction(args.max_search_limit)
            if 'qwq' in args.model_name.lower() or 'sky-t1' in args.model_name.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
            elif 'deepseek' in args.model_name.lower():
                instruction += gpqa_search_o1_examples_dpsk
                user_prompt = get_task_instruction_multi_choice(question, model_name='dpsk')
            elif 'llama' in args.model_name.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(question)

        elif args.dataset_name == 'livecode':
            instruction = get_code_search_o1_instruction(args.max_search_limit)
            question_title = item.get('question_title', '')
            if 'qwq' in args.model_name.lower() or 'deepseek' in args.model_name.lower() or 'sky-t1' in args.model_name.lower():
                user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
            else:
                user_prompt = get_task_instruction_code(question)
        else:
            instruction = get_multiqa_search_o1_instruction(args.max_search_limit)
            user_prompt = get_task_instruction_openqa(question)

        prompt = instruction + user_prompt
        active_sequences.append({
            'item': item,
            'prompt': prompt,
            'output': '',
            'finished': False,
            'history': [],
            'search_count': 0,
            'executed_search_queries': set(),
        })

    # Initialize batch output records
    batch_output_records = []
    start_time = time.time()

    # Create semaphore for concurrent API calls
    semaphore = asyncio.Semaphore(args.concurrent_limit)

    # Process all sequences concurrently
    tasks = [
        process_single_sequence(
            seq=seq,
            client=client,
            semaphore=semaphore,
            args=args,
            search_cache=search_cache,
            url_cache=url_cache,
            batch_output_records=batch_output_records
        )
        for seq in active_sequences
    ]

    # Run all sequences concurrently with progress bar
    with tqdm(total=len(tasks)) as pbar:
        async def track_progress(task):
            result = await task
            pbar.update(1)
            return result
            
        tracked_tasks = [track_progress(task) for task in tasks]
        completed_sequences = await asyncio.gather(*tracked_tasks)

    total_time = time.time() - start_time

    # Save batch output records
    t = time.localtime()
    batch_output_file = os.path.join(output_dir, f'{args.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.info_extract.json')
    with open(batch_output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_output_records, f, ensure_ascii=False, indent=2)

    # Prepare output list and save results
    output_list = [seq['output'] for seq in completed_sequences]
    
    if args.eval:
        run_evaluation(filtered_data, [seq['prompt'] for seq in completed_sequences], output_list, args.dataset_name, output_dir, total_time, args.split)
    else:
        t = time.localtime()
        result_json_name = f'{args.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
        
        for item, seq in zip(filtered_data, completed_sequences):
            item['Output'] = seq['output']
            
        with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
            json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

    # Save caches
    save_caches()
    print("Process completed.")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
