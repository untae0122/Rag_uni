# run_naive_rag.py
import os
import json
import time
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import argparse
import csv
import random
import asyncio
import numpy as np

from search.bing_search import (
    bing_web_search,
    extract_relevant_info,
    fetch_page_content,
    extract_snippet_with_context,
    google_serper_search,
    extract_relevant_info_serper,
)
from evaluate.evaluate import run_evaluation, extract_answer_fn
from openai import AsyncOpenAI

import re
import string
from nltk.tokenize import sent_tokenize
import torch
from prompts.prompts import (
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
    get_naive_rag_instruction, 
    get_query_plan_instruction,
)
import aiohttp

def parse_args():
    parser = argparse.ArgumentParser(description="Run Naive RAG for various datasets and models.")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset to use.")
    parser.add_argument('--split', type=str, required=True, help="Dataset split to use.")
    parser.add_argument('--subset_num', type=int, default=-1, help="Number of examples to process. Defaults to all if not specified.")
    parser.add_argument('--top_k', type=int, default=10, help="Number of top search results to retrieve.")
    parser.add_argument('--max_doc_len', type=int, default=3000, help="Maximum length of each searched document.")
    parser.add_argument('--model_name', type=str, default="QwQ-32B", help="Name of the model to use")
    parser.add_argument('--api_base_url', type=str, required=True, help="Base URL for the API endpoint")
    parser.add_argument('--aux_model_name', type=str, default="Qwen2.5-72B-Instruct", help="Name of the model to use")
    parser.add_argument('--aux_api_base_url', type=str, required=True, help="Base URL for the API endpoint")
    parser.add_argument('--use_jina', action='store_true', help="Whether to use Jina API for document fetching.")
    parser.add_argument('--jina_api_key', type=str, default='None', help="Your Jina API Key to Fetch URL Content.")
    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument('--top_p', type=float, default=0.8, help="Top-p sampling parameter.")
    parser.add_argument('--top_k_sampling', type=int, default=20, help="Top-k sampling parameter.")
    parser.add_argument('--repetition_penalty', type=float, default=None, help="Repetition penalty. If not set, defaults based on the model.")
    parser.add_argument('--max_tokens', type=int, default=32768, help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset.")
    parser.add_argument('--bing_subscription_key', type=str, default=None, help="Bing Search API subscription key.")
    parser.add_argument('--bing_endpoint', type=str, default="https://api.bing.microsoft.com/v7.0/search", help="Bing Search API endpoint.")
    parser.add_argument('--serper_api_key', type=str, default=None, help="Google Serper API key.")
    parser.add_argument('--search_engine', type=str, default="bing", choices=["bing", "serper"], help="Search engine to use (bing or serper). Default: bing")
    parser.add_argument('--concurrent_limit', type=int, default=50, help="Maximum number of concurrent API calls")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--eval', action='store_true', help="Whether to run evaluation")
    parser.add_argument('--apply_query_planning', action='store_true', help="Whether to apply query planning for search")
    return parser.parse_args()

async def generate_response(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model_name: str,
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
                    max_tokens=min(max_tokens, 32768 - 1000),  # Reserve 1000 tokens for prompt
                    timeout=600,
                )
                return response.choices[0].message.content
        except Exception as e:
            if attempt == retry_limit - 1:
                print(f"Failed after {retry_limit} attempts: {e}")
                return ""
            if "maximum context length" in str(e):
                max_tokens = max_tokens // 2
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
) -> List[str]:
    """Generate all responses concurrently with a limit"""
    semaphore = asyncio.Semaphore(concurrent_limit)
    
    tasks = [
        generate_response(
            client, prompt, semaphore, temperature, top_p, max_tokens, model_name
        )
        for prompt in prompts
    ]
    
    with tqdm(total=len(tasks)) as pbar:
        async def track_progress(task):
            result = await task
            pbar.update(1)
            return result
            
        tracked_tasks = [track_progress(task) for task in tasks]
        responses = await asyncio.gather(*tracked_tasks)
    
    return responses

async def parse_query_plan(response: str) -> List[str]:
    """Parse the query plan response to extract sub-queries"""
    try:
        # Try to find and parse JSON content
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            json_content = json.loads(match.group())
            if 'query_plan' in json_content:
                query_plan = json_content['query_plan'][:3]  # Take first 3 queries
                # print('query_plan', query_plan)
                return query_plan
    except:
        pass
    # Fallback: return empty list if parsing fails
    return []

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
    
    client = AsyncOpenAI(
        api_key="empty",
        base_url=args.api_base_url,
    )
    
    # Add aux_client initialization
    aux_client = AsyncOpenAI(
        api_key="empty",
        base_url=args.aux_api_base_url,
    )
    
    # Paths to datasets
    if args.dataset_name == 'math500':
        data_path = f'./data/MATH500/{args.split}.json'
    elif args.dataset_name == 'gpqa':
        data_path = f'./data/GPQA/{args.split}.json'
    elif args.dataset_name == 'supergpqa':
        data_path = f'./data/SuperGPQA/{args.split}.json'
    elif args.dataset_name == 'aime':
        data_path = f'./data/AIME/{args.split}.json'
    elif args.dataset_name == 'amc':
        data_path = f'./data/AMC/{args.split}.json'
    elif args.dataset_name == 'livecode':
        data_path = f'./data/LiveCodeBench/{args.split}.json'
    elif args.dataset_name == 'openthoughts':
        data_path = f'./data/OpenThoughts/{args.split}.json'
    elif args.dataset_name == 'gaia':
        data_path = f'./data/GAIA/{args.split}.json'
    elif args.dataset_name == 'hle':
        data_path = f'./data/HLE/{args.split}.json'
    elif args.dataset_name == 'webwalker':
        data_path = f'./data/WebWalkerQA/{args.split}.json'
    elif args.dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'medmcqa', 'pubhealth']:
        data_path = f'./data/QA_Datasets/{args.dataset_name}.json'
    else:
        raise ValueError(f"Unsupported dataset_name: {args.dataset_name}")

    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        if args.subset_num != -1:
            data = data[:args.subset_num]

    # ---------------------- Caching Mechanism ----------------------
    # Define cache directories and file paths
    cache_dir = './cache'
    search_cache_path = os.path.join(cache_dir, f'{args.search_engine}_search_cache.json')
    url_cache_path = os.path.join(cache_dir, 'url_cache.json')

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Load existing caches or initialize empty dictionaries
    if os.path.exists(search_cache_path):
        with open(search_cache_path, 'r', encoding='utf-8') as f:
            search_cache = json.load(f)
    else:
        search_cache = {}

    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache = json.load(f)
    else:
        url_cache = {}

    # Function to save caches
    def save_caches():
        with open(search_cache_path, 'w', encoding='utf-8') as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)
        with open(url_cache_path, 'w', encoding='utf-8') as f:
            json.dump(url_cache, f, ensure_ascii=False, indent=2)

    # ---------------------- Model Loading ----------------------
    # Set model short name
    if 'qwq' in args.model_name.lower():
        model_short_name = 'qwq'
    elif 'deepseek' in args.model_name.lower():
        if 'llama-8b' in args.model_name.lower():
            model_short_name = 'dpsk-llama-8b'
        elif 'qwen-1.5b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-1.5b'
        elif 'qwen-7b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-7b'
        elif 'qwen-32b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-32b'
        elif 'qwen-14b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-14b'
        elif 'llama-70b' in args.model_name.lower():
            model_short_name = 'dpsk-llama-70b'
    elif 'sky-t1' in args.model_name.lower():
        model_short_name = 'sky-t1'
    else:
        model_short_name = args.model_name.split('/')[-1].lower().replace('-instruct', '')

    if args.apply_query_planning:
        method = 'plan_rag'
    else:
        method = 'naive_rag'
    
    # Set output directory
    if model_short_name in ['qwq', 'dpsk-llama-8b', 'dpsk-qwen-1.5b', 'dpsk-qwen-7b', 'dpsk-qwen-32b', 'sky-t1', 'dpsk-qwen-14b']:
        if args.dataset_name in ['math500', 'gpqa', 'supergpqa', 'aime', 'amc', 'livecode', 'openthoughts']:
            output_dir = f'./outputs/{args.dataset_name}.{model_short_name}.{method}'
        else:
            output_dir = f'./outputs/runs.qa/{args.dataset_name}.{model_short_name}.{method}'
    else:
        output_dir = f'./outputs/runs.baselines/{args.dataset_name}.{model_short_name}.{method}'
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------- Search and Document Retrieval ----------------------
    print("Performing Web Searches for all questions...")

    # Initialize a list to hold relevant information for each question
    all_relevant_info = []

    for item in tqdm(data, desc=f"Searching with {args.search_engine}"):
        question = item['Question']
        results = {} # Initialize results
        
        if args.apply_query_planning:
            # Generate query plan using aux model
            plan_prompt = get_query_plan_instruction(question)
            plan_response = await generate_response(
                aux_client,  # Use aux_client instead of client
                plan_prompt, 
                asyncio.Semaphore(1),
                args.temperature,
                args.top_p,
                args.max_tokens,
                args.aux_model_name,  # Use aux_model_name instead of model_name
            )
            
            sub_queries = await parse_query_plan(plan_response)
            if not sub_queries:  # Fallback to original question if parsing fails
                sub_queries = [question]
            
            # Collect results from all sub-queries
            all_results_data = [] # Renamed to avoid conflict with 'results' variable for single search
            for sub_query in sub_queries:
                sub_query = str(sub_query)
                current_search_results = {}
                if sub_query in search_cache:
                    current_search_results = search_cache[sub_query]
                else:
                    if args.search_engine == "bing":
                        current_search_results = bing_web_search(sub_query[:500], args.bing_subscription_key, args.bing_endpoint, market='en-US', language='en')
                    elif args.search_engine == "serper":
                        current_search_results = google_serper_search(sub_query[:500], args.serper_api_key)
                    search_cache[sub_query] = current_search_results
                
                if args.search_engine == "bing":
                    relevant_sub_info = extract_relevant_info(current_search_results)[:5]  # top-5 for each sub-query
                elif args.search_engine == "serper":
                    relevant_sub_info = extract_relevant_info_serper(current_search_results)[:5]
                else: # Should not happen
                    relevant_sub_info = []
                all_results_data.extend(relevant_sub_info)
            
            all_relevant_info.append(all_results_data)
        else:
            # Original search logic
            if question in search_cache:
                results = search_cache[question]
            else:
                search_question = question[:500] if args.dataset_name == 'livecode' else question
                if args.search_engine == "bing":
                    results = bing_web_search(search_question, args.bing_subscription_key, args.bing_endpoint, market='en-US', language='en')
                elif args.search_engine == "serper":
                    results = google_serper_search(search_question, args.serper_api_key)
                search_cache[question] = results

            if args.search_engine == "bing":
                relevant_info_for_item = extract_relevant_info(results)[:args.top_k]
            elif args.search_engine == "serper":
                relevant_info_for_item = extract_relevant_info_serper(results)[:args.top_k]
            else: # Should not happen
                relevant_info_for_item = []
            all_relevant_info.append(relevant_info_for_item)

    # Save search cache after retrieval
    save_caches()
    print("Search cache saved.")

    # Collect all unique URLs to fetch
    unique_urls = set()
    url_snippets_map = {}

    for relevant_info in all_relevant_info:
        for info in relevant_info:
            url = info['url']
            snippet = info.get('snippet', "")
            unique_urls.add(url)
            url_snippets_map[url] = snippet

    # Determine which URLs need to be fetched
    urls_to_fetch = [url for url in unique_urls if url not in url_cache]

    print(f"Fetching {len(urls_to_fetch)} unique URLs...")
    fetched_contents = fetch_page_content(
        urls_to_fetch,
        use_jina=args.use_jina,
        jina_api_key=args.jina_api_key,
        show_progress=True,
        # snippets=url_snippets_map
    )

    # Update URL cache with fetched contents
    for url, content in fetched_contents.items():
        url_cache[url] = content

    # Save URL cache after fetching
    save_caches()
    print("URL cache saved.")

    # ---------------------- Prompt Construction ----------------------
    print("Constructing prompts for generation...")
    input_prompts = []

    for idx, item in enumerate(tqdm(data, desc="Constructing Prompts")):
        question = item['Question']

        formatted_documents = ""
        relevant_info = all_relevant_info[idx]
        for i, doc_info in enumerate(relevant_info):
            url = doc_info['url']
            snippet = doc_info.get('snippet', "")
            raw_context = url_cache.get(url, "")
            success, context = extract_snippet_with_context(raw_context, snippet, context_chars=args.max_doc_len)
            if success:
                context = context
            else:
                context = raw_context[:2 * args.max_doc_len]

            # Clean snippet from HTML tags if any
            clean_snippet = re.sub('<[^<]+?>', '', snippet)  # Removes HTML tags

            formatted_documents += f"**Document {i + 1}:**\n"
            formatted_documents += f"**Title:** {doc_info.get('title', '')}\n"
            formatted_documents += f"**URL:** {url}\n"
            formatted_documents += f"**Snippet:** {clean_snippet}\n"
            formatted_documents += f"**Content:** {context}\n\n"

        # Construct the instruction with documents and question
        instruction = get_naive_rag_instruction(question, formatted_documents)
        # print(instruction)

        # Get task-specific prompt
        if args.dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'webwalker', 'gaia', 'hle']:
            if 'qwq' in args.model_name.lower() or 'sky-t1' in args.model_name.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')
            elif 'deepseek' in args.model_name.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='dpsk')
            else:
                user_prompt = get_task_instruction_openqa(question)
        elif args.dataset_name in ['math500', 'aime', 'amc']:
            if 'qwq' in args.model_name.lower() or 'sky-t1' in args.model_name.lower() or 'deepseek' in args.model_name.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')
            else:
                user_prompt = get_task_instruction_math(question)
        elif args.dataset_name in ['gpqa']:
            if 'qwq' in args.model_name.lower() or 'sky-t1' in args.model_name.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
            elif 'deepseek' in args.model_name.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='dpsk')
            elif 'llama' in args.model_name.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(question)
        elif args.dataset_name == 'livecode':
            question_title = item.get('question_title', '')
            if 'qwq' in args.model_name.lower() or 'deepseek' in args.model_name.lower() or 'sky-t1' in args.model_name.lower():
                user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
            else:
                user_prompt = get_task_instruction_code(question)
        elif args.dataset_name == 'openthoughts':
            domain = item['domain']
            if domain == 'math':
                if 'qwq' in args.model_name.lower() or 'sky-t1' in args.model_name.lower() or 'deepseek' in args.model_name.lower():
                    user_prompt = get_task_instruction_math(question, model_name='qwq')
                else:
                    user_prompt = get_task_instruction_math(question)
            elif domain == 'code':
                question_title = item.get('question_title', '')
                if 'qwq' in args.model_name.lower() or 'sky-t1' in args.model_name.lower() or 'deepseek' in args.model_name.lower():
                    user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
                else:
                    user_prompt = get_task_instruction_code(question)
            elif domain == 'puzzle':
                if 'qwq' in args.model_name.lower() or 'sky-t1' in args.model_name.lower():
                    user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
                elif 'deepseek' in args.model_name.lower():
                    user_prompt = get_task_instruction_multi_choice(question, model_name='dpsk')
                elif 'llama' in args.model_name.lower():
                    user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
                else:
                    user_prompt = get_task_instruction_multi_choice(question)
        elif args.dataset_name == 'supergpqa':
            question_type = item['question_type']
            if question_type == 'generation':
                if 'qwq' in args.model_name.lower() or 'sky-t1' in args.model_name.lower():
                    user_prompt = get_task_instruction_openqa(question, model_name='qwq')
                elif 'deepseek' in args.model_name.lower():
                    user_prompt = get_task_instruction_openqa(question, model_name='dpsk')
                elif 'llama' in args.model_name.lower():
                    user_prompt = get_task_instruction_openqa(question, model_name='llama')
                else:
                    user_prompt = get_task_instruction_openqa(question)
            elif question_type == 'multi-choice':
                if 'qwq' in args.model_name.lower() or 'sky-t1' in args.model_name.lower():
                    user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
                elif 'deepseek' in args.model_name.lower():
                    user_prompt = get_task_instruction_multi_choice(question, model_name='dpsk')
                else:
                    user_prompt = get_task_instruction_multi_choice(question)
        else:
            user_prompt = ""  # Default to empty if dataset not matched

        # Combine instruction and user prompt
        full_prompt = instruction + "\n\n" + user_prompt

        # Just append the full prompt directly
        input_prompts.append(full_prompt)

    # ---------------------- Generation ----------------------
    print("Generating answers...")
    
    start_time = time.time()
    output_list = await generate_all_responses(
        client,
        input_prompts,
        args.concurrent_limit,
        args.temperature,
        args.top_p,
        args.max_tokens,
        args.model_name,
    )
    total_time = time.time() - start_time

    # ---------------------- Evaluation ----------------------
    if args.eval:
        print("Evaluating generated answers...")
        run_evaluation(
            filtered_data=data,
            input_list=input_prompts,
            output_list=output_list,
            dataset_name=args.dataset_name,
            output_dir=output_dir,
            total_time=total_time,
            split=args.split,
        )
    else:
        # Save raw outputs and prompts without evaluation
        for item, prompt, result in zip(data, input_prompts, output_list):
            item['prompt'] = prompt
            if isinstance(result, str):
                item['Output'] = result
            else:
                item['Output'] = result.outputs[0].text
        
        t = time.localtime()
        result_json_name = f'{args.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
        # Save prediction results
        with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)

    # ---------------------- Update Search and URL Cache ----------------------
    print('Updating Search and URL Cache...')
    # Load existing caches or initialize empty dictionaries
    if os.path.exists(search_cache_path):
        with open(search_cache_path, 'r', encoding='utf-8') as f:
            search_cache_new = json.load(f)
    else:
        search_cache_new = {}

    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache_new = json.load(f)
    else:
        url_cache_new = {}

    search_cache.update(search_cache_new)
    url_cache.update(url_cache_new)

    save_caches()

    print("Process completed.")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
