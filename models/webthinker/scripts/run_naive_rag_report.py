# run_web_thinker.py
import os
import json
import time
import re
from tqdm import tqdm
import numpy as np
import torch
import string
from typing import Optional, Tuple, List, Dict, Set
import argparse
import random
import asyncio
import aiohttp

from openai import AsyncOpenAI

from search.bing_search import (
    bing_web_search, 
    extract_relevant_info, 
    fetch_page_content, 
    fetch_page_content_async,
    extract_snippet_with_context,
    bing_web_search_async,
    google_serper_search_async,
    extract_relevant_info_serper
)
from prompts.prompts_report import (
    get_standard_rag_report_instruction,
)

from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
import langid
import signal


error_indicators = [
    'limit exceeded',
    'Error fetching',
    'Account balance not enough',
    'Invalid bearer token',
    'HTTP error occurred',
    'Error: Connection error occurred',
    'Error: Request timed out',
    'Unexpected error',
    'Please turn on Javascript',
    'Enable JavaScript',
    'port=443',
    'Please enable cookies',
]

def parse_args():
    parser = argparse.ArgumentParser(description="Run naive RAG for various datasets and models.")
    parser.add_argument('--single_question', type=str, default=None, help="Single question to process instead of dataset")
    parser.add_argument('--dataset_name', type=str, required=False, default='custom', help="Name of the dataset to use.")
    parser.add_argument('--split', type=str, required=False, default='test', help="Dataset split to use.")
    parser.add_argument('--subset_num', type=int, default=-1, help="Number of examples to process. Defaults to all if not specified.")

    parser.add_argument('--temperature', type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument('--top_p', type=float, default=0.8, help="Top-p sampling parameter.")
    parser.add_argument('--top_k', type=int, default=10, help="Maximum number of search documents to return.")
    parser.add_argument('--keep_links', action='store_true', default=False, help="Whether to keep links in fetched web content")
    parser.add_argument('--use_jina', action='store_true', help="Whether to use Jina API for document fetching.")
    parser.add_argument('--jina_api_key', type=str, default='None', help="Your Jina API Key to Fetch URL Content.")
    parser.add_argument('--bing_subscription_key', type=str, default=None, help="Bing Search API subscription key.")
    parser.add_argument('--bing_endpoint', type=str, default="https://api.bing.microsoft.com/v7.0/search", help="Bing Search API endpoint.")
    parser.add_argument('--serper_api_key', type=str, default=None, help="Google Serper API key.")
    parser.add_argument('--search_engine', type=str, default="bing", choices=["bing", "serper"], help="Search engine to use (bing or serper).")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for generation.")
    parser.add_argument('--api_base_url', type=str, required=True, help="Base URL for the API endpoint")
    parser.add_argument('--api_key', type=str, default="empty", help="API key for the model service")
    parser.add_argument('--model_name', type=str, default="QwQ-32B", help="Name of the model to use")
    parser.add_argument('--max_tokens', type=int, default=None, help="Maximum number of tokens to generate")
    parser.add_argument('--concurrent_limit', type=int, default=32, help="Maximum number of concurrent API calls")
    return parser.parse_args()


async def extract_between(text, start_marker, end_marker):
    """Extracts text between two markers in a string."""
    pattern = re.escape(end_marker[::-1]) + r"(.*?)" + re.escape(start_marker[::-1])
    
    try:
        # Run pattern matching with timeout
        matches = re.findall(pattern, text[::-1], flags=re.DOTALL)
        if matches:
            return matches[0][::-1].strip()
        return None
    except Exception as e:
        print(f"---Error:---\n{str(e)}")
        print(f"-------------------")
        return None

def format_search_results(relevant_info: List[Dict]) -> str:
    """Format search results into a readable string"""
    formatted_documents = ""
    for i, doc_info in enumerate(relevant_info):
        doc_info['title'] = doc_info['title'].replace('<b>','').replace('</b>','')
        doc_info['snippet'] = doc_info['snippet'].replace('<b>','').replace('</b>','')
        formatted_documents += f"***Web Page {i + 1}:***\n"
        formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
        # formatted_documents += f"Title: {doc_info['title']}\n"
        # formatted_documents += f"URL: {doc_info['url']}\n"
        # formatted_documents += f"Snippet: {doc_info['snippet']}\n\n"
        # if 'page_info' in doc_info:
        #     formatted_documents += f"Web Page Information: {doc_info['page_info']}\n\n\n\n"
    return formatted_documents

def extract_markdown_content(text):
    """Extract content between ```markdown and ``` tags."""
    pattern = r"```markdown\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return text

def judge_zh(input_str: str):
    assert isinstance(input_str, str), input_str
    if len(input_str) == 0:
        return False
    detect_result = langid.classify(input_str)
    if detect_result[0] == 'zh':
        return True
    else:
        return False



async def generate_response(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    temperature: float = 0.7,
    top_p: float = 0.8,
    retry_limit: int = 3,
    model_name: str = "gpt-3.5-turbo",
    max_tokens: Optional[int] = None,
) -> str:
    """Generate a response using the chat API"""
    for attempt in range(retry_limit):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout=600,
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"Generate Response Error occurred: {e}, Starting retry attempt {attempt + 1}")
            if attempt == retry_limit - 1:
                print(f"Failed after {retry_limit} attempts: {e}")
                return ""
            await asyncio.sleep(1 * (attempt + 1))
    return ""


async def process_single_sequence(
    question: str,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    search_cache: Dict,
    url_cache: Dict,
) -> Dict:
    """Process a single question through RAG pipeline"""
    
    results = {}
    # Search for relevant documents
    try:
        if question in search_cache:
            results = search_cache[question]
        else:
            if args.search_engine == "bing":
                if not args.bing_subscription_key:
                    print(f"Error: Bing search engine is selected, but BING_SUBSCRIPTION_KEY is not provided for question: {question}")
                    results = {}
                else:
                    results = await bing_web_search_async(question, args.bing_subscription_key, args.bing_endpoint)
            elif args.search_engine == "serper":
                if not args.serper_api_key:
                    print(f"Error: Serper search engine is selected, but SERPER_API_KEY is not provided for question: {question}")
                    results = {}
                else:
                    results = await google_serper_search_async(question, args.serper_api_key)
            else:
                print(f"Error: Unknown search engine: {args.search_engine}")
                results = {}
            
            if results: # Only cache if results are not empty (i.e., search was successful)
                search_cache[question] = results

    except Exception as e:
        print(f"Error during search for '{question}' using {args.search_engine}: {e}")
        results = {}

    # Extract and process relevant documents
    relevant_info = []
    if results:
        if args.search_engine == "bing":
            relevant_info = extract_relevant_info(results)[:args.top_k]
        elif args.search_engine == "serper":
            relevant_info = extract_relevant_info_serper(results)[:args.top_k]
    
    # Fetch page content for each result
    documents = []
    for idx, doc_info in enumerate(relevant_info):
        url = doc_info['url']
        if url not in url_cache:
            try:
                contents = await fetch_page_content_async(
                    [url], 
                    use_jina=args.use_jina, 
                    jina_api_key=args.jina_api_key, 
                    keep_links=args.keep_links
                )
                raw_content = contents[url]
                if not any(indicator.lower() in raw_content.lower() for indicator in error_indicators):
                    url_cache[url] = raw_content
                    # Set context size based on document index
                    context_chars = 8000 if idx < 5 else 4000
                    # Extract snippet with context
                    is_success, content = extract_snippet_with_context(raw_content, doc_info['snippet'], context_chars=context_chars)
                    documents.append({
                        'title': doc_info['title'],
                        'url': url,
                        'content': content
                    })
            except Exception as e:
                print(f"Error fetching URL {url}: {e}")
        else:
            raw_content = url_cache[url]
            # Set context size based on document index
            context_chars = 8000 if idx < 5 else 4000
            # Extract snippet with context
            is_success, content = extract_snippet_with_context(raw_content, doc_info['snippet'], context_chars=context_chars)
            documents.append({
                'title': doc_info['title'],
                'url': url,
                'content': content
            })

    # Generate response using RAG
    prompt = get_standard_rag_report_instruction(question, documents)
    response = await generate_response(
        client=client,
        prompt=prompt,
        semaphore=semaphore,
        temperature=args.temperature,
        top_p=args.top_p,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
    )

    article = extract_markdown_content(response)

    return {
        'question': question,
        'prompt': prompt,
        'response': response,
        'article': article,
        'documents': documents
    }


async def main_async():
    args = parse_args()

    # Validate API keys based on selected search engine
    if args.search_engine == "bing" and not args.bing_subscription_key:
        print("Error: Bing search engine is selected, but --bing_subscription_key is not provided.")
        return
    elif args.search_engine == "serper" and not args.serper_api_key:
        print("Error: Serper search engine is selected, but --serper_api_key is not provided.")
        return
    elif args.search_engine not in ["bing", "serper"]:
        print(f"Error: Invalid search engine '{args.search_engine}'. Choose 'bing' or 'serper'.")
        return

    # Set random seed
    if args.seed is None:
        args.seed = int(time.time())
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load or prepare data
    if args.single_question:
        filtered_data = [{'Question': args.single_question}]
    else:
        data_path = f'./data/{args.dataset_name}/{args.split}.json'
        with open(data_path, 'r', encoding='utf-8') as f:
            filtered_data = json.load(f)
        
        if args.subset_num != -1:
            filtered_data = random.sample(filtered_data, min(args.subset_num, len(filtered_data)))

    # Setup caching
    os.makedirs('./cache', exist_ok=True)
    search_cache_path = f'./cache/{args.search_engine}_search_cache.json'
    url_cache_path = './cache/url_cache.json'
    
    search_cache = json.load(open(search_cache_path)) if os.path.exists(search_cache_path) else {}
    url_cache = json.load(open(url_cache_path)) if os.path.exists(url_cache_path) else {}

    # Setup output directory
    output_dir = f'./outputs/{args.dataset_name}.{args.model_name}.naive_rag'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize API client
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base_url,
    )

    # Create semaphore for concurrent API calls
    semaphore = asyncio.Semaphore(args.concurrent_limit)

    # Process all questions concurrently
    tasks = [
        process_single_sequence(
            question=item['Question'],
            client=client,
            semaphore=semaphore,
            args=args,
            search_cache=search_cache,
            url_cache=url_cache,
        )
        for item in filtered_data
    ]

    # Run all tasks with progress bar
    with tqdm(total=len(tasks)) as pbar:
        async def track_progress(task):
            result = await task
            pbar.update(1)
            return result
        
        results = await asyncio.gather(*[track_progress(task) for task in tasks])

    # Save results as JSON
    timestamp = time.strftime("%m.%d,%H:%M", time.localtime())
    output_path = os.path.join(output_dir, f'{args.split}.{timestamp}.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Create and save markdown files
    t = time.localtime()
    random_num = str(random.randint(0, 99)).zfill(2)
    markdown_dir = os.path.join(output_dir, f'markdown.{args.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.{random_num}')
    os.makedirs(markdown_dir, exist_ok=True)

    # Save individual markdown files for each result
    for i, result in enumerate(results):
        if result['response'].strip():  # Only save if response is not empty
            markdown_filename = f'article_{i+1}.md'
            
            # Add question as context at the top of the file
            question_context = f"Question: {result['question']}\n\n"
            
            with open(os.path.join(markdown_dir, markdown_filename), 'w', encoding='utf-8') as f:
                f.write(result['article'])

    # Save caches
    with open(search_cache_path, 'w', encoding='utf-8') as f:
        json.dump(search_cache, f, ensure_ascii=False, indent=2)
    with open(url_cache_path, 'w', encoding='utf-8') as f:
        json.dump(url_cache, f, ensure_ascii=False, indent=2)

    print("Process completed.")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
