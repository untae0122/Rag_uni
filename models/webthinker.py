
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
import sys
from collections import Counter, defaultdict

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from .webthinker_utils.prompts import (
    get_deep_web_explorer_instruction, 
    get_web_page_reader_instruction,
    get_search_intent_instruction,
    get_click_intent_instruction,
    get_multiqa_search_o1_instruction, 
    get_task_instruction_openqa, 
)
from .webthinker_utils.evaluate import extract_answer_fn
from src.retrieval import E5_Retriever
# from search.bing_search import ... (Omitted as we focus on E5)

# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

BEGIN_CLICK_LINK = "<|begin_click_link|>"
END_CLICK_LINK = "<|end_click_link|>"
BEGIN_CLICK_RESULT = "<|begin_click_result|>"
END_CLICK_RESULT = "<|end_click_result|>"

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

invalid_search_queries = [
    "and end with",
    "search query",
    "query",
    "your query here",
    "your query",
    "your search query",
]

def extract_relevant_info_e5(results, top_k=5):
    """
    E5 검색 결과를 WebThinker 포맷으로 변환
    results: E5_Retriever.search()가 반환한 리스트 [{'id', 'title', 'contents', 'score', 'is_poisoned'}]
    """
    relevant_info = []
    if not isinstance(results, list):
        return relevant_info
    
    for i, doc in enumerate(results[:top_k]):
        # E5_Retriever는 'contents' 키를 반환
        title = doc.get('title', f"Document {i+1}")
        text = doc.get('contents', doc.get('text', ''))  # 'contents' 우선
        doc_id = doc.get('id', f"doc_id_{i}")
        
        relevant_info.append({
            'title': title,
            'url': doc_id,  # 실제 URL이 아니라 doc_id
            'snippet': text[:500] + "..." if len(text) > 500 else text,
            'page_info': text,  # 전체 텍스트
            'is_poisoned': doc.get('is_poisoned', False)  # 추가 정보
        })
    return relevant_info


def extract_between(text, start_marker, end_marker):
    """Extracts text between two markers in a string."""
    try:
        pattern = re.escape(end_marker[::-1]) + r"(.*?)" + re.escape(start_marker[::-1])
        # Run pattern matching with timeout
        matches = re.findall(pattern, text[::-1], flags=re.DOTALL)
        if matches:
            return matches[0][::-1].strip()
        return None
    except Exception as e:
        return None

def format_search_results(relevant_info: List[Dict]) -> str:
    """Format search results into a readable string"""
    formatted_documents = ""
    for i, doc_info in enumerate(relevant_info):
        doc_for_model = {
            'title': doc_info.get('title', '').replace('<b>','').replace('</b>',''),
            'url': doc_info.get('url', ''),
            'snippet': doc_info.get('snippet', '').replace('<b>','').replace('</b>',''),
            'page_info': doc_info.get('page_info', '')
        }
        formatted_documents += f"***Web Page {i + 1}:***\n"
        formatted_documents += json.dumps(doc_for_model, ensure_ascii=False, indent=2) + "\n"
    return formatted_documents


async def generate_response(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    generate_mode: str = "chat",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 32768,
    repetition_penalty: float = 1.0,
    top_k: int = 1,
    min_p: float = 0.0,
    model_name: str = "QwQ-32B",
    stop: List[str] = [END_SEARCH_QUERY],
    retry_limit: int = 3,
    bad_words: List[str] = None,
    tokenizer=None,
    aux_tokenizer=None
) -> Tuple[str, str]:
    """Generate a single response with retry logic"""
    if bad_words is None:
        bad_words = [f"{END_SEARCH_RESULT}\n\n{tokenizer.eos_token if tokenizer else ''}"]
    for attempt in range(retry_limit):
        try:
            async with semaphore:
                if generate_mode == "chat":
                    messages = [{"role": "user", "content": prompt}]
                    # Determine which tokenizer/model to use logic is external or simple here
                    # For simplicity, if tokenizer provided use it
                    if tokenizer and ('qwq' in model_name.lower() or 'deepseek' in model_name.lower() or 'r1' in model_name.lower()):
                        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    elif aux_tokenizer:
                        formatted_prompt = aux_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    else:
                        formatted_prompt = prompt # Fallback

                    if ('deepseek' in model_name.lower() or 'r1' in model_name.lower()) and "<think>\n" not in formatted_prompt:
                        formatted_prompt = formatted_prompt + "<think>\n"
                else:
                    formatted_prompt = prompt

                response = await client.completions.create(
                    model=model_name,
                    prompt=formatted_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop,
                    extra_body={
                        'top_k': top_k,
                        'include_stop_str_in_output': True,
                        'repetition_penalty': repetition_penalty,
                        # 'bad_words': bad_words,
                        # 'min_p': min_p
                    },
                    timeout=3600,
                )
                return formatted_prompt, response.choices[0].text
        except Exception as e:
            if "maximum context length" in str(e).lower():
                max_tokens = max_tokens // 2
            if attempt == retry_limit - 1:
                return "", ""
            await asyncio.sleep(1 * (attempt + 1))
    return "", ""


async def generate_deep_web_explorer(
    client: AsyncOpenAI,
    aux_client: AsyncOpenAI,
    search_query: str,
    document: str,
    search_intent: str,
    args: argparse.Namespace,
    search_cache: Dict,
    url_cache: Dict,
    semaphore: asyncio.Semaphore,
    tokenizer,
    aux_tokenizer,
    retriever=None,
) -> Tuple[str, str, List[Dict]]:
    """
    Generate deep web exploration with multiple search and click operations.
    Returns (output, original_prompt, explorer_steps).
    """
    prompt = get_deep_web_explorer_instruction(search_query=search_query, search_intent=search_intent, search_result=document)
    output = ""
    original_prompt = ""
    explorer_steps = []  
    total_tokens = len(prompt.split())  
    MAX_TOKENS = 30000
    MAX_INTERACTIONS = 10  
    clicked_urls = set()  
    executed_search_queries = set()  
    total_interactions = 0
    finished = False
    first_generation = True

    while True:
        formatted_prompt, response = await generate_response(
            client=client if 'qwq' in args.model_name.lower() else aux_client,
            model_name=args.model_name if 'qwq' in args.model_name.lower() else args.aux_model_name,
            prompt=prompt,
            semaphore=semaphore,
            generate_mode="chat" if first_generation else "completion",
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k_sampling,
            min_p=args.min_p,
            stop=[END_SEARCH_QUERY, END_CLICK_LINK],
            tokenizer=tokenizer,
            aux_tokenizer=aux_tokenizer
        )

        if first_generation:
            original_prompt = formatted_prompt
            prompt = formatted_prompt
        
        clean_response = response.replace('</think>\n','')
        output += clean_response
        prompt += clean_response
        
        total_tokens = len(prompt.split()) + len(response.split())
        first_generation = False

        if total_tokens >= MAX_TOKENS or total_interactions >= MAX_INTERACTIONS:
            break

        # Check for search query
        if response.rstrip().endswith(END_SEARCH_QUERY):
            new_query = extract_between(response, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
            total_interactions += 1
            if new_query is None or END_SEARCH_QUERY in new_query or len(new_query) <= 5 or new_query in invalid_search_queries:
                continue
            if new_query:
                if new_query in executed_search_queries:
                    search_result = f"\n{BEGIN_SEARCH_RESULT}\nYou have already searched for this query. Please use the previously found information.\n{END_SEARCH_RESULT}\n\nOkay,"
                    output += search_result
                    prompt += output
                    total_tokens += len(search_result.split())
                    continue

                executed_search_queries.add(new_query)  
                
                # Execute search
                if new_query in search_cache:
                    results = search_cache[new_query]
                else:
                    try:
                        if args.search_engine == "e5":
                            if retriever:
                                results = retriever.search(new_query, k=args.top_k)
                            else:
                                results = []
                        # else: # Bing/Serper not supported in this unified version for now
                        #    results = {} 
                        search_cache[new_query] = results
                    except Exception as e:
                        results = {}

                if args.search_engine == "e5":
                    relevant_info = extract_relevant_info_e5(results, top_k=args.top_k)
                # else:
                #    relevant_info = []

                formatted_documents = format_search_results(relevant_info)
                
                thought_explorer = clean_response.split(BEGIN_SEARCH_QUERY)[0].strip() if BEGIN_SEARCH_QUERY in clean_response else clean_response.strip()
                search_docs_no_snippet = [{k: v for k, v in d.items() if k != 'snippet'} for d in relevant_info if isinstance(d, dict)]
                explorer_steps.append({
                    "type": "search",
                    "thought": thought_explorer,
                    "search_query": new_query,
                    "observation": formatted_documents,
                    "search_documents": search_docs_no_snippet,
                })
                
                search_result = f"\n{BEGIN_SEARCH_RESULT}\n{formatted_documents}\n{END_SEARCH_RESULT}\n"
                output += search_result
                prompt += output
                total_tokens += len(search_result.split())
                
        # Check for click link (simplified for E5)
        elif response.rstrip().endswith(END_CLICK_LINK):
            url = extract_between(response, BEGIN_CLICK_LINK, END_CLICK_LINK)
            total_interactions += 1
            _, click_intent = await generate_response(
                client=aux_client,
                model_name=args.aux_model_name,
                max_tokens=1000,
                prompt=get_click_intent_instruction(output),
                semaphore=semaphore,
                tokenizer=tokenizer, 
                aux_tokenizer=aux_tokenizer
            )

            if url and click_intent:
                if url in clicked_urls:
                    click_result = f"\n{BEGIN_CLICK_RESULT}\nYou have already clicked this URL.\n{END_CLICK_RESULT}\n\nOkay,"
                    output += click_result
                    prompt += output
                    total_tokens += len(click_result.split())
                    continue

                clicked_urls.add(url)
                
                if args.search_engine == "e5":
                    content = "E5 retriever already contains full document content. No need to fetch."
                else:
                    content = "Fetching unsupported."

                has_error = any(indicator.lower() in content.lower() for indicator in error_indicators) or content == ''
                
                if has_error:
                    summary = "Unable to fetch the page content. You can try other links."
                else:
                    reader_prompt = get_web_page_reader_instruction(click_intent, content)
                    _, summary = await generate_response(
                        client=aux_client,
                        prompt=reader_prompt,
                        semaphore=semaphore,
                        max_tokens=3600,
                        model_name=args.aux_model_name,
                        tokenizer=tokenizer,
                        aux_tokenizer=aux_tokenizer
                    )

                click_result = f"\n{BEGIN_CLICK_RESULT}\n{summary}\n{END_CLICK_RESULT}\n"
                output += click_result
                prompt += output
                total_tokens += len(click_result.split())
        
        else:
            finished = True
            break

    if not finished and (total_tokens >= MAX_TOKENS or total_interactions >= MAX_INTERACTIONS):
        output += f"\n{BEGIN_CLICK_RESULT}\nYou have reached the limit for clicking links.\n{END_CLICK_RESULT}\n\nOK, I will now provide the final information based on my collected information.\n\n**Final Information:**"
        prompt += output
        _, final_response = await generate_response(
            client=client if 'qwq' in args.model_name.lower() else aux_client,
            model_name=args.model_name if 'qwq' in args.model_name.lower() else args.aux_model_name,
            prompt=prompt,
            semaphore=semaphore,
            generate_mode="completion",
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=512,
            repetition_penalty=1.2,
            top_k=args.top_k_sampling,
            min_p=args.min_p,
            tokenizer=tokenizer,
            aux_tokenizer=aux_tokenizer
        )
        output += final_response

    return output, original_prompt, explorer_steps


async def process_single_sequence(
    seq: Dict,
    client: AsyncOpenAI,
    aux_client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    search_cache: Dict,
    url_cache: Dict,
    retriever=None,
    tokenizer=None,
    aux_tokenizer=None
) -> Dict:
    sidx = seq.get('_sample_idx', 0)
    stotal = seq.get('_total_samples', 0)

    MAX_TOKENS = 40000
    total_tokens = len(seq['prompt'].split())
    
    seq['web_explorer'] = []
    seq['step_stats'] = []

    formatted_prompt, response = await generate_response(
        client=client,
        model_name=args.model_name,
        prompt=seq['prompt'],
        semaphore=semaphore,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        top_k=args.top_k_sampling,
        min_p=args.min_p,
        stop=[END_SEARCH_QUERY],
        tokenizer=tokenizer,
        aux_tokenizer=aux_tokenizer
    )
    
    if stotal > 0:
        print(f"[{sidx}/{stotal}] Started")
    
    clean_response = response.replace('</think>\n', '')
    tokens_this_response = len(response.split())
    total_tokens += tokens_this_response
    
    seq['output'] += clean_response
    seq['history'].append(clean_response)
    seq['original_prompt'] = formatted_prompt
    seq['prompt'] = formatted_prompt + clean_response

    while not seq['finished']:
        if not seq['output'].rstrip().endswith(END_SEARCH_QUERY):
            final_thought = clean_response.strip()
            if final_thought:
                _step = len(seq['step_stats'])
                seq['step_stats'].append({
                    'step': _step,
                    'thought': final_thought,
                    'search_query': None,
                    'action': 'Finish[]',
                    'observation': '[DEBUG]Answer generated without additional search',
                    'search_documents': None,
                    'extracted_info': None,
                    'is_search': False,
                    'source': 'main',
                })
            seq['finished'] = True
            break
        
        search_query = extract_between(response, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
        seq['search_count'] += 1
        current_step = seq['search_count']
        if stotal > 0:
             print(f"[{sidx}/{stotal}] Step {current_step}")

        thought_before_search = clean_response.split(BEGIN_SEARCH_QUERY)[0].strip() if BEGIN_SEARCH_QUERY in clean_response else clean_response.strip()

        if seq['search_count'] <= args.max_search_limit and total_tokens < MAX_TOKENS:
            if search_query is None or len(search_query) <= 5 or END_SEARCH_QUERY in search_query or search_query in invalid_search_queries: 
                continue

            if search_query in seq['executed_search_queries']:
                append_text = f"\n\n{BEGIN_SEARCH_RESULT}You have already searched for this query.{END_SEARCH_RESULT}\n\nOkay,"
                seq['prompt'] += append_text
                seq['output'] += append_text
                seq['history'].append(append_text)
                total_tokens += len(append_text.split())
                continue

            _, search_intent = await generate_response(
                client=aux_client,
                model_name=args.aux_model_name,
                max_tokens=1000,
                prompt=get_search_intent_instruction(seq['output']),
                semaphore=semaphore,
                tokenizer=tokenizer,
                aux_tokenizer=aux_tokenizer
            )

            results = {}
            if search_query in search_cache:
                results = search_cache[search_query]
            else:
                try:
                    if args.search_engine == "e5":
                        print(f"[Very Warning]Search query {search_query} found in search cache")
                        if retriever:
                            results = retriever.search(search_query, k=args.top_k)
                        else:
                            results = []
                    search_cache[search_query] = results
                except Exception as e:
                    results = {}

            if args.search_engine == "e5":
                relevant_info = extract_relevant_info_e5(results, top_k=args.top_k)
            else:
                relevant_info = []

            formatted_documents = format_search_results(relevant_info)

            analysis, explorer_prompt, explorer_steps = await generate_deep_web_explorer(
                client=client,
                aux_client=aux_client,
                search_query=search_query,
                search_intent=search_intent,
                document=formatted_documents,
                args=args,
                search_cache=search_cache,
                url_cache=url_cache,
                semaphore=semaphore,
                retriever=retriever,
                tokenizer=tokenizer,
                aux_tokenizer=aux_tokenizer
            )

            extracted_info = extract_answer_fn(analysis, mode='summary')

            seq['web_explorer'].append({
                "search_query": search_query,
                "Input": explorer_prompt,
                "Output": analysis,
                "Extracted_info": extracted_info
            })
            
            append_text = f"\n\n{BEGIN_SEARCH_RESULT}{extracted_info}{END_SEARCH_RESULT}\n\n"
            seq['prompt'] += append_text
            seq['output'] += append_text
            seq['history'].append(append_text)
            
            main_search_docs = [{k: v for k, v in d.items() if k != 'snippet'} for d in relevant_info if isinstance(d, dict)]
            
            # Calculate poisoned flags for main determination
            poisoned_flags = [d.get('is_poisoned', False) for d in relevant_info if isinstance(d, dict)]
            
            # 1. Main Search Step
            _step = len(seq['step_stats'])
            seq['step_stats'].append({
                 'step': _step,
                 'thought': thought_before_search,
                 'search_query': search_query,
                 'search_intent': search_intent,
                 'action': f"Search[{search_query}]",
                 'observation': extracted_info,
                 'search_documents': main_search_docs,
                 'extracted_info': None,
                 'is_search': True,
                 'source': 'main',
                 'poisoned_flags': poisoned_flags,
                 'any_poisoned': any(poisoned_flags)
            })
            
            # 2. Web Explorer Steps
            for es in explorer_steps:
                 if es.get("type") != "search":
                     continue
                 _step = len(seq['step_stats'])
                 seq['step_stats'].append({
                    'step': _step,
                    'thought': es.get('thought', ''),
                    'search_query': es.get('search_query', ''),
                    'action': f"Search[{es.get('search_query', '')}]",
                    'observation': es.get('observation', ''),
                    'search_documents': es.get('search_documents', []),
                    'extracted_info': None,
                    'is_search': True,
                    'source': 'web_explorer',
                 })

            # 3. Conclusion Step
            _step = len(seq['step_stats'])
            seq['step_stats'].append({
                'step': _step,
                'thought': extracted_info,
                'search_query': None,
                'action': None,
                'observation': None,
                'search_documents': None,
                'extracted_info': extracted_info,
                'is_search': False,
                'source': 'web_explorer'
            })
            
            total_tokens += len(append_text.split())
            seq['executed_search_queries'].add(search_query)

        else:
             # Limit reached
             append_text = f"\n\n{BEGIN_SEARCH_RESULT} You have reached the maximum number of searches. Please answer based on your current knowledge. {END_SEARCH_RESULT}\n\n"
             seq['prompt'] += append_text
             seq['output'] += append_text
             seq['history'].append(append_text)
             total_tokens += len(append_text.split())
             # Force finish next loop?
        
        # Next generation
        formatted_prompt, response = await generate_response(
            client=client,
            model_name=args.model_name,
            prompt=seq['prompt'],
            semaphore=semaphore,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k_sampling,
            min_p=args.min_p,
            stop=[END_SEARCH_QUERY],
            tokenizer=tokenizer,
            aux_tokenizer=aux_tokenizer
        )
        
        clean_response = response.replace('</think>\n', '')
        tokens_this_response = len(response.split())
        total_tokens += tokens_this_response
        
        seq['output'] += clean_response
        seq['history'].append(clean_response)
        seq['original_prompt'] = formatted_prompt # Updates
        seq['prompt'] = formatted_prompt + clean_response
        
        seq['prompt'] = formatted_prompt + clean_response
        
    # Extract final answer for metric calculation
    # Using 'qa' mode as default for HotPotQA style
    seq['answer'] = extract_answer_fn(seq['output'], mode='qa')
    
    # Convert sets to lists for JSON serialization
    seq['executed_search_queries'] = list(seq['executed_search_queries'])
    
    return seq


class WebThinkerAgent:
    def __init__(self, args, tokenizer=None, aux_tokenizer=None):
        self.args = args
        self.tokenizer = tokenizer
        self.aux_tokenizer = aux_tokenizer
        
        # Initialize Clients
        self.client = AsyncOpenAI(
            api_key=args.api_key,
            base_url=args.api_base_url,
        )
        self.aux_client = AsyncOpenAI(
            api_key=args.aux_api_key,
            base_url=args.aux_api_base_url,
        )
        
        # Initialize Retriever
        if args.search_engine == "e5":
            print(f"Initializing Retriever from {args.index_dir}")
            retrieval_model_name = getattr(args, 'retrieval_model_name', 'intfloat/e5-large-v2') 
            self.retriever = E5_Retriever(
                corpus_path=args.corpus_path,
                index_dir=args.index_dir,
                poisoned_corpus_path=args.poisoned_corpus_path,
                poisoned_index_dir=args.poisoned_index_dir,
                model_name=retrieval_model_name,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.retriever = None

        self.semaphore = asyncio.Semaphore(getattr(args, 'concurrent_limit', 10))
        self.search_cache = {}
        self.url_cache = {}

    async def run_batch(self, inputs: List[Dict]):
        """
        inputs: List of dicts, each having 'prompt' and 'question' etc.
        """
        tasks = []
        for i, item in enumerate(inputs):
            # Prepare sequence dict
            seq = {
                'prompt': item['prompt'], # Prompt should be pre-constructed
                'history': [],
                'output': "",
                'finished': False,
                'search_count': 0,
                'executed_search_queries': set(),
                'original_prompt': "",
                'question': item.get('question', ''), # Metadata
                '_sample_idx': i+1,
                '_total_samples': len(inputs)
            }
            
            tasks.append(
                process_single_sequence(
                    seq=seq,
                    client=self.client,
                    aux_client=self.aux_client,
                    semaphore=self.semaphore,
                    args=self.args,
                    search_cache=self.search_cache,
                    url_cache=self.url_cache,
                    retriever=self.retriever,
                    tokenizer=self.tokenizer,
                    aux_tokenizer=self.aux_tokenizer
                )
            )
        
        results = await asyncio.gather(*tasks)
        return results
