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

from openai import AsyncOpenAI
from transformers import AutoTokenizer

from .webthinker_utils.prompts import (
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
from .webthinker_utils.evaluate import extract_answer_fn
from src.retrieval import E5_Retriever

# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

def extract_between(text, start_marker, end_marker):
    """Extracts text between two markers in a string."""
    try:
        pattern = re.escape(start_marker) + r"(.*?)" + re.escape(end_marker)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None
    except Exception as e:
        print(f"Error extracting between markers: {e}")
        return None

def extract_relevant_info_e5(results, top_k=5):
    """
    E5 검색 결과를 WebThinker/Search-o1 포맷으로 변환
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
    tokenizer=None,
    stop: List[str] = [END_SEARCH_QUERY],
    seed: int = None,
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
                    max_tokens=min(max_tokens, 32768),
                    stop=stop,
                    seed=seed,
                    extra_body={
                        'top_k': top_k,
                        'include_stop_str_in_output': True,
                        'repetition_penalty': repetition_penalty,
                        # 'min_p': min_p
                    },
                    timeout=1500,
                )
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
        tokenizer=None,
        seed=getattr(client, '_seed', None),  # Pass seed implicitly or explicitly
    )
    
    extracted_info = extract_answer_fn(raw_output, mode='infogen')
    return extracted_info

async def process_single_sequence(
    seq: Dict,
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
    search_cache: Dict,
    url_cache: Dict,
    retriever=None,
    tokenizer=None,
    turn: int = 0,
) -> Dict:
    """Process a single sequence through its entire reasoning chain"""
    sidx = seq.get('_sample_idx', 0)
    stotal = seq.get('_total_samples', 0)
    
    seq['step_stats'] = []

    if stotal > 0:
        print(f"[{sidx}/{stotal}] Started")
        
    MAX_TURNS = getattr(args, 'max_turn', 15)
        
    while not seq['finished'] and turn < MAX_TURNS:
        # Generate next step in reasoning
        text = await generate_response(
            client=client,
            prompt=seq['prompt'],
            semaphore=semaphore,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            top_k=getattr(args, 'top_k_sampling', 20),
            min_p=args.min_p,
            model_name=args.model_name,
            tokenizer=tokenizer,
            seed=getattr(args, 'seed', None)
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
                        if args.search_engine == "e5":
                           if retriever:
                               results = retriever.search(search_query, k=args.top_k)
                           else:
                               results = []
                        search_cache[search_query] = results
                    except Exception as e:
                        print(f"Error during search query '{search_query}' using {args.search_engine}: {e}")
                        search_cache[search_query] = {}
                        results = {}
                
                if args.search_engine == "e5":
                    relevant_info = extract_relevant_info_e5(results, top_k=args.top_k)
                else: 
                    relevant_info = []
                    
                seq['relevant_info'] = relevant_info

                # Process documents
                formatted_documents = ""
                # Since E5 provides full document in contents/page_info, we format directly
                for i, doc_info in enumerate(relevant_info):
                    # For E5, simulate context by getting page_info
                    raw_context = doc_info.get('page_info', '')
                    context = raw_context[:getattr(args, 'max_doc_len', 3000)*2] # simple truncation
                    doc_for_model = {
                         'title': doc_info.get('title', ''),
                         'url': doc_info.get('url', ''),
                         'snippet': doc_info.get('snippet', ''),
                         'context': context
                    }
                    formatted_documents += f"**Web Page {i + 1}:**\n"
                    formatted_documents += json.dumps(doc_for_model, ensure_ascii=False, indent=2) + "\n"

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

                search_docs_no_snippet = [{k: v for k, v in d.items() if k != 'snippet'} for d in relevant_info if isinstance(d, dict)]
                poisoned_flags = [d.get('is_poisoned', False) for d in relevant_info if isinstance(d, dict)]
                thought_before_search = text.split(BEGIN_SEARCH_QUERY)[0].strip() if BEGIN_SEARCH_QUERY in text else text.strip()

                # Generate webpage analysis
                analysis = await generate_webpage_to_reasonchain(
                    client=client,
                    original_question=seq['question'],
                    prev_reasoning=truncated_prev_reasoning,
                    search_query=search_query,
                    document=formatted_documents,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    top_k=getattr(args, 'top_k_sampling', 20),
                    min_p=args.min_p,
                    model_name=args.model_name,
                    semaphore=semaphore
                )

                # Output appending
                append_text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"
                seq['prompt'] += append_text
                seq['output'] += append_text
                seq['history'].append(append_text)
                
                seq['search_count'] += 1
                seq['executed_search_queries'].add(search_query)

                # Record for run_attack.py evaluation metrics
                _step = len(seq['step_stats'])
                seq['step_stats'].append({
                     'step': _step,
                     'thought': thought_before_search,
                     'search_query': search_query,
                     'action': f"Search[{search_query}]",
                     'observation': analysis,
                     'search_documents': search_docs_no_snippet,
                     'is_search': True,
                     'poisoned_flags': poisoned_flags,
                     'any_poisoned': any(poisoned_flags)
                })

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
            final_thought = text.strip()
            if final_thought:
                _step = len(seq['step_stats'])
                seq['step_stats'].append({
                    'step': _step,
                    'thought': final_thought,
                    'search_query': None,
                    'action': 'Finish[]',
                    'observation': None,
                    'search_documents': None,
                    'is_search': False,
                    'source': 'main',
                })
            seq['finished'] = True

        turn += 1

    # Extract final prediction
    seq['prediction'] = extract_answer_fn(seq['output'], mode='qa', extract_answer=True)
    seq['executed_search_queries'] = list(seq['executed_search_queries'])

    return seq

class SearchO1Agent:
    def __init__(self, args, tokenizer=None):
        self.args = args
        self.tokenizer = tokenizer
        self.client = None
        
        # Initialize Retriever
        if getattr(args, 'search_engine', None) == "e5":
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

        self.semaphore = None
        self.search_cache = {}
        self.url_cache = {}

    async def run_batch(self, inputs: List[Dict]):
        """
        inputs: List of dicts, each having 'question' and optionally other metadata.
        We format 'prompt' here according to run_search_o1.py logic.
        """
        if self.client is None:
             self.client = AsyncOpenAI(
                api_key=self.args.api_key,
                base_url=self.args.api_base_url,
            )
        self.semaphore = asyncio.Semaphore(getattr(self.args, 'concurrent_limit', 10))

        tasks = []
        for i, item in enumerate(inputs):
            question = item.get('question', item.get('query', ''))
            
            # Default to openqa formatting to match general run_search_o1 behavior
            # The exact subset (e.g. math/code/gpqa) might require specific flags, 
            # but usually for attack benchmark, it's openQA or singleqa style
            instruction = get_multiqa_search_o1_instruction(getattr(self.args, 'max_search_limit', 10))
            
            model_name_lower = self.args.model_name.lower() if self.args.model_name else ""
            if 'qwq' in model_name_lower or 'sky-t1' in model_name_lower:
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')
            elif 'deepseek' in model_name_lower:
                user_prompt = get_task_instruction_openqa(question, model_name='dpsk')
            else:
                user_prompt = get_task_instruction_openqa(question)

            prompt = instruction + user_prompt

            seq = {
                'prompt': prompt,
                'history': [],
                'output': "",
                'finished': False,
                'search_count': 0,
                'executed_search_queries': set(),
                'original_prompt': "",
                'question': question, 
                '_sample_idx': i+1,
                '_total_samples': len(inputs)
            }
            
            tasks.append(
                process_single_sequence(
                    seq=seq,
                    client=self.client,
                    semaphore=self.semaphore,
                    args=self.args,
                    search_cache=self.search_cache,
                    url_cache=self.url_cache,
                    retriever=self.retriever,
                    tokenizer=self.tokenizer
                )
            )
        
        results = await asyncio.gather(*tasks)
        return results
