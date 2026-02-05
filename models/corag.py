
import os
import sys
import json
import torch
import asyncio
from typing import List, Dict
from tqdm import tqdm
from datasets import Dataset

# Add corag_utils to sys.path to allow imports from it
current_dir = os.path.dirname(os.path.abspath(__file__))
corag_utils_path = os.path.join(current_dir, 'corag_utils')
if corag_utils_path not in sys.path:
    sys.path.insert(0, corag_utils_path)

# Import from corag_utils (which mirrors corag/src)
# These imports rely on corag_utils being in sys.path
from agent.corag_agent import CoRagAgent
from agent.agent_utils import RagPath
from vllm_client import VllmClient
from utils import save_json_to_file

# Import Unified components
from src.retrieval import E5_Retriever

class CoRagModel:
    def __init__(self, args):
        self.args = args
        
        # Initialize VllmClient
        # args should have vllm_host and vllm_port
        self.vllm_client = VllmClient(
            model=args.model_name_or_path if hasattr(args, 'model_name_or_path') else args.model_name,
            host=getattr(args, 'vllm_host', 'localhost'),
            port=getattr(args, 'vllm_port', 8000)
        )
        
        # Initialize E5 Retriever
        if args.search_engine == "e5":
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

        # Initialize CoRagAgent
        # CoRagAgent requires a corpus dataset usually, but if using E5_Retriever, it might be optional or used for title lookup
        # In generate_trajs.py: corpus = Dataset.from_dict({})
        self.corpus = Dataset.from_dict({})
        
        self.agent = CoRagAgent(
            vllm_client=self.vllm_client,
            corpus=self.corpus,
            retriever=self.retriever
        )

    def run_batch(self, inputs: List[Dict]):
        """
        Run CoRag inference on a batch of inputs.
        Each input dict must have 'question' (or 'query').
        """
        results = []
        # CoRag uses sync VllmClient, so we don't need async here necessarily, 
        # but to keep consistent with other agents, we might want to wrap in async or run sequentially.
        # Since VllmClient in corag uses requests.post (blocking), we run sequentially.
        
        for item in tqdm(inputs, desc="CoRag Inference"):
            query = item.get('query', item.get('question'))
            if not query:
                continue
            
            # Logic from generate_trajs.py _generate_single_example
            # Mapping args
            decode_strategy = getattr(self.args, 'decode_strategy', 'greedy')
            max_path_length = getattr(self.args, 'max_path_length', 3)
            temperature = getattr(self.args, 'temperature', 0.0)
            
            try:
                if decode_strategy == 'greedy' or max_path_length < 1:
                    path: RagPath = self.agent.sample_path(
                        query=query, 
                        task_desc='answer multi-hop questions',
                        max_path_length=max_path_length,
                        temperature=0., 
                        max_tokens=64
                    )
                elif decode_strategy == 'tree_search':
                    path: RagPath = self.agent.tree_search(
                        query=query, 
                        task_desc='answer multi-hop questions',
                        max_path_length=max_path_length,
                        temperature=temperature, 
                        max_tokens=64
                    )
                elif decode_strategy == 'best_of_n':
                    path: RagPath = self.agent.best_of_n(
                        query=query, 
                        task_desc='answer multi-hop questions',
                        max_path_length=max_path_length,
                        temperature=temperature,
                        n=getattr(self.args, 'best_n', 4),
                        max_tokens=64
                    )
                else:
                    path = None # Handle error
                
                if path:
                    all_path_docs = []
                    for docs in path.past_documents:
                        all_path_docs.extend(docs)
                    
                    prediction = self.agent.generate_final_answer(
                        corag_sample=path,
                        task_desc='answer multi-hop questions',
                        documents=all_path_docs,
                        max_message_length=getattr(self.args, 'max_len', 4096),
                        temperature=0., 
                        max_tokens=128
                    )
                    
                    # steps for logging
                    steps = []
                    for i in range(len(path.past_subqueries)):
                        step = {
                            'subquery': path.past_subqueries[i],
                            'retrieved_results': path.past_retriever_results[i] if path.past_retriever_results else [],
                            'subanswer': path.past_subanswers[i],
                        }
                        steps.append(step)

                    result_item = {
                        'id': item.get('id', ''),
                        'query': query,
                        'correct_answer': item.get('answer', ''),
                        'steps': steps,
                        'final_answer': prediction,
                        'prediction': prediction
                    }
                    results.append(result_item)
                else:
                    results.append({'id': item.get('id'), 'error': 'Failed to generate path'})

            except Exception as e:
                print(f"Error processing {query}: {e}")
                results.append({'id': item.get('id'), 'error': str(e)})

        return results
