
import os
import sys
import glob
import json
import torch
import tqdm
from typing import List, Dict, Optional
from src.retrieval.models.encoder import SimpleEncoder

def load_jsonl(path: str) -> List[Dict]:
    print(f"Loading corpus from {path}...")
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} documents.")
    return data

def derive_index_path(corpus_path: str, model_name: str) -> str:
    """
    Derive index directory path from corpus path and model name.
    Format: {corpus_dir}/{model_safe_name}_index_{corpus_filename_no_ext}
    """
    corpus_dir = os.path.dirname(os.path.abspath(corpus_path))
    corpus_name = os.path.splitext(os.path.basename(corpus_path))[0]
    
    # Safe model name (handle paths)
    if os.path.isdir(model_name):
            # It's a path, take the last directory name
            model_safe_name = os.path.basename(os.path.normpath(model_name))
    else:
            model_safe_name = model_name.replace('/', '_')
            
    index_name = f"{model_safe_name}_index_{corpus_name}"
    return os.path.join(corpus_dir, index_name)

def build_index(
    corpus_path: str,
    index_dir: str,
    model_name: str = 'intfloat/e5-large-v2',
    batch_size: int = 1024,
    shard_size: int = 100000,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Builds the E5/Contriever index for the given corpus and saves it to index_dir.
    """
    
    # Check if index already exists
    if os.path.exists(index_dir) and glob.glob(os.path.join(index_dir, 'embedding-shard-*.pt')):
        print(f"Index already exists at {index_dir}. Skipping build.")
        return

    print(f"Building index at {index_dir} using {model_name}...")
    os.makedirs(index_dir, exist_ok=True)

    # Load Corpus
    corpus = load_jsonl(corpus_path)
    if not corpus:
        print("Empty corpus, skipping indexing.")
        return

    # Initialize Encoder
    print(f"Initializing Encoder: {model_name}")
    encoder = SimpleEncoder(model_name_or_path=model_name, device=device, batch_size=batch_size)

    # Encoding Loop
    print(f"Starting encoding with shard size {shard_size}...")
    
    total_docs = len(corpus)
    for shard_idx, start_idx in enumerate(range(0, total_docs, shard_size)):
        end_idx = min(start_idx + shard_size, total_docs)
        shard_corpus = corpus[start_idx:end_idx]
        
        print(f"Encoding shard {shard_idx}: docs {start_idx} to {end_idx}")
        
        # SimpleEncoder.encode_corpus expects a list of dicts with 'title' and 'text'/'contents'
        # Ensure compatibility if keys differ
        normalized_corpus = []
        for doc in shard_corpus:
            norm_doc = {
                'title': doc.get('title', ''),
                'text': doc.get('contents', doc.get('text', '')) # SimpleEncoder uses 'text' or 'contents'. We normalize to what SimpleEncoder uses inside? 
                # Actually SimpleEncoder.encode_corpus looks for 'text' or 'contents'. 
                # We can just pass doc if it has those keys. But safe to normalize.
            }
            normalized_corpus.append(norm_doc)

        embeddings = encoder.encode_corpus(normalized_corpus)
        
        # Save Shard
        save_path = os.path.join(index_dir, f'embedding-shard-{shard_idx}.pt')
        torch.save(embeddings.cpu(), save_path)
        print(f"Saved shard {shard_idx} to {save_path}")

    print("Indexing completed successfully.")
