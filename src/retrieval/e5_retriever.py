
import os
import glob
import json
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from logzero import logger
try:
    from .models.encoder import SimpleEncoder
except (ImportError, ValueError):
    from models.encoder import SimpleEncoder

class E5_Retriever:
    """
    A standalone retriever class for E5/Contriever models using the custom SimpleEncoder.
    """
    def __init__(
        self,
        corpus_path: str,
        index_dir: Optional[str] = None,
        poisoned_corpus_path: Optional[str] = None,
        poisoned_index_dir: Optional[str] = None,
        model_name: str = 'intfloat/e5-large-v2',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        batch_size: int = 32
    ):
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name
        
        print(f"Initializing E5_Retriever with model: {model_name}")
        self.encoder = SimpleEncoder(model_name, device, batch_size=batch_size)

        # 1. Resolve Index Paths and Auto-Index if needed
        if corpus_path:
            resolved_index_dir = index_dir or self._derive_index_path(corpus_path, model_name)
            self._ensure_index_exists(corpus_path, resolved_index_dir)
            clean_corpus = self._load_corpus(corpus_path)
            clean_embeddings = self._load_embeddings_from_dir(resolved_index_dir)
        else:
            clean_corpus = []
            clean_embeddings = None
        
        # [CRITICAL] Truncate corpus if index is smaller (indexing might be incomplete)
        if clean_embeddings is not None and len(clean_corpus) > clean_embeddings.shape[0]:
            print(f"Warning: Clean corpus ({len(clean_corpus)}) is larger than clean index ({clean_embeddings.shape[0]}). Truncating corpus to match.")
            clean_corpus = clean_corpus[:clean_embeddings.shape[0]]
            
        self.num_clean_docs = len(clean_corpus)
        
        if poisoned_corpus_path:
            resolved_poison_index_dir = poisoned_index_dir or self._derive_index_path(poisoned_corpus_path, model_name)
            self._ensure_index_exists(poisoned_corpus_path, resolved_poison_index_dir)
            poison_corpus = self._load_corpus(poisoned_corpus_path)
            poison_embeddings = self._load_embeddings_from_dir(resolved_poison_index_dir)
        else:
            poison_corpus = []
            poison_embeddings = None

        # 2. Merge Corpora
        self.corpus = clean_corpus + poison_corpus
        print(f"Total Corpus: {len(self.corpus)} (Clean: {len(clean_corpus)}, Poisoned: {len(poison_corpus)})")

        # 3. Merge Embeddings
        all_embs = []
        if clean_embeddings is not None:
            all_embs.append(clean_embeddings)
        if poison_embeddings is not None:
            all_embs.append(poison_embeddings)
            
        if all_embs:
            self.all_embeddings = torch.cat(all_embs, dim=0).to(self.device)
            print(f"Total Index loaded. Shape: {self.all_embeddings.shape}")
            
            # Final sanity check
            # if len(self.corpus) != self.all_embeddings.shape[0]:
                 # raise ValueError(f"CRITICAL: Corpus size ({len(self.corpus)}) does not match Index size ({self.all_embeddings.shape[0]})!")
        else:
            self.all_embeddings = None
            print("Warning: No embeddings loaded.")

    def _derive_index_path(self, corpus_path: str, model_name: str) -> str:
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

    def _ensure_index_exists(self, corpus_path: str, index_dir: str):
        """
        Check if index exists, otherwise create it.
        """
        if not os.path.exists(index_dir) or not glob.glob(os.path.join(index_dir, '*.pt')):
            print(f"Index not found at {index_dir}. Creating index...")
            self._create_index(corpus_path, index_dir)
        else:
            print(f"Index found at {index_dir}. Using existing index.")

    def _create_index(self, corpus_path: str, index_dir: str):
        os.makedirs(index_dir, exist_ok=True)
        corpus = self._load_corpus(corpus_path)
        if not corpus:
            print("Empty corpus, skipping indexing.")
            return

        print(f"Encoding {len(corpus)} documents...")
        all_embeddings = self.encoder.encode_corpus(corpus)
        
        # Save as a single shard for simplicity or multiple if large?
        # Let's save as single shard for now for simplicity, or chunk it.
        # Original code used shards? _load_embeddings_from_dir supports shards.
        # We save as 'embedding-shard-0.pt'
        save_path = os.path.join(index_dir, 'embedding-shard-0.pt')
        torch.save(all_embeddings.cpu(), save_path)
        print(f"Saved index to {save_path}")

    def _load_corpus(self, corpus_path: str) -> List[Dict]:
        corpus = []
        if not corpus_path: return corpus
        try:
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    corpus.append(json.loads(line))
        except FileNotFoundError:
            print(f"Warning: Corpus file not found at {corpus_path}.")
        return corpus

    def _load_embeddings_from_dir(self, index_dir: str) -> Optional[torch.Tensor]:
        print(f"Loading index shards from {index_dir}...")
        embeddings = []
        shard_files = sorted(glob.glob(os.path.join(index_dir, 'embedding-shard-*.pt')), 
                             key=lambda x: int(x.split('embedding-shard-')[-1].split('.pt')[0]) if '-' in x else 0)
        
        if not shard_files:
            return None
        
        for shard in shard_files:
            try:
                emb = torch.load(shard, map_location='cpu')
                embeddings.append(emb)
            except Exception as e:
                print(f"Error loading shard {shard}: {e}")

        if embeddings:
            return torch.cat(embeddings, dim=0) # Return on CPU for merging
        return None

    def encode_queries(self, queries: List[str]) -> torch.Tensor:
        # Delegate to the Clean SimpleEncoder
        return self.encoder.encode_queries(queries).to(self.device)

    # for oracle attack
    def add_corpus(self, corpus: List[Dict]):
        # embed new corpus
        if not corpus:
            return

        if self.all_embeddings is not None:
            new_embeddings = self.encoder.encode_corpus(corpus).to(self.all_embeddings.dtype).to(self.device)
            self.all_embeddings = torch.cat([self.all_embeddings, new_embeddings], dim=0)
        else:
            new_embeddings = self.encoder.encode_corpus(corpus).to(self.device)
            self.all_embeddings = new_embeddings
        
        self.corpus.extend(corpus)

        logger.info(f"Added {len(corpus)} new corpus to the retriever.")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for the query and return top-k documents.
        Returns a list of dicts: {'id': ..., 'title': ..., 'contents': ..., 'score': ...}
        """
        if self.all_embeddings is None:
            return []

        query_emb = self.encode_queries([query]) # [1, dim]
        
        # Ensure dtypes match (safety cast)
        if self.all_embeddings is not None:
            query_emb = query_emb.to(self.all_embeddings.dtype)

        # Dot product similarity
        scores = torch.matmul(query_emb, self.all_embeddings.T).squeeze(0) # [num_docs]
        
        # Top-k
        top_k_scores, top_k_indices = torch.topk(scores, k=k)
        
        results = []
        for i in range(k):
            idx = top_k_indices[i].item()
            score = top_k_scores[i].item()
            
            doc = self.corpus[idx] if idx < len(self.corpus) else {}
            results.append({
                'id': doc.get('_id', doc.get('id', str(idx))),
                'title': doc.get('title', ''),
                'contents': doc.get('text', doc.get('contents', '')),
                'score': score,
                'is_poisoned': idx >= self.num_clean_docs
            })
            
        return results

if __name__ == "__main__":
    pass
