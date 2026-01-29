import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

@dataclass
class CommonArguments(TrainingArguments):
    """
    Unified arguments for all RAG models (Corag, ReAct, WebThinker).
    Includes experimental modes.
    """
    # --- Basic Execution ---
    max_len: int = field(
        default=4096,
        metadata={"help": "The maximum total input sequence length."}
    )
    dry_run: bool = field(
        default=False,
        metadata={'help': 'Set True for debugging (process fewer examples)'}
    )
    
    # --- Experiment Modes ---
    attack_mode: str = field(
        default='base',
        metadata={
            'help': 'Experiment mode: base, static_main_main, static_main_sub, static_sub_sub, dynamic_retrieval, oracle_injection, surrogate',
            'choices': ['base', 'static_main_main', 'static_main_sub', 'static_sub_sub', 'dynamic_retrieval', 'oracle_injection', 'surrogate']
        }
    )
    
    # --- Paths (Can be overridden by specific scripts) ---
    index_dir: str = field(default="../datasets/hotpotqa/e5_index", metadata={'help': 'Path to E5 index'})
    corpus_path: str = field(default="../datasets/hotpotqa/corpus.jsonl", metadata={'help': 'Path to clean corpus'})
    
    # For Static Modes (2-4)
    poisoned_index_dir: Optional[str] = field(default=None, metadata={'help': 'Path to poisoned index (for static modes)'})
    poisoned_corpus_path: Optional[str] = field(default=None, metadata={'help': 'Path to poisoned corpus (for static modes)'})
    
    # --- Retriever Config ---
    retriever_model: str = field(
        default="/home/work/Redteaming/data1/VIDEO_HALLUCINATION/hf_cache/hub/models--intfloat--e5-large-v2/snapshots/f169b11e22de13617baa190a028a32f3493550b6",
        metadata={'help': 'Path to retriever model weights'}
    )
    
    # --- Attack/Model Params ---
    model_path: Optional[str] = field(default=None, metadata={'help': 'Path to generator model (LLM)'})
    adv_model_path: Optional[str] = field(default=None, metadata={'help': 'Path to adversary model (for dynamic modes)'})
    
    # --- Remote Attacker Config ---
    attacker_api_base: Optional[str] = field(default=None, metadata={'help': 'vLLM API Base URL for Attacker (e.g. http://localhost:8001/v1)'})
    attacker_api_key: Optional[str] = field(default="EMPTY", metadata={'help': 'API Key for Attacker vLLM'})
    attacker_model_name: Optional[str] = field(default=None, metadata={'help': 'Model name to use in API call'})

    # --- Legacy/Corag Specific (kept for compatibility) ---
    eval_task: str = field(default='hotpotqa', metadata={'help': 'evaluation task'})
    eval_split: str = field(default='validation', metadata={'help': 'evaluation split'})
    max_path_length: int = field(default=3, metadata={"help": "maximum path length (reasoning steps)"})
    decode_strategy: str = field(default='greedy', metadata={'help': 'decoding strategy'})
    sample_temperature: float = field(default=0.7, metadata={"help": "temperature for sampling"})
    num_threads: int = field(default=32, metadata={"help": "number of threads for retriever search"})
    
    # Sharding
    data_path: Optional[str] = field(default=None, metadata={'help': 'Data path'})
    trajs_path: Optional[str] = field(default=None, metadata={'help': 'Trajs path'})
    num_shards: Optional[int] = field(default=None, metadata={'help': 'Total number of shards'})
    shard_id: Optional[int] = field(default=None, metadata={'help': 'Current shard id (0-indexed)'})

    def __post_init__(self):
        super().__post_init__()
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
