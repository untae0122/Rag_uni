# RAG Unified Experiment Guide

This repository contains the unified codebase for conducting adversarial attacks on RAG agents (ReAct, WebThinker, Corag).
The `src/common/config.py` and `src/attacks/attack_manager.py` handle the core logic for unified experiment modes.

## Data Preparation & Directory Structure
To run all experiments (1-6) without issues, ensure your data directory is structured as follows. 
It is recommended to place the `datasets` folder parallel to `rag_unified` or update the paths in your commands accordingly.

```
/path/to/datasets/
└── hotpotqa/
    ├── e5_index/                   # [Required All] FAISS index folder for Clean Corpus
    │   ├── index.faiss
    │   └── index.pkl
    ├── corpus.jsonl                # [Required All] Clean Corpus (Source of knowledge)
    ├── hotpotqa.json               # [Required All] Input Questions (Evaluation Dataset)
    │                               # Format: [{"id":..., "question":..., "correct_answer":..., "incorrect_answer":...}]
    ├── poisoned_main_main.jsonl    # [Required Exp 2] Static Poisoned Corpus (Main-Main)
    ├── poisoned_main_sub.jsonl     # [Required Exp 3] Static Poisoned Corpus (Main-Sub)
    └── poisoned_sub_sub.jsonl      # [Required Exp 4] Static Poisoned Corpus (Sub-Sub)
```

### File Format Requirements
1.  **Input Dataset (`hotpotqa.json`)**
    ```json
    [
      {
        "id": "q1",
        "question": "Query text...",
        "correct_answer": "Real Answer",
        "incorrect_answer": "Target Answer"
      }
    ]
    ```
2.  **Poisoned Corpus (`poisoned_*.jsonl`)**
    ```json
    {"_id": "p1", "title": "...", "text": "Poisoned content...", "metadata": {"is_poisoned": true}}
    ```

---

## Experiment Modes
We support 7 experiment modes (currently 1-6 are fully implemented):

| ID | Mode Name | Description | Key Arguments |
|---|---|---|---|
| **1** | `base` | Clean Baseline (No Attack) | `--attack_mode base` |
| **2** | `static_main_main` | Static Poisoning (Main Query & Target) | `--attack_mode static_main_main` `--poisoned_corpus_path ...` |
| **3** | `static_main_sub` | Static Poisoning (Main Query & Sub Target) | `--attack_mode static_main_sub` `--poisoned_corpus_path ...` |
| **4** | `static_sub_sub` | Static Poisoning (Sub Query & Sub Target) | `--attack_mode static_sub_sub` `--poisoned_corpus_path ...` |
| **5** | `dynamic_retrieval` | Dynamic Attack (Oracle-Soft) - Real-time poisoning of retriever | `--attack_mode dynamic_retrieval` |
| **6** | `oracle_injection` | Oracle Attack (Oracle-Hard) - Direct injection of poisoned context | `--attack_mode oracle_injection` |
| **7** | `surrogate` | Surrogate Model Attack (Our Method) | `--attack_mode surrogate` (Coming Soon) |

---

## 1. ReAct Agent
**Path**: `models/react/attack_react.py`

### Common Arguments
- `--model_path`: Path to the main LLM (e.g., Llama-3).
- `--retriever_model`: Path to E5 retriever.
- `--index_dir`: Path to E5 index (clean).
- `--data_path`: Path to evaluation dataset (JSON).

### Experiment Commands

#### Exp 1: Base (Clean)
```bash
python models/react/attack_react.py \
    --attack_mode base \
    --model_path /path/to/llama3 \
    --index_dir /path/to/datasets/hotpotqa/e5_index \
    --data_path /path/to/datasets/hotpotqa/hotpotqa.json
```

#### Exp 2-4: Static Poisoning
Requires a pre-generated poisoned corpus file (jsonl).
```bash
python models/react/attack_react.py \
    --attack_mode static_main_main \
    --poisoned_corpus_path /path/to/datasets/hotpotqa/poisoned_main_main.jsonl \
    --model_path /path/to/llama3 \
    --index_dir /path/to/datasets/hotpotqa/e5_index \
    --data_path /path/to/datasets/hotpotqa/hotpotqa.json
```
*(Replace `static_main_main` with `static_main_sub` or `static_sub_sub` as needed)*

#### Exp 5: Dynamic Retrieval (Oracle-Soft)
Generates poisoned documents on-the-fly and adds them to the retriever.
```bash
python models/react/attack_react.py \
    --attack_mode dynamic_retrieval \
    --model_path /path/to/llama3 \
    --index_dir /path/to/datasets/hotpotqa/e5_index \
    --data_path /path/to/datasets/hotpotqa/hotpotqa.json
```
*Note: This will initialize `AttackManager` which loads the Attacker LLM (e.g., Qwen).*

#### Exp 6: Oracle Injection (Oracle-Hard)
Bypasses retriever and directly injects poisoned documents into the agent's observation.
```bash
python models/react/attack_react.py \
    --attack_mode oracle_injection \
    --model_path /path/to/llama3 \
    --index_dir /path/to/datasets/hotpotqa/e5_index \
    --data_path /path/to/datasets/hotpotqa/hotpotqa.json
```

---

## 2. WebThinker Agent
**Path**: `models/webthinker/scripts/run_web_thinker_for_attack.py`
**Note**: WebThinker is fully async and uses `AsyncOpenAI` client.

### Experiment Commands

#### Exp 1: Base (Clean)
```bash
python models/webthinker/scripts/run_web_thinker_for_attack.py \
    --attack_mode base \
    --api_base_url http://localhost:8000/v1 \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --index_dir /path/to/datasets/hotpotqa/e5_index \
    --data_path /path/to/datasets/hotpotqa/hotpotqa.json
```

#### Exp 2-4: Static Poisoning
```bash
python models/webthinker/scripts/run_web_thinker_for_attack.py \
    --attack_mode static_sub_sub \
    --poisoned_corpus_path /path/to/datasets/hotpotqa/poisoned_sub_sub.jsonl \
    --api_base_url http://localhost:8000/v1 \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --data_path /path/to/datasets/hotpotqa/hotpotqa.json
```

#### Exp 5-6: Dynamic & Oracle
WebThinker reuses the `aux_model` (or a specified attacker endpoint) for generating attacks.
```bash
python models/webthinker/scripts/run_web_thinker_for_attack.py \
    --attack_mode dynamic_retrieval \
    --api_base_url http://localhost:8000/v1 \
    --model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
    --aux_model_name "Qwen/Qwen2.5-32B-Instruct" \
    --index_dir /path/to/datasets/hotpotqa/e5_index \
    --data_path /path/to/datasets/hotpotqa/hotpotqa.json
```
*(Use `--attack_mode oracle_injection` for Exp 6)*

---

## 3. CoRag Agent
**Path**: `models/corag/src/inference/attack_corag.py`

### Experiment Commands

#### Exp 1: Base (Clean)
```bash
python models/corag/src/inference/attack_corag.py \
    --attack_mode base \
    --retriever_model /path/to/e5 \
    --index_dir /path/to/datasets/hotpotqa/e5_index
    # Note: Corag assumes dataset path internally or via config override in this version
```

#### Exp 2-4: Static Poisoning
```bash
python models/corag/src/inference/attack_corag.py \
    --attack_mode static_main_main \
    --poisoned_corpus_path /path/to/datasets/hotpotqa/poisoned_main_main.jsonl \
    --retriever_model /path/to/e5
```

#### Exp 5: Dynamic Retrieval
```bash
python models/corag/src/inference/attack_corag.py \
    --attack_mode dynamic_retrieval \
    --retriever_model /path/to/e5
```

#### Exp 6: Oracle Injection
```bash
python models/corag/src/inference/attack_corag.py \
    --attack_mode oracle_injection \
    --retriever_model /path/to/e5
```
