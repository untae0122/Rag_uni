import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.attacks.attack_manager import AttackManager, AttackMode
# Need to import config to setup AttackManager? 
# AttackManager needs adv_generator (LLM) or we pass one.

def generate_surrogate_corpus(
    model_path: str, 
    data_path: str, 
    output_corpus_path: str,
    attacker_model_path: str
):
    """
    1. Load Surrogate Model.
    2. Predict subqueries for each question in data_path.
    3. Use AttackManager to generate poisoned documents for these subqueries.
    4. Save to output_corpus_path.
    """
    
    # 1. Load Surrogate
    print(f"Loading Surrogate Model: {model_path}")
    surrogate_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    surrogate_tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 2. Load Evaluation Data
    print(f"Loading questions from: {data_path}")
    with open(data_path, 'r') as f:
        questions = json.load(f) # List of dicts
        
    # 3. Load Attacker (for generating poison text)
    # We can reuse AttackManager logic, but AttackManager is designed for single-step injection usually.
    # However, generate_adversarial_content(query, target_response) returns content.
    # We need to initialize AttackManager with a generator.
    
    print(f"Initializing AttackManager with attacker model: {attacker_model_path}")
    # Note: To avoid loading two LLMs on one GPU, we might need to be careful.
    # If surrogate and attacker fit, great. If not, offload or sequential.
    
    from vllm import LLM, SamplingParams
    attacker_llm = LLM(model=attacker_model_path, tensor_parallel_size=1, gpu_memory_utilization=0.4)
    attacker_tokenizer_ = attacker_llm.get_tokenizer()
    
    attack_manager = AttackManager(
        adv_generator=attacker_llm,
        adv_tokenizer=attacker_tokenizer_,
        adv_sampling_params=SamplingParams(temperature=0.7, max_tokens=512)
    )
    
    poisoned_corpus_items = []
    
    print("Generating predictions and poison...")
    for item in tqdm(questions):
        qid = item.get('id', '')
        query = item.get('question', '')
        target_answer = item.get('incorrect_answer', '')
        
        # Predict Subquery
        # Construct prompt: "Question: {q}\n" -> Predict "Thought: ... Action: Search[...]"
        # This assumes the surrogate learned to start generating immediately.
        prompt = f"Question: {query}\n"
        inputs = surrogate_tokenizer(prompt, return_tensors="pt").to(surrogate_model.device)
        
        # Generate until "]" or newline?
        # We want to capture "Thought: ... Action: Search[subquery]"
        # Let's generate max 128 tokens.
        outputs = surrogate_model.generate(
            **inputs, 
            max_new_tokens=128, 
            do_sample=False, # Deterministic prediction
            stop_strings=["]", "\nObservation"], # Stop after action
            tokenizer=surrogate_tokenizer
        )
        output_text = surrogate_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse Subquery
        # Look for "Search[...]" in the generated text (after prompt)
        generated_part = output_text[len(prompt):]
        subquery = None
        if "Search[" in generated_part:
            try:
                subquery = generated_part.split("Search[")[1].split("]")[0]
            except:
                pass
        
        # Fallback: if no search, maybe poison the main query? 
        # Experiment 7 usually targets the predicted subquery.
        target_query = subquery if subquery else query
        
        # Generate Poisoned Document
        # We want to make the retriever retrieve this doc for `target_query`, 
        # but the content should support `target_answer` for `query`.
        # AttackManager.generate_adversarial_content generates text that supports target_answer.
        
        # We need to craft the "title" and "body".
        # AttackManager.generate_adversarial_content returns (title, content).
        
        adv_title, adv_content = attack_manager.generate_adversarial_content(
            query=query, # The main question the *Agent* is trying to answer (context for poison)
                         # Wait, generate_adversarial_content might need the *target* context?
                         # Usually we want the doc to say "The answer to {query} is {target_answer}".
            target_response=target_answer
        )
        
        # Create Corpus Item
        # ID: poison_{qid}
        # Text: We want this to be retrieved by `target_query`.
        # So we might need to optimize it?
        # AttackManager has 'poison_retriever' logic which adds it to index.
        # But here we are creating a STATIC corpus file.
        # So we just save the item.
        
        # IMPORTANT: To make it retrieved by `target_query`, the doc should contain `target_query` keywords?
        # Current AttackManager generation is basic. 
        # Ideally we append `target_query` to valid corpus item or generate relevant text.
        # For now, we use the generated content.
        
        poisoned_item = {
            "_id": f"poison_{qid}",
            "title": adv_title,
            "text": adv_content,
            "metadata": {
                "original_id": qid,
                "target_answer": target_answer,
                "is_poisoned": True,
                "targeted_query": target_query # Predicted subquery
            }
        }
        poisoned_corpus_items.append(poisoned_item)

    # Save Corpus
    os.makedirs(os.path.dirname(output_corpus_path), exist_ok=True)
    with open(output_corpus_path, 'w', encoding='utf-8') as f:
        for item in poisoned_corpus_items:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
            
    print(f"Generated {len(poisoned_corpus_items)} poisoned documents to {output_corpus_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to trained surrogate model")
    parser.add_argument("--data_path", default="../datasets/hotpotqa/hotpotqa.json", help="Evaluation dataset")
    parser.add_argument("--output_corpus_path", default="../datasets/hotpotqa/poisoned_surrogate.jsonl", help="Output corpus path")
    parser.add_argument("--attacker_model_path", default="/home/work/Redteaming/data1/REDTEAMING_LLM/cache/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe", help="Path to attacker LLM")
    
    args = parser.parse_args()
    
    generate_surrogate_corpus(
        args.model_path,
        args.data_path,
        args.output_corpus_path,
        args.attacker_model_path
    )
