import os
import sys
import json
import torch
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    HfArgumentParser
)
from datasets import load_dataset

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="meta-llama/Meta-Llama-3-8B-Instruct")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to training data jsonl"})

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(f"Loading model: {model_args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )

    # Load Dataset
    print(f"Loading data from: {data_args.data_path}")
    dataset = load_dataset("json", data_files=data_args.data_path, split="train")
    
    def tokenize_function(examples):
        # Format: Prompt + Completion
        # We act as a causal LM.
        # Ideally we only train on completion loss, but simple CLM on full text is often enough for this scale.
        
        texts = [p + c for p, c in zip(examples['prompt'], examples['completion'])]
        return tokenizer(texts, padding="max_length", truncation=True, max_length=1024)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
