import os
import torch
import json
import random
import numpy as np
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class LLMEcosystem:
    def __init__(self, name, pop_size, diversity_type, base_model_name="distilgpt2"):
        self.name = name
        self.pop_size = pop_size
        self.diversity_type = diversity_type
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initial models
        self.models = []
        for i in range(pop_size):
            model = AutoModelForCausalLM.from_pretrained(base_model_name)
            self.models.append(model)
            
        self.generation = 0
        self.current_data_path = "datasets/wikitext_initial/train.jsonl"
        self.test_data_path = "datasets/wikitext_initial/test.jsonl"
        
        self.results = []

    def evaluate(self, model, gen):
        model.to("cuda")
        test_dataset = load_dataset("json", data_files=self.test_data_path, split="train")
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, max_length=512)
        
        tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        
        trainer = Trainer(
            model=model,
            eval_dataset=tokenized_test,
            data_collator=data_collator
        )
        
        eval_results = trainer.evaluate()
        perplexity = np.exp(eval_results["eval_loss"])
        model.to("cpu")
        return perplexity

    def generate_data(self, model, num_samples=100, model_idx=0):
        model.to("cuda")
        model.eval()
        generated_texts = []
        
        # Use some prompts from the initial data to start generation
        initial_data = load_dataset("json", data_files="datasets/wikitext_initial/train.jsonl", split="train")
        prompts = initial_data.select(range(min(num_samples, len(initial_data))))["text"]
        # Take first 20 tokens as prompt
        prompts = [" ".join(p.split()[:20]) for p in prompts]
        
        for i in tqdm(range(num_samples), desc=f"Gen {self.generation} Model {model_idx} generating"):
            prompt = prompts[i % len(prompts)]
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            
            # Diversity strategies
            temp = 0.7
            if self.diversity_type == "heterogeneous":
                # Give each "individual" a slightly different personality/style via temp
                temp = 0.5 + (model_idx * 0.2) 
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=100, 
                    do_sample=True, 
                    temperature=temp,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_texts.append({"text": text})
            
        model.to("cpu")
        return generated_texts

    def train_next_gen(self, pooled_data_path):
        new_models = []
        
        dataset = load_dataset("json", data_files=pooled_data_path, split="train")
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], truncation=True, max_length=512)
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )

        for i in range(self.pop_size):
            print(f"Training Model {i} for Generation {self.generation + 1}")
            # Each model in the next gen starts from the previous gen's models?
            # Or they all start from base? "birthing new LLMs" usually means training on data.
            # If they are descendants, they should probably start from the parent model.
            # For N=1, it's clear. For N>1, does each model have a parent?
            # Let's say Model i of Gen T+1 is trained on the pool, starting from Model i of Gen T.
            
            base_model = self.models[i]
            model = get_peft_model(base_model, lora_config)
            
            training_args = TrainingArguments(
                output_dir=f"results/{self.name}/gen_{self.generation+1}_model_{i}",
                per_device_train_batch_size=4,
                num_train_epochs=3,
                logging_steps=10,
                save_strategy="no",
                report_to="none"
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator
            )
            
            trainer.train()
            final_model = model.merge_and_unload()
            new_models.append(final_model)
            
        self.models = new_models
        self.generation += 1

    def run_generation(self):
        print(f"\n--- Generation {self.generation} ---")
        
        # 1. Evaluate current models
        perplexities = []
        for i, model in enumerate(self.models):
            ppl = self.evaluate(model, self.generation)
            perplexities.append(ppl)
            print(f"Model {i} Perplexity: {ppl:.2f}")
        
        avg_ppl = np.mean(perplexities)
        self.results.append({
            "generation": self.generation,
            "avg_perplexity": avg_ppl,
            "all_perplexities": perplexities
        })

        # 2. Generate data
        pool = []
        for i, model in enumerate(self.models):
            gen_data = self.generate_data(model, num_samples=200, model_idx=i)
            pool.extend(gen_data)
        
        # 3. Save pooled data
        pool_path = f"results/{self.name}/gen_{self.generation}_pool.jsonl"
        os.makedirs(os.path.dirname(pool_path), exist_ok=True)
        with open(pool_path, "w") as f:
            for item in pool:
                f.write(json.dumps(item) + "\n")
        
        # 4. Train next generation
        self.train_next_gen(pool_path)

    def save_results(self):
        with open(f"results/{self.name}_summary.json", "w") as f:
            json.dump(self.results, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--pop_size", type=int, default=1)
    parser.add_argument("--diversity", type=str, default="homogeneous")
    parser.add_argument("--gens", type=int, default=3)
    args = parser.parse_args()

    set_seed(42)
    eco = LLMEcosystem(args.name, args.pop_size, args.diversity)
    for _ in range(args.gens):
        eco.run_generation()
    
    # Final evaluation
    perplexities = []
    for i, model in enumerate(eco.models):
        ppl = eco.evaluate(model, eco.generation)
        perplexities.append(ppl)
    avg_ppl = np.mean(perplexities)
    eco.results.append({
        "generation": eco.generation,
        "avg_perplexity": avg_ppl,
        "all_perplexities": perplexities
    })
    
    eco.save_results()
