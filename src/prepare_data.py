import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

def prepare_data():
    print("Loading wikitext dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # Filter out empty lines
    dataset = dataset.filter(lambda x: len(x["text"]) > 100)
    # Take a small subset for faster experimentation
    subset = dataset.select(range(1000))
    os.makedirs("datasets/wikitext_initial", exist_ok=True)
    subset.to_json("datasets/wikitext_initial/train.jsonl")
    print(f"Saved 1000 samples to datasets/wikitext_initial/train.jsonl")

    # Also prepare a test set
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_dataset = test_dataset.filter(lambda x: len(x["text"]) > 100)
    test_subset = test_dataset.select(range(200))
    test_subset.to_json("datasets/wikitext_initial/test.jsonl")
    print(f"Saved 200 samples to datasets/wikitext_initial/test.jsonl")

if __name__ == "__main__":
    prepare_data()
