import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def load_results(name):
    with open(f"results/{name}_summary.json", "r") as f:
        return json.load(f)

def get_unique_ngrams_ratio(data_path, n=3):
    texts = []
    with open(data_path, "r") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    
    all_ngrams = []
    for text in texts:
        words = text.split()
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
        all_ngrams.extend(ngrams)
    
    if not all_ngrams:
        return 0
    
    unique_ngrams = len(set(all_ngrams))
    return unique_ngrams / len(all_ngrams)

def analyze():
    names = ["baseline_n1", "pop_n3_homo", "pop_n3_hetero"]
    labels = ["N=1 Homogeneous", "N=3 Homogeneous", "N=3 Heterogeneous"]
    
    plt.figure(figsize=(10, 6))
    
    for name, label in zip(names, labels):
        results = load_results(name)
        gens = [r["generation"] for r in results]
        ppls = [r["avg_perplexity"] for r in results]
        plt.plot(gens, ppls, marker='o', label=label)
        
        print(f"\n--- {label} ---")
        for g, p in zip(gens, ppls):
            print(f"Gen {g}: PPL = {p:.2f}")
            
        # Diversity of last generation pool
        last_gen = gens[-2] # The last generation that generated data
        pool_path = f"results/{name}/gen_{last_gen}_pool.jsonl"
        diversity = get_unique_ngrams_ratio(pool_path)
        print(f"Final Pool Diversity (3-gram ratio): {diversity:.4f}")

    plt.xlabel("Generation")
    plt.ylabel("Average Perplexity (on Human Test Set)")
    plt.title("LLM Ecosystem Collapse: Perplexity over Generations")
    plt.legend()
    plt.grid(True)
    plt.savefig("figures/perplexity_trend.png")
    print("\nSaved figure to figures/perplexity_trend.png")

if __name__ == "__main__":
    analyze()
