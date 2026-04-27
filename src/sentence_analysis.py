import json

def get_unique_sentences_ratio(data_path):
    sentences = []
    with open(data_path, "r") as f:
        for line in f:
            text = json.loads(line)["text"]
            # Split by common sentence delimiters
            sents = [s.strip() for s in text.replace("\n", ".").split(".") if len(s.strip()) > 5]
            sentences.extend(sents)
    
    if not sentences:
        return 0
    
    unique_sents = len(set(sentences))
    return unique_sents / len(sentences)

def analyze_sentences():
    names = ["baseline_n1", "pop_n3_homo", "pop_n3_hetero"]
    for name in names:
        pool_path = f"results/{name}/gen_2_pool.jsonl"
        ratio = get_unique_sentences_ratio(pool_path)
        print(f"{name}: Unique Sentence Ratio = {ratio:.4f}")

if __name__ == "__main__":
    analyze_sentences()
