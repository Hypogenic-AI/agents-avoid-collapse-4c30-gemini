import json
import urllib.request
import os
import time

papers = [
    {
        "id": "2305.17493",
        "title": "AI models collapse when trained on recursively generated data",
        "filename": "2305_17493_model_collapse.pdf"
    },
    {
        "id": "2402.04259",
        "title": "Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents",
        "filename": "2402_04259_cooperate_collapse.pdf"
    },
    {
        "id": "2407.08480",
        "title": "Epistemic diversity across language models mitigates knowledge collapse",
        "filename": "2407_08480_epistemic_diversity.pdf"
    },
    {
        "id": "2402.11894",
        "title": "Data Augmentation using Large Language Models",
        "filename": "2402_11894_data_augmentation.pdf"
    },
    {
        "id": "2408.06456",
        "title": "Model Collapse Does Not Mean What You Think",
        "filename": "2408_06456_model_collapse_revisited.pdf"
    }
]

os.makedirs("papers", exist_ok=True)

for p in papers:
    url = f"https://arxiv.org/pdf/{p['id']}.pdf"
    filepath = os.path.join("papers", p["filename"])
    print(f"Downloading {p['title']} from {url}")
    try:
        urllib.request.urlretrieve(url, filepath)
        time.sleep(2)  # be nice to arxiv
    except Exception as e:
        print(f"Failed to download {p['id']}: {e}")
