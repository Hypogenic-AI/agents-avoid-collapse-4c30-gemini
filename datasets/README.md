# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: HuggingFaceTB/cosmopedia (Sample)

### Overview
- **Source**: HuggingFaceTB/cosmopedia
- **Size**: Varies (Very large full dataset, we download a small sample for demonstration)
- **Format**: HuggingFace Dataset
- **Task**: Language Modeling, Synthetic Data Generation
- **Splits**: train
- **License**: Apache 2.0

### Download Instructions

**Using HuggingFace (recommended):**
```python
from datasets import load_dataset
# Using streaming to only get a small portion or just downloading a subset
dataset = load_dataset("HuggingFaceTB/cosmopedia", "auto_math_text", split="train[:1%]")
dataset.save_to_disk("datasets/cosmopedia_sample")
```

### Loading the Dataset

Once downloaded, load with:
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/cosmopedia_sample")
```

### Sample Data
See `datasets/samples.json` for a few rows of sample data.

### Notes
- We use synthetic datasets like Cosmopedia to experiment with LLM populations training on each other's (or external) synthetic data.
