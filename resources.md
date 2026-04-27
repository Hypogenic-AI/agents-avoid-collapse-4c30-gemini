## Resources Catalog

### Summary
This document catalogs all resources gathered for the research project investigating the minimum viable population of LLM ecosystems to prevent model/information collapse. It includes key foundational papers, a cloned repository for simulating dataset contamination, and a sample dataset of synthetic data.

### Papers
Total papers downloaded: 5

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| AI models collapse when trained on recursively generated data | Shumailov et al. | 2024 | papers/2305_17493_model_collapse.pdf | Defines model collapse mathematically and empirically. |
| Cooperate or Collapse: Emergence of Sustainable Cooperation... | Piatti et al. | 2024 | papers/2402_04259_cooperate_collapse.pdf | Studies multi-agent collapse via resource management. |
| Epistemic diversity across language models mitigates knowledge collapse | Hodel & West | 2025 | papers/2407_08480_epistemic_diversity.pdf | Core to the hypothesis: proves diversity prevents collapse. |
| Data Augmentation using Large Language Models | Ding et al. | 2024 | papers/2402_11894_data_augmentation.pdf | Reviews LLM-generated synthetic data pipelines. |
| Model Collapse Does Not Mean What You Think | Schaeffer et al. | 2025 | papers/2408_06456_model_collapse_revisited.pdf | Argues collapse is avoidable in realistic mixed-data scenarios. |

See `papers/README.md` for detailed descriptions.

### Datasets
Total datasets downloaded: 1

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| Cosmopedia (Sample) | HuggingFace | 10 samples (JSON), HF format | Synthetic Text Generation | datasets/cosmopedia_sample/ | Example of large-scale synthetic text for LLM training. |

See `datasets/README.md` for detailed descriptions.

### Code Repositories
Total repositories cloned: 1

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| dataset-contamination | github.com/moskomule/dataset-contamination | Simulating model collapse | code/dataset-contamination/ | Provides codebase to evaluate generative model recursive training. |

See `code/README.md` for detailed descriptions.

### Resource Gathering Notes

#### Search Strategy
Searched Semantic Scholar and arXiv using the `find_papers.py` script with keywords related to "LLM model collapse," "epistemic diversity," "synthetic data," and "cooperate or collapse".

#### Selection Criteria
Selected papers that directly addressed the mathematical theory of model collapse, the multi-agent cooperative collapse, and specifically the role of population diversity in mitigating these collapses.

#### Challenges Encountered
Finding the exact boundary of what constitutes an "individual" in a population of LLMs required searching for papers on "epistemic diversity" rather than purely biological population terms.

#### Gaps and Workarounds
Simulating an entire LLM pre-training run is computationally prohibitive. Workarounds include using smaller LLMs or focusing on fine-tuning / low-rank adaptations over multiple generations.

### Recommendations for Experiment Design

1. **Primary dataset(s)**: Cosmopedia for simulating text ecosystems, or a smaller QA dataset to measure factual collapse.
2. **Baseline methods**: Single-agent recursive self-training (where N=1). This should demonstrate a rapid baseline collapse.
3. **Evaluation metrics**: Lexical diversity (Self-BLEU), Vendi Score, and perplexity of the generated distribution relative to the original human distribution.
4. **Code to adapt/reuse**: The `dataset-contamination` repo can be adapted to loop over text generation models instead of image models, or serve as a structural reference for building the recursive training pipeline.
