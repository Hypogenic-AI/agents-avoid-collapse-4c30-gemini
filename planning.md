## Motivation & Novelty Assessment

### Why This Research Matters
As LLM-generated content increasingly populates the internet, future LLMs will inevitably be trained on AI-generated data. Understanding the conditions under which this leads to "model collapse" is critical for the long-term sustainability of the AI ecosystem. If we can identify a "Minimum Viable Population" (MVP), we can design better data collection and training strategies to maintain model quality and diversity.

### Gap in Existing Work
Most research on model collapse (e.g., Shumailov et al., 2024) focuses on single-model recursive loops. While recent work (Hodel & West, 2025) discusses diversity, there is limited empirical work defining the *quantitative* MVP threshold or the exact nature of "individuality" required to sustain the population.

### Our Novel Contribution
We will empirically test the "Minimum Viable Population" hypothesis by simulating ecosystems of varying sizes and diversity levels. We will specifically investigate what counts as "different enough" (e.g., different temperature, different seeds, or different architectures) to count as distinct individuals that contribute to ecosystem stability.

### Experiment Justification
- **Experiment 1 (Baseline)**: N=1 recursive training. This establishes the "extinction" or collapse rate for a single individual.
- **Experiment 2 (Population Size)**: Vary N (3, 5) with homogeneous models. Tests if numbers alone provide enough stochastic diversity to slow collapse.
- **Experiment 3 (Epistemic Diversity)**: Vary N with heterogeneous models (different architectures/priors). Tests if structural diversity is the key to MVP.

## Research Question
Is there a minimum viable population for an LLM information ecosystem such that it does not collapse? Will such an ecosystem always collapse? What counts as different enough to be an individual in the population?

## Hypothesis Decomposition
1.  **H1 (Population Size)**: Larger population sizes (N > 1) with shared data pools significantly delay model collapse compared to a single-model loop (N=1).
2.  **H2 (Diversity Requirement)**: A population of identical models (same architecture, same pre-training) will eventually collapse regardless of N, as they will converge to the same biased output.
3.  **H3 (Epistemic Diversity)**: A population with heterogeneous "individuals" (different architectures or training data priors) can reach a stable equilibrium, avoiding total collapse.

## Proposed Methodology

### Approach
We will simulate a recursive training ecosystem using small LLMs. In each generation, models will generate data that is then pooled and used to fine-tune the next generation of models.

### Experimental Steps
1.  **Environment Setup**: Install `transformers`, `datasets`, `peft`, and `torch`.
2.  **Model Selection**: Use `distilgpt2` or `gpt2` as the base model for efficiency.
3.  **Initial Data**: Use a subset of `wikitext-103` (human-written).
4.  **Generations**: Run for 5 generations.
5.  **Configurations**:
    - **Control (N=1)**: One model training on its own output.
    - **Homogeneous Population (N=3, N=5)**: Multiple identical models training on a shared pool of their combined outputs.
    - **Heterogeneous Population (N=3)**: Different models (e.g., `gpt2`, `distilgpt2`, and a custom-weighted version) training on a shared pool.
6.  **Metrics**:
    - **Perplexity**: On a held-out human test set (measures quality).
    - **Vendi Score / Self-BLEU**: Measures diversity of generated text.
    - **Vocabulary Overlap**: Measures shrinkage of the "tail" of the distribution.

### Baselines
- **N=1 Homogeneous**: The classic model collapse scenario.
- **Human Data (Target)**: The initial distribution we aim to preserve.

### Evaluation Metrics
- **Perplexity (PPL)**: Primary measure of model degradation.
- **Semantic Drift**: Using embeddings to measure how far the generated distribution moves from the original.
- **Vocabulary Zipf's Law consistency**: Checking if the model still produces rare words.

### Statistical Analysis Plan
- T-tests between final perplexities of N=1 vs N=5.
- Correlation analysis between Population Diversity (initial) and Rate of Collapse.

## Timeline and Milestones
- **M1: Setup & Baseline (N=1)**: 1.5 hours.
- **M2: Population Experiments (N=3, 5)**: 2.5 hours.
- **M3: Analysis & Visualization**: 1 hour.
- **M4: Final Report**: 1 hour.

## Potential Challenges
- **Compute Time**: Fine-tuning many models can be slow. *Mitigation*: Use PEFT (LoRA) and small models.
- **Data Quality**: Generated data might become nonsensical too quickly. *Mitigation*: Use reasonable sampling parameters (temperature=0.7, top_p=0.9).

## Success Criteria
- Identifying a clear trend (or lack thereof) between population size/diversity and collapse rate.
- Defining a quantitative metric for "individuality" that correlates with stability.
