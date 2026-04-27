# LLM MVP Research: How Many Agents to Avoid Collapse?

This repository contains the code and results for an investigation into the "Minimum Viable Population" (MVP) for LLM ecosystems.

## Key Findings
- **More isn't always better**: A population of 3 identical LLMs collapsed **faster** than a single LLM loop when sharing a common data pool.
- **Diversity is the MVP**: Stability requires "epistemic diversity" (different sampling, architectures, or priors). Simply increasing the number of agents (N) accelerates collapse by amplifying shared biases.
- **Visual Collapse**: Models rapidly degenerate into repeating single sentences, a phenomenon amplified by uncoordinated multi-agent learning.

## Quick Start
1. **Setup**:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
2. **Prepare Data**:
   ```bash
   python src/prepare_data.py
   ```
3. **Run Experiments**:
   ```bash
   python src/simulation.py --name baseline_n1 --pop_size 1 --gens 3
   python src/simulation.py --name pop_n3_homo --pop_size 3 --gens 3
   ```
4. **Analyze**:
   ```bash
   python src/analysis.py
   python src/sentence_analysis.py
   ```

## Repository Structure
- `src/`: Simulation and analysis scripts.
- `results/`: Raw data pools and summary JSON files.
- `figures/`: Visualizations of perplexity trends.
- `REPORT.md`: Comprehensive research report.

## Reproducibility
All experiments were conducted on NVIDIA A6000 GPUs using `distilgpt2` and LoRA (PEFT). Random seeds were fixed for reproducibility.
