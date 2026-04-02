# ST5230 Project: Prompt Sensitivity of Uncertainty Estimates in LLMs

## Research Question
How sensitive are LLM uncertainty estimates to semantically equivalent prompt perturbations?

## Models
| Role | Model | Params | Source |
|------|-------|--------|--------|
| Small open-source | `mistralai/ministral-3b-2512` | 3B | OpenRouter |
| Medium open-source | `meta-llama/llama-3.1-8b-instruct` | 8B | OpenRouter |
| Large proprietary | `openai/gpt-4o` | undisclosed | OpenRouter |

See `docs/model_changes.txt` for why these models were chosen over the originally proposed ones.

## Datasets
- **SciQ** — 250 samples (science multiple-choice)
- **TruthfulQA** — 250 samples (common misconception multiple-choice)

## Prompt Variants
Each sample is evaluated under 4 semantically equivalent prompt formats:
- V0: Baseline (instruction-first, `A)` format)
- V1: Reworded (different phrasing, `A.` format)
- V2: Question-first (`(A)` format)
- V3: Minimal (no instruction, `A:` format)

## Key Metrics
- **Prediction Flip Rate** — how often the predicted answer changes across prompts
- **Logit Margin Variance** — variance of (top-1 minus top-2) logit across prompts
- **Hidden Instability** — samples where prediction is stable but confidence fluctuates

## Quick Start
```bash
pip install -r requirements.txt

# Prepare data (once)
python sources/data_utils.py

# Run experiments (example)
python run_experiment.py --model meta-llama/llama-3.1-8b-instruct --dataset sciq --openrouter

# Analyze results
python analyze_results.py
```

See `steps.txt` for full reproduction instructions.

## Project Structure
```
├── run_experiment.py           # Main experiment runner
├── analyze_results.py          # Analysis + 5 figures
├── sources/
│   ├── data_utils.py           # Dataset download & sampling
│   ├── model_inference.py      # API inference (OpenRouter)
│   ├── prompt_variants.py      # 4 prompt variant definitions
│   └── uncertainty_metrics.py  # USS, Flip Rate, LM Variance
├── outputs/
│   ├── data/                   # Sampled datasets
│   ├── results/                # Per-model inference results
│   └── figures/                # Generated PDF figures
├── docs/
│   ├── Group3_Project_Proposal.pdf
│   └── model_changes.txt       # Model selection decision log
└── steps.txt                   # Reproduction guide (中文)
```
