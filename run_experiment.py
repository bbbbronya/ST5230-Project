"""
Main experiment runner.

For each (model, dataset) combination:
  1. Load the pre-sampled dataset from data/.
  2. For every sample, run inference under all 4 prompt variants.
  3. Compute per-sample metrics (USS, Flip, Logit Margin Variance, …).
  4. Save incremental results to results/<model_tag>/<dataset>.jsonl
     (one JSON line per sample, so the run can be resumed after interruption).
  5. After all samples, append an aggregate summary to
     results/<model_tag>/<dataset>_summary.json.

Usage
-----
# Run Llama-3 8B on SciQ (open-source path):
python run_experiment.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset sciq \
    --load_in_8bit

# Run GPT-4o on TruthfulQA (API):
python run_experiment.py \
    --model gpt-4o \
    --dataset truthfulqa

# Resume from a checkpoint (skip already-processed samples):
python run_experiment.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset sciq \
    --resume
"""

import argparse
import json
import os
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Allow importing from sources/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sources"))

from data_utils import (
    load_dataset_from_disk,
    load_sciq,
    load_truthfulqa,
    save_dataset,
)
from model_inference import load_model
from prompt_variants import NUM_VARIANTS, build_prompt
from uncertainty_metrics import aggregate_metrics, compute_sample_metrics

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "outputs", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "results")

DATASET_LOADERS = {
    "sciq": (load_sciq, os.path.join(DATA_DIR, "sciq_sample.json")),
    "truthfulqa": (load_truthfulqa, os.path.join(DATA_DIR, "truthfulqa_sample.json")),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_tag(model_name: str) -> str:
    """Filesystem-safe tag derived from model name."""
    return model_name.replace("/", "_").replace(":", "-")


def _load_or_prepare_dataset(dataset_name: str) -> list:
    loader_fn, cache_path = DATASET_LOADERS[dataset_name]
    if os.path.exists(cache_path):
        print(f"  Loading cached dataset from {cache_path}")
        return load_dataset_from_disk(cache_path)
    print(f"  Downloading and sampling {dataset_name} …")
    data = loader_fn()
    save_dataset(data, cache_path)
    return data


def _load_checkpoint(out_path: str) -> set:
    """Return set of already-processed sample indices (for resume)."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        done.add(rec["sample_idx"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    return done


def _append_record(out_path: str, record: dict):
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core experiment loop
# ---------------------------------------------------------------------------

def run(
    model_name: str,
    dataset_name: str,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    api_key: str | None = None,
    openrouter: bool = False,
    resume: bool = True,
    max_samples: int | None = None,
):
    tag = _model_tag(model_name)
    out_dir = os.path.join(RESULTS_DIR, tag)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{dataset_name}.jsonl")
    summary_path = os.path.join(out_dir, f"{dataset_name}_summary.json")

    # --- Dataset ---
    print(f"\n[1/3] Loading dataset: {dataset_name}")
    data = _load_or_prepare_dataset(dataset_name)
    if max_samples:
        data = data[:max_samples]
    print(f"  {len(data)} samples ready.")

    # --- Resume checkpoint ---
    done_indices: set = set()
    if resume:
        done_indices = _load_checkpoint(out_path)
        if done_indices:
            print(f"  Resuming: {len(done_indices)} samples already processed.")

    # --- Model ---
    print(f"\n[2/3] Loading model: {model_name}")
    model_kwargs: dict = {}
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    if api_key:
        model_kwargs["api_key"] = api_key
    if openrouter:
        model_kwargs["base_url"] = "https://openrouter.ai/api/v1"

    model = load_model(model_name, **model_kwargs)
    print("  Model loaded.")

    # --- Inference ---
    print(f"\n[3/3] Running inference ({NUM_VARIANTS} prompt variants per sample) …\n")
    all_metrics = []

    for sample_idx, item in enumerate(data):
        if sample_idx in done_indices:
            continue

        question = item["question"]
        options = item["options"]
        correct_letter = item["correct_letter"]
        num_options = len(options)

        variant_results = []
        for variant in range(NUM_VARIANTS):
            prompt_text = build_prompt(dataset_name, variant, question, options)
            try:
                result = model.infer(prompt_text, num_options=num_options)
            except Exception as exc:
                print(f"  [sample {sample_idx}, variant {variant}] ERROR: {exc}")
                result_dict = {
                    "variant": variant,
                    "predicted_letter": "ERROR",
                    "entropy": float("nan"),
                    "logit_margin": float("nan"),
                    "answer_logits": {},
                    "error": str(exc),
                }
            else:
                result_dict = {
                    "variant": variant,
                    "predicted_letter": result.predicted_letter,
                    "entropy": result.entropy,
                    "logit_margin": result.logit_margin,
                    "answer_logits": result.answer_logits,
                }
            variant_results.append(result_dict)

        # Filter out error variants before computing metrics
        valid_results = [r for r in variant_results if r["predicted_letter"] != "ERROR"]
        metrics = compute_sample_metrics(valid_results) if valid_results else {}

        record = {
            "sample_idx": sample_idx,
            "question": question,
            "options": options,
            "correct_letter": correct_letter,
            "variant_results": variant_results,
            "metrics": metrics,
        }

        _append_record(out_path, record)
        all_metrics.append(metrics)

        # Progress log
        correct_preds = sum(
            1 for r in valid_results if r["predicted_letter"] == correct_letter
        )
        flip = metrics.get("flip", False)
        uss = metrics.get("uss", float("nan"))
        print(
            f"  [{sample_idx + 1:>4}/{len(data)}] "
            f"correct_preds={correct_preds}/{len(valid_results)}  "
            f"flip={flip}  uss={uss:.3f}"
        )

    # --- Aggregate summary ---
    if all_metrics:
        summary = aggregate_metrics(all_metrics)
        summary["model"] = model_name
        summary["dataset"] = dataset_name
        summary["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n=== Summary ===")
        print(f"  Prediction flip rate : {summary['prediction_flip_rate']:.3f}")
        print(f"  Mean USS             : {summary['uss_mean']:.3f} ± {summary['uss_std']:.3f}")
        print(f"  Mean entropy         : {summary['mean_entropy_mean']:.3f}")
        print(f"  Saved → {summary_path}")

    print(f"\nResults saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prompt sensitivity of LLM uncertainty estimates"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Model name or path. "
            "e.g. meta-llama/Meta-Llama-3-8B-Instruct  |  gpt-4o  |  gpt-5"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_LOADERS.keys()),
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load Llama model in 8-bit (requires bitsandbytes).",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load Llama model in 4-bit (requires bitsandbytes).",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (overrides OPENAI_API_KEY env var).",
    )
    parser.add_argument(
        "--openrouter",
        action="store_true",
        help="Use OpenRouter API (reads OPENROUTER_API_KEY from .env or env var).",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Start fresh, ignoring any existing checkpoint.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit to first N samples (for quick testing).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        model_name=args.model,
        dataset_name=args.dataset,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        api_key=args.api_key,
        openrouter=args.openrouter,
        resume=not args.no_resume,
        max_samples=args.max_samples,
    )
