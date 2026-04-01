"""
Analysis and visualisation of experiment results.

Loads all .jsonl result files from results/ and produces:

1. Per-model, per-dataset aggregate table
   (flip_rate, USS, logit_margin_variance, corr_lmvar_flip, hidden_instability, accuracy)
2. USS distribution box plots (per model/dataset)
3. Prediction flip rate bar chart
4. Logit margin variance vs flip status box plots  [replaces USS-entropy scatter]
5. Per-variant accuracy line plots
6. Logit margin distribution per prompt variant    [replaces entropy-by-variant]
7. Logit margin variance distribution histogram    [new: SciQ vs TruthfulQA overlay]

Note on entropy/USS: entropy is unreliable in this experiment because non-answer
option tokens rarely appear in the top-20 logprobs returned by the API, so they
are assigned a floor value of -20.0. USS (which depends on entropy) is therefore
also unreliable. The primary confidence-instability metric is logit_margin_variance.

Usage
-----
python analyze_results.py
python analyze_results.py --results_dir path/to/results --figures_dir path/to/figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sources"))
from uncertainty_metrics import (
    aggregate_metrics,
    correlation_uss_flip,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "outputs", "figures")

# Threshold for "high confidence instability" used in hidden-instability detection.
# Chosen to be just below the SciQ mean lmvar (5.39); samples above this with
# no prediction flip are flagged as showing hidden instability.
LMVAR_INSTABILITY_THRESHOLD = 5.0


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_all_results(results_dir: str) -> Dict[str, Dict[str, List[dict]]]:
    """
    Returns nested dict: { model_tag : { dataset : [sample_record, …] } }
    """
    all_results: Dict[str, Dict[str, List[dict]]] = {}

    for model_dir in sorted(os.listdir(results_dir)):
        model_path = os.path.join(results_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        all_results[model_dir] = {}
        for fname in sorted(os.listdir(model_path)):
            if not fname.endswith(".jsonl"):
                continue
            dataset = fname.replace(".jsonl", "")
            records = []
            with open(os.path.join(model_path, fname), "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            if records:
                all_results[model_dir][dataset] = records

    return all_results


def records_to_metrics_df(records: List[dict]) -> pd.DataFrame:
    """Flatten per-sample metrics into a DataFrame."""
    rows = []
    for rec in records:
        m = rec.get("metrics", {})
        row = {
            "sample_idx": rec.get("sample_idx"),
            "correct_letter": rec.get("correct_letter"),
            **m,
        }
        # Add per-variant accuracy
        variant_results = rec.get("variant_results", [])
        correct = rec.get("correct_letter", "")
        accs = [
            int(r["predicted_letter"] == correct)
            for r in variant_results
            if r.get("predicted_letter") not in ("ERROR", None)
        ]
        row["mean_accuracy"] = float(np.mean(accs)) if accs else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _corr_lmvar_flip(metrics_list: List[dict]) -> Optional[float]:
    """Point-biserial correlation between logit_margin_variance and flip (0/1)."""
    lmv = np.array([m["logit_margin_variance"] for m in metrics_list])
    flp = np.array([int(m["flip"]) for m in metrics_list])
    if flp.std() == 0 or lmv.std() == 0:
        return None
    corr, _ = pointbiserialr(flp, lmv)
    return float(corr)


def _count_hidden_instability(records: List[dict]) -> int:
    """
    Count samples where prediction is stable (flip=False) but confidence
    fluctuates significantly (logit_margin_variance > LMVAR_INSTABILITY_THRESHOLD).
    """
    count = 0
    for rec in records:
        m = rec.get("metrics", {})
        if not m.get("flip", True) and m.get("logit_margin_variance", 0) > LMVAR_INSTABILITY_THRESHOLD:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(all_results: Dict):
    rows = []
    for model_tag, datasets in all_results.items():
        for dataset, records in datasets.items():
            metrics_list = [r["metrics"] for r in records if r.get("metrics")]
            if not metrics_list:
                continue
            agg = aggregate_metrics(metrics_list)
            df = records_to_metrics_df(records)

            corr_uss = correlation_uss_flip(metrics_list)
            corr_lmv = _corr_lmvar_flip(metrics_list)
            hi_count = _count_hidden_instability(records)

            rows.append(
                {
                    "model": model_tag,
                    "dataset": dataset,
                    "n": agg.get("n_samples"),
                    "flip_rate": f"{agg.get('prediction_flip_rate', 0):.3f}",
                    "uss_mean±std": f"{agg.get('uss_mean', 0):.3f}±{agg.get('uss_std', 0):.3f}",
                    "lm_var_mean": f"{agg.get('logit_margin_variance_mean', 0):.3f}",
                    "mean_lm": f"{agg.get('mean_logit_margin_mean', 0):.3f}",
                    "mean_acc": f"{df['mean_accuracy'].mean():.3f}",
                    "corr_uss_flip": f"{corr_uss:.3f}" if corr_uss is not None else "N/A",
                    "corr_lmvar_flip": f"{corr_lmv:.3f}" if corr_lmv is not None else "N/A",
                    "hidden_instab": hi_count,
                }
            )

    if not rows:
        print("No results found.")
        return

    df = pd.DataFrame(rows)
    print("\n=== Aggregate Results ===")
    print(df.to_string(index=False))
    print(f"\n  hidden_instab: flip=False AND logit_margin_variance > {LMVAR_INSTABILITY_THRESHOLD}")
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _ensure_figures_dir(figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)


def plot_uss_distributions(all_results: Dict, figures_dir: str):
    """Box plot of USS distributions across models and datasets.

    Note: USS is computed from entropy, which is unreliable when non-answer
    tokens fall outside the API top-20 logprobs window. Interpret with caution.
    """
    _ensure_figures_dir(figures_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax_idx, dataset in enumerate(["sciq", "truthfulqa"]):
        ax = axes[ax_idx]
        data_by_model = {}
        for model_tag, datasets in all_results.items():
            if dataset in datasets:
                records = datasets[dataset]
                uss_vals = [
                    r["metrics"]["uss"]
                    for r in records
                    if r.get("metrics") and not np.isnan(r["metrics"].get("uss", float("nan")))
                ]
                if uss_vals:
                    data_by_model[model_tag] = uss_vals

        if not data_by_model:
            ax.set_title(f"{dataset} (no data)")
            continue

        labels = list(data_by_model.keys())
        values = [data_by_model[l] for l in labels]
        ax.boxplot(values, labels=labels, patch_artist=True)
        ax.set_title(f"USS Distribution — {dataset}")
        ax.set_ylabel("Uncertainty Stability Score")
        ax.set_ylim(-2.0, 1.05)
        ax.axhline(y=0.5, color="red", linestyle="--", linewidth=0.8, label="threshold=0.5")
        ax.tick_params(axis="x", rotation=15)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(figures_dir, "uss_distributions.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_flip_rate_bar(all_results: Dict, figures_dir: str):
    """Bar chart of prediction flip rates."""
    _ensure_figures_dir(figures_dir)
    models, datasets_seen = [], set()
    data: Dict[str, Dict[str, float]] = defaultdict(dict)

    for model_tag, datasets in all_results.items():
        models.append(model_tag)
        for dataset, records in datasets.items():
            datasets_seen.add(dataset)
            flips = [r["metrics"].get("flip", False) for r in records if r.get("metrics")]
            data[model_tag][dataset] = float(np.mean(flips)) if flips else 0.0

    datasets_list = sorted(datasets_seen)
    n_models = len(models)
    n_datasets = len(datasets_list)
    x = np.arange(n_models)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, n_models * 2), 5))
    for d_idx, dataset in enumerate(datasets_list):
        vals = [data[m].get(dataset, 0.0) for m in models]
        offset = (d_idx - (n_datasets - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=dataset)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Model")
    ax.set_ylabel("Prediction Flip Rate")
    ax.set_title("Prediction Flip Rate by Model and Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend()
    plt.tight_layout()

    out = os.path.join(figures_dir, "flip_rate_bar.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_lmvar_flip_comparison(all_results: Dict, figures_dir: str):
    """Box plot: logit_margin_variance grouped by flip status (True vs False).

    Shows whether samples with prediction flips have higher confidence instability,
    directly addressing the 'uncertainty variance correlates with flip' hypothesis.
    One figure per (model, dataset) combination.
    """
    _ensure_figures_dir(figures_dir)

    for model_tag, datasets in all_results.items():
        for dataset, records in datasets.items():
            lmv_no_flip, lmv_flip = [], []
            for rec in records:
                m = rec.get("metrics", {})
                lmv = m.get("logit_margin_variance", float("nan"))
                flip = m.get("flip", None)
                if flip is None or np.isnan(lmv):
                    continue
                if flip:
                    lmv_flip.append(lmv)
                else:
                    lmv_no_flip.append(lmv)

            if not lmv_no_flip and not lmv_flip:
                continue

            fig, ax = plt.subplots(figsize=(6, 5))
            data = []
            labels = []
            if lmv_no_flip:
                data.append(lmv_no_flip)
                labels.append(f"No Flip\n(n={len(lmv_no_flip)})")
            if lmv_flip:
                data.append(lmv_flip)
                labels.append(f"Flip\n(n={len(lmv_flip)})")

            bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
            colors = ["steelblue", "tomato"]
            for patch, color in zip(bp["boxes"], colors[: len(data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.axhline(
                y=LMVAR_INSTABILITY_THRESHOLD,
                color="gray", linestyle="--", linewidth=0.8,
                label=f"instability threshold={LMVAR_INSTABILITY_THRESHOLD}",
            )
            ax.set_ylabel("Logit Margin Variance")
            ax.set_title(
                f"Confidence Instability by Flip Status\n{model_tag} / {dataset}"
            )
            ax.legend(fontsize=8)
            plt.tight_layout()

            safe_model = model_tag.replace("/", "_")
            out = os.path.join(figures_dir, f"lmvar_flip_comparison_{safe_model}_{dataset}.pdf")
            plt.savefig(out, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out}")


def plot_per_variant_accuracy(all_results: Dict, figures_dir: str):
    """Line plot of accuracy per prompt variant."""
    _ensure_figures_dir(figures_dir)

    for model_tag, datasets in all_results.items():
        for dataset, records in datasets.items():
            variant_accs: Dict[int, List[int]] = defaultdict(list)
            for rec in records:
                correct = rec.get("correct_letter", "")
                for vr in rec.get("variant_results", []):
                    v = vr.get("variant")
                    pred = vr.get("predicted_letter")
                    if pred not in ("ERROR", None) and v is not None:
                        variant_accs[v].append(int(pred == correct))

            if not variant_accs:
                continue

            variants = sorted(variant_accs.keys())
            means = [np.mean(variant_accs[v]) for v in variants]
            stds = [np.std(variant_accs[v]) / np.sqrt(len(variant_accs[v])) for v in variants]

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.errorbar(variants, means, yerr=stds, marker="o", capsize=4)
            ax.set_xticks(variants)
            ax.set_xticklabels([f"V{v}" for v in variants])
            ax.set_xlabel("Prompt Variant")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Accuracy per Variant — {model_tag} / {dataset}")
            ax.set_ylim(0, 1.0)
            plt.tight_layout()

            safe_model = model_tag.replace("/", "_")
            out = os.path.join(figures_dir, f"variant_acc_{safe_model}_{dataset}.pdf")
            plt.savefig(out, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out}")


def plot_logit_margin_by_variant(all_results: Dict, figures_dir: str):
    """Box plot of logit margin distribution per prompt variant.

    Shows how the model's confidence (top-1 vs top-2 logit gap) changes across
    the four prompt variants, revealing which variant induces most/least confidence.
    One figure per (model, dataset) combination.
    """
    _ensure_figures_dir(figures_dir)

    for model_tag, datasets in all_results.items():
        for dataset, records in datasets.items():
            variant_margins: Dict[int, List[float]] = defaultdict(list)
            for rec in records:
                for vr in rec.get("variant_results", []):
                    v = vr.get("variant")
                    lm = vr.get("logit_margin", float("nan"))
                    if v is not None and not np.isnan(lm):
                        variant_margins[v].append(lm)

            if not variant_margins:
                continue

            variants = sorted(variant_margins.keys())
            data = [variant_margins[v] for v in variants]
            labels = [f"V{v}" for v in variants]

            fig, ax = plt.subplots(figsize=(7, 4))
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("steelblue")
                patch.set_alpha(0.6)

            ax.set_xlabel("Prompt Variant")
            ax.set_ylabel("Logit Margin (top-1 − top-2)")
            ax.set_title(
                f"Confidence (Logit Margin) per Variant\n{model_tag} / {dataset}"
            )
            plt.tight_layout()

            safe_model = model_tag.replace("/", "_")
            out = os.path.join(figures_dir, f"logit_margin_by_variant_{safe_model}_{dataset}.pdf")
            plt.savefig(out, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out}")


def plot_logit_margin_variance_distribution(all_results: Dict, figures_dir: str):
    """Histogram of logit_margin_variance, SciQ vs TruthfulQA overlaid.

    Shows the overall distribution of confidence instability and the difference
    between datasets. One figure per model.
    """
    _ensure_figures_dir(figures_dir)

    dataset_colors = {"sciq": "steelblue", "truthfulqa": "tomato"}

    for model_tag, datasets in all_results.items():
        collected = {}
        for dataset, records in datasets.items():
            vals = [
                r["metrics"]["logit_margin_variance"]
                for r in records
                if r.get("metrics") and not np.isnan(r["metrics"].get("logit_margin_variance", float("nan")))
            ]
            if vals:
                collected[dataset] = vals

        if not collected:
            continue

        _, ax = plt.subplots(figsize=(8, 4))
        all_vals = [v for vals in collected.values() for v in vals]
        bins = np.linspace(0, min(np.percentile(all_vals, 98), 60), 40)

        for dataset, vals in collected.items():
            color = dataset_colors.get(dataset, "gray")
            ax.hist(
                vals, bins=bins, alpha=0.6, color=color,
                label=f"{dataset} (n={len(vals)}, mean={np.mean(vals):.2f})",
                density=True,
            )

        ax.axvline(
            x=LMVAR_INSTABILITY_THRESHOLD,
            color="black", linestyle="--", linewidth=0.9,
            label=f"instability threshold={LMVAR_INSTABILITY_THRESHOLD}",
        )
        ax.set_xlabel("Logit Margin Variance")
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of Confidence Instability — {model_tag}")
        ax.legend(fontsize=9)
        plt.tight_layout()

        safe_model = model_tag.replace("/", "_")
        out = os.path.join(figures_dir, f"lmvar_distribution_{safe_model}.pdf")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(results_dir: str, figures_dir: str):
    print(f"Loading results from: {results_dir}")
    all_results = load_all_results(results_dir)

    if not all_results:
        print("No result files found. Run run_experiment.py first.")
        return

    total_models = len(all_results)
    total_records = sum(
        len(recs) for datasets in all_results.values() for recs in datasets.values()
    )
    print(f"Found {total_models} model(s), {total_records} total sample records.\n")

    # Summary table
    print_summary_table(all_results)

    # Plots
    print("\nGenerating plots …")
    plot_uss_distributions(all_results, figures_dir)
    plot_flip_rate_bar(all_results, figures_dir)
    plot_lmvar_flip_comparison(all_results, figures_dir)
    plot_per_variant_accuracy(all_results, figures_dir)
    plot_logit_margin_by_variant(all_results, figures_dir)
    plot_logit_margin_variance_distribution(all_results, figures_dir)

    print(f"\nAll figures saved to: {figures_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse prompt-sensitivity experiment results")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--figures_dir", type=str, default=FIGURES_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.results_dir, args.figures_dir)
