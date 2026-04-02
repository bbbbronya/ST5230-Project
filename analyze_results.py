"""
Analysis and visualisation of experiment results.

Generates 5 figures + 1 summary table for the report:

  Fig 1  Flip Rate + LM Variance (cross-model grouped bar, two subplots)
  Fig 2  LMVar vs Flip status (box plot, cross-model, by dataset)
  Fig 3  Per-Variant Accuracy (line plot, cross-model, by dataset)
  Fig 4  LM Variance distribution (histogram, cross-model, by dataset)
  Fig 5  Hidden instability scatter (mean LM vs LM variance, colored by flip)

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
from uncertainty_metrics import aggregate_metrics, correlation_uss_flip

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "outputs", "figures")

LMVAR_INSTABILITY_THRESHOLD = 5.0

# Display names for models (order matters: small → large)
MODEL_ORDER = [
    "mistralai_ministral-3b-2512",
    "meta-llama_llama-3.1-8b-instruct",
    "openai_gpt-4o",
]
MODEL_LABELS = {
    "mistralai_ministral-3b-2512": "Ministral-3B",
    "meta-llama_llama-3.1-8b-instruct": "Llama-3.1-8B",
    "openai_gpt-4o": "GPT-4o",
}
DATASET_LABELS = {"sciq": "SciQ", "truthfulqa": "TruthfulQA"}
DATASET_ORDER = ["sciq", "truthfulqa"]
MODEL_COLORS = {
    "mistralai_ministral-3b-2512": "#E24A33",
    "meta-llama_llama-3.1-8b-instruct": "#348ABD",
    "openai_gpt-4o": "#2CA02C",
}
DATASET_COLORS = {"sciq": "#348ABD", "truthfulqa": "#E24A33"}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_all_results(results_dir: str) -> Dict[str, Dict[str, List[dict]]]:
    all_results: Dict[str, Dict[str, List[dict]]] = {}
    for model_dir in sorted(os.listdir(results_dir)):
        model_path = os.path.join(results_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        if model_dir not in MODEL_ORDER:
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


def _get_model_label(tag: str) -> str:
    return MODEL_LABELS.get(tag, tag)


def _corr_lmvar_flip(metrics_list: List[dict]) -> Optional[float]:
    lmv = np.array([m["logit_margin_variance"] for m in metrics_list])
    flp = np.array([int(m["flip"]) for m in metrics_list])
    if flp.std() == 0 or lmv.std() == 0:
        return None
    corr, _ = pointbiserialr(flp, lmv)
    return float(corr)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(all_results: Dict) -> pd.DataFrame:
    rows = []
    for model_tag in MODEL_ORDER:
        if model_tag not in all_results:
            continue
        datasets = all_results[model_tag]
        for dataset in DATASET_ORDER:
            if dataset not in datasets:
                continue
            records = datasets[dataset]
            metrics_list = [r["metrics"] for r in records if r.get("metrics")]
            if not metrics_list:
                continue
            agg = aggregate_metrics(metrics_list)

            corr_lmv = _corr_lmvar_flip(metrics_list)

            # Hidden instability
            hi_count = sum(
                1 for r in records
                if r.get("metrics") and not r["metrics"].get("flip", True)
                and r["metrics"].get("logit_margin_variance", 0) > LMVAR_INSTABILITY_THRESHOLD
            )

            # Accuracy
            all_accs = []
            for rec in records:
                for vr in rec.get("variant_results", []):
                    if vr.get("predicted_letter") not in ("ERROR", None):
                        all_accs.append(int(vr["predicted_letter"] == rec["correct_letter"]))

            rows.append({
                "Model": _get_model_label(model_tag),
                "Dataset": DATASET_LABELS.get(dataset, dataset),
                "N": agg.get("n_samples"),
                "Accuracy": f"{np.mean(all_accs):.3f}",
                "Flip Rate": f"{agg.get('prediction_flip_rate', 0):.3f}",
                "LMVar Mean": f"{agg.get('logit_margin_variance_mean', 0):.2f}",
                "Corr(LMVar,Flip)": f"{corr_lmv:.3f}" if corr_lmv is not None else "N/A",
                "Hidden Instab.": f"{hi_count} ({hi_count / agg.get('n_samples', 1) * 100:.1f}%)",
            })

    if not rows:
        print("No results found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print("\n=== Table 1: Aggregate Results ===\n")
    print(df.to_string(index=False))
    print(f"\n  Hidden Instab. = flip=False AND logit_margin_variance > {LMVAR_INSTABILITY_THRESHOLD}")
    return df


# ---------------------------------------------------------------------------
# Fig 1: Flip Rate + LM Variance (grouped bar, two subplots)
# ---------------------------------------------------------------------------

def plot_fig1_main_results(all_results: Dict, figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    models_present = [m for m in MODEL_ORDER if m in all_results]
    n_models = len(models_present)
    n_datasets = len(DATASET_ORDER)
    x = np.arange(n_models)
    width = 0.30

    for ax, metric_key, ylabel, title in [
        (ax1, "flip_rate", "Prediction Flip Rate", "Prediction Flip Rate"),
        (ax2, "lm_var", "Logit Margin Variance (mean)", "Logit Margin Variance"),
    ]:
        for d_idx, dataset in enumerate(DATASET_ORDER):
            vals = []
            for model_tag in models_present:
                records = all_results.get(model_tag, {}).get(dataset, [])
                metrics_list = [r["metrics"] for r in records if r.get("metrics")]
                if metric_key == "flip_rate":
                    v = np.mean([m["flip"] for m in metrics_list]) if metrics_list else 0
                else:
                    v = np.mean([m["logit_margin_variance"] for m in metrics_list]) if metrics_list else 0
                vals.append(v)

            offset = (d_idx - (n_datasets - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width,
                          label=DATASET_LABELS.get(dataset, dataset),
                          color=DATASET_COLORS[dataset], alpha=0.8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}" if metric_key == "flip_rate" else f"{val:.2f}",
                        ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([_get_model_label(m) for m in models_present])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        if metric_key == "flip_rate":
            ax.set_ylim(0, min(1.0, max(vals) * 1.5 + 0.05))

    plt.tight_layout()
    out = os.path.join(figures_dir, "fig1_main_results.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 2: LMVar vs Flip status (box plot, by dataset)
# ---------------------------------------------------------------------------

def plot_fig2_lmvar_vs_flip(all_results: Dict, figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    models_present = [m for m in MODEL_ORDER if m in all_results]

    for ax_idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[ax_idx]
        positions = []
        data = []
        colors = []
        labels_done = set()
        tick_positions = []
        tick_labels = []

        for m_idx, model_tag in enumerate(models_present):
            records = all_results.get(model_tag, {}).get(dataset, [])
            if not records:
                continue

            lmv_no_flip = [r["metrics"]["logit_margin_variance"]
                           for r in records if r.get("metrics") and not r["metrics"]["flip"]]
            lmv_flip = [r["metrics"]["logit_margin_variance"]
                        for r in records if r.get("metrics") and r["metrics"]["flip"]]

            base = m_idx * 3
            if lmv_no_flip:
                positions.append(base)
                data.append(lmv_no_flip)
                colors.append("steelblue")
            if lmv_flip:
                positions.append(base + 1)
                data.append(lmv_flip)
                colors.append("tomato")
            tick_positions.append(base + 0.5)
            tick_labels.append(_get_model_label(model_tag))

        if data:
            bp = ax.boxplot(data, positions=positions, widths=0.7,
                            patch_artist=True, notch=False)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.axhline(y=LMVAR_INSTABILITY_THRESHOLD, color="gray",
                    linestyle="--", linewidth=0.8, label=f"threshold={LMVAR_INSTABILITY_THRESHOLD}")
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)}")
        if ax_idx == 0:
            ax.set_ylabel("Logit Margin Variance")

        # Manual legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="steelblue", alpha=0.7, label="No Flip"),
            Patch(facecolor="tomato", alpha=0.7, label="Flip"),
        ]
        ax.legend(handles=legend_elements, fontsize=9)

    fig.suptitle("Confidence Instability by Flip Status", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(figures_dir, "fig2_lmvar_vs_flip.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 3: Per-Variant Accuracy (line plot, by dataset)
# ---------------------------------------------------------------------------

def plot_fig3_variant_accuracy(all_results: Dict, figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    models_present = [m for m in MODEL_ORDER if m in all_results]
    variants = [0, 1, 2, 3]
    variant_labels = ["V0\nBaseline", "V1\nReworded", "V2\nQuestion-first", "V3\nMinimal"]

    for ax_idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[ax_idx]
        for model_tag in models_present:
            records = all_results.get(model_tag, {}).get(dataset, [])
            if not records:
                continue

            means = []
            stds = []
            for v in variants:
                accs = [
                    int(r["variant_results"][v]["predicted_letter"] == r["correct_letter"])
                    for r in records if len(r.get("variant_results", [])) > v
                    and r["variant_results"][v].get("predicted_letter") not in ("ERROR", None)
                ]
                means.append(np.mean(accs) if accs else 0)
                stds.append(np.std(accs) / np.sqrt(len(accs)) if len(accs) > 1 else 0)

            color = MODEL_COLORS[model_tag]
            ax.errorbar(variants, means, yerr=stds, marker="o", capsize=4,
                        label=_get_model_label(model_tag), color=color, linewidth=2)

        ax.set_xticks(variants)
        ax.set_xticklabels(variant_labels, fontsize=9)
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_ylim(0.4, 1.02)
        ax.legend(fontsize=9)
        if ax_idx == 0:
            ax.set_ylabel("Accuracy")

    fig.suptitle("Accuracy per Prompt Variant", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(figures_dir, "fig3_variant_accuracy.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 4: LM Variance distribution (histogram, by dataset)
# ---------------------------------------------------------------------------

def plot_fig4_lmvar_distribution(all_results: Dict, figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    models_present = [m for m in MODEL_ORDER if m in all_results]

    for ax_idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[ax_idx]
        all_vals = []
        for model_tag in models_present:
            records = all_results.get(model_tag, {}).get(dataset, [])
            vals = [r["metrics"]["logit_margin_variance"]
                    for r in records if r.get("metrics")]
            all_vals.extend(vals)

        if not all_vals:
            continue
        bins = np.linspace(0, min(np.percentile(all_vals, 97), 30), 35)

        for model_tag in models_present:
            records = all_results.get(model_tag, {}).get(dataset, [])
            vals = [r["metrics"]["logit_margin_variance"]
                    for r in records if r.get("metrics")]
            if not vals:
                continue
            color = MODEL_COLORS[model_tag]
            ax.hist(vals, bins=bins, alpha=0.5, color=color,
                    label=f"{_get_model_label(model_tag)} (mean={np.mean(vals):.2f})",
                    density=True)

        ax.axvline(x=LMVAR_INSTABILITY_THRESHOLD, color="black",
                    linestyle="--", linewidth=0.9, label=f"threshold={LMVAR_INSTABILITY_THRESHOLD}")
        ax.set_xlabel("Logit Margin Variance")
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.legend(fontsize=8)
        if ax_idx == 0:
            ax.set_ylabel("Density")

    fig.suptitle("Distribution of Confidence Instability", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(figures_dir, "fig4_lmvar_distribution.pdf")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 5: Hidden instability scatter
# ---------------------------------------------------------------------------

def plot_fig5_hidden_instability(all_results: Dict, figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models_present = [m for m in MODEL_ORDER if m in all_results]

    for ax_idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[ax_idx]

        for model_tag in models_present:
            records = all_results.get(model_tag, {}).get(dataset, [])
            if not records:
                continue

            x_no_flip, y_no_flip = [], []
            x_flip, y_flip = [], []

            for r in records:
                m = r.get("metrics", {})
                if not m:
                    continue
                mean_lm = m.get("mean_logit_margin", 0)
                lmvar = m.get("logit_margin_variance", 0)
                if m.get("flip", False):
                    x_flip.append(mean_lm)
                    y_flip.append(lmvar)
                else:
                    x_no_flip.append(mean_lm)
                    y_no_flip.append(lmvar)

            color = MODEL_COLORS[model_tag]
            label = _get_model_label(model_tag)
            ax.scatter(x_no_flip, y_no_flip, c=color, marker="o", alpha=0.3, s=20,
                       label=f"{label} (no flip)")
            ax.scatter(x_flip, y_flip, c=color, marker="x", alpha=0.8, s=30,
                       label=f"{label} (flip)")

        ax.axhline(y=LMVAR_INSTABILITY_THRESHOLD, color="gray",
                    linestyle="--", linewidth=0.8)
        ax.set_xlabel("Mean Logit Margin")
        ax.set_ylabel("Logit Margin Variance")
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.legend(fontsize=7, ncol=2, loc="upper right")

    fig.suptitle("Hidden Instability: Stable Predictions with Fluctuating Confidence",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(figures_dir, "fig5_hidden_instability.pdf")
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

    # Figures
    print("\nGenerating figures …")
    plot_fig1_main_results(all_results, figures_dir)
    plot_fig2_lmvar_vs_flip(all_results, figures_dir)
    plot_fig3_variant_accuracy(all_results, figures_dir)
    plot_fig4_lmvar_distribution(all_results, figures_dir)
    plot_fig5_hidden_instability(all_results, figures_dir)

    print(f"\nAll figures saved to: {figures_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse prompt-sensitivity experiment results")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--figures_dir", type=str, default=FIGURES_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.results_dir, args.figures_dir)
