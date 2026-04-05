"""
Analysis and visualisation of experiment results.

Organised into three report sections + supplementary figures:

  === Overall Stability ===
  Fig 1  Flip Rate + LM Variance (cross-model grouped bar)
  Fig 2  USS Distribution (box plot, cross-model, by dataset)
  Fig 3  Stability Heatmap (model × dataset, multiple metrics)

  === Uncertainty-Flip Correlation ===
  Fig 4  LMVar vs Flip status (box plot, cross-model, by dataset)
  Fig 5  Uncertainty-Flip Scatter (LMVar vs Mean LM, colored by flip)

  === Prompt Type Comparison ===
  Fig 6  Per-Variant Accuracy (line plot, cross-model, by dataset)
  Fig 7  Per-Variant LM Variance (bar chart, cross-model, by dataset)

  === Supplementary ===
  Fig 8  LM Variance Distribution (histogram overlay)

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sources"))
from uncertainty_metrics import aggregate_metrics, correlation_uss_flip

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "outputs", "figures")

LMVAR_INSTABILITY_THRESHOLD = 5.0

# Display names for models (order: small → large)
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
VARIANT_LABELS = ["V0\nBaseline", "V1\nReworded", "V2\nQuestion-first", "V3\nMinimal"]


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


def _label(tag: str) -> str:
    return MODEL_LABELS.get(tag, tag)


def _corr_lmvar_flip(metrics_list: List[dict]) -> Optional[float]:
    lmv = np.array([m["logit_margin_variance"] for m in metrics_list])
    flp = np.array([int(m["flip"]) for m in metrics_list])
    if flp.std() == 0 or lmv.std() == 0:
        return None
    corr, _ = pointbiserialr(flp, lmv)
    return float(corr)


def _models_present(all_results: Dict) -> List[str]:
    return [m for m in MODEL_ORDER if m in all_results]


def _ensure(figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# Summary table (Table 1)
# ---------------------------------------------------------------------------

def print_summary_table(all_results: Dict) -> pd.DataFrame:
    rows = []
    for model_tag in MODEL_ORDER:
        if model_tag not in all_results:
            continue
        for dataset in DATASET_ORDER:
            records = all_results[model_tag].get(dataset, [])
            if not records:
                continue
            metrics_list = [r["metrics"] for r in records if r.get("metrics")]
            if not metrics_list:
                continue
            agg = aggregate_metrics(metrics_list)
            corr_lmv = _corr_lmvar_flip(metrics_list)
            hi_count = sum(
                1 for r in records
                if r.get("metrics") and not r["metrics"].get("flip", True)
                and r["metrics"].get("logit_margin_variance", 0) > LMVAR_INSTABILITY_THRESHOLD
            )
            all_accs = []
            for rec in records:
                for vr in rec.get("variant_results", []):
                    if vr.get("predicted_letter") not in ("ERROR", None):
                        all_accs.append(int(vr["predicted_letter"] == rec["correct_letter"]))
            n = agg.get("n_samples", 1)
            rows.append({
                "Model": _label(model_tag),
                "Dataset": DATASET_LABELS.get(dataset, dataset),
                "N": n,
                "Accuracy": f"{np.mean(all_accs):.3f}",
                "Flip Rate": f"{agg.get('prediction_flip_rate', 0):.3f}",
                "USS": f"{agg.get('uss_mean', 0):.3f}",
                "LMVar": f"{agg.get('logit_margin_variance_mean', 0):.2f}",
                "r(LMVar,Flip)": f"{corr_lmv:.3f}" if corr_lmv is not None else "N/A",
                "Hidden Instab.": f"{hi_count} ({hi_count / n * 100:.1f}%)",
            })

    if not rows:
        print("No results found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print("\n=== Table 1: Aggregate Results ===\n")
    print(df.to_string(index=False))
    print(f"\n  Hidden Instab. = flip=False AND logit_margin_variance > {LMVAR_INSTABILITY_THRESHOLD}")
    return df


# ===================================================================
# SECTION A: Overall Stability
# ===================================================================

# --- Fig 1: Flip Rate + LM Variance (grouped bar) ---

def plot_fig1_main_results(all_results: Dict, figures_dir: str):
    _ensure(figures_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    models = _models_present(all_results)
    x = np.arange(len(models))
    width = 0.30

    for ax, metric_key, ylabel, title in [
        (ax1, "flip_rate", "Prediction Flip Rate", "Prediction Flip Rate"),
        (ax2, "lm_var", "Logit Margin Variance (mean)", "Logit Margin Variance"),
    ]:
        for d_idx, dataset in enumerate(DATASET_ORDER):
            vals = []
            for mt in models:
                ml = [r["metrics"] for r in all_results.get(mt, {}).get(dataset, []) if r.get("metrics")]
                if metric_key == "flip_rate":
                    vals.append(np.mean([m["flip"] for m in ml]) if ml else 0)
                else:
                    vals.append(np.mean([m["logit_margin_variance"] for m in ml]) if ml else 0)
            offset = (d_idx - 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          label=DATASET_LABELS[dataset], color=DATASET_COLORS[dataset], alpha=0.8)
            for bar, val in zip(bars, vals):
                fmt = f"{val:.3f}" if metric_key == "flip_rate" else f"{val:.2f}"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        fmt, ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([_label(m) for m in models])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    plt.tight_layout()
    out = os.path.join(figures_dir, "fig1_main_results.pdf")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


# --- Fig 2: USS Distribution (box plot) ---

def plot_fig2_uss_distribution(all_results: Dict, figures_dir: str):
    _ensure(figures_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    models = _models_present(all_results)
    for ax_idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[ax_idx]
        data_list, labels = [], []
        for mt in models:
            records = all_results.get(mt, {}).get(dataset, [])
            vals = [r["metrics"]["uss"] for r in records
                    if r.get("metrics") and not np.isnan(r["metrics"].get("uss", float("nan")))]
            if vals:
                data_list.append(vals)
                labels.append(_label(mt))

        if data_list:
            bp = ax.boxplot(data_list, tick_labels=labels, patch_artist=True, notch=False)
            for patch, mt in zip(bp["boxes"], models):
                patch.set_facecolor(MODEL_COLORS.get(mt, "gray"))
                patch.set_alpha(0.7)

        ax.axhline(y=0.5, color="red", linestyle="--", linewidth=0.8, label="USS=0.5")
        ax.axhline(y=0.0, color="gray", linestyle=":", linewidth=0.6)
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_ylim(-2.5, 1.1)
        ax.legend(fontsize=8)
        if ax_idx == 0:
            ax.set_ylabel("Uncertainty Stability Score (USS)")

    fig.suptitle("USS Distribution Across Models", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(figures_dir, "fig2_uss_distribution.pdf")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


# --- Fig 3: Stability Heatmap ---

def plot_fig3_stability_heatmap(all_results: Dict, figures_dir: str):
    _ensure(figures_dir)

    models = _models_present(all_results)
    metrics_to_show = [
        ("Flip Rate", lambda ml: np.mean([m["flip"] for m in ml])),
        ("LMVar Mean", lambda ml: np.mean([m["logit_margin_variance"] for m in ml])),
        ("USS Mean", lambda ml: np.mean([m["uss"] for m in ml])),
        ("Accuracy", None),  # special handling
    ]
    n_metrics = len(metrics_to_show)
    n_models = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax_idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[ax_idx]
        matrix = np.zeros((n_metrics, n_models))

        for m_idx, mt in enumerate(models):
            records = all_results.get(mt, {}).get(dataset, [])
            ml = [r["metrics"] for r in records if r.get("metrics")]
            for k_idx, (name, fn) in enumerate(metrics_to_show):
                if name == "Accuracy":
                    accs = []
                    for rec in records:
                        for vr in rec.get("variant_results", []):
                            if vr.get("predicted_letter") not in ("ERROR", None):
                                accs.append(int(vr["predicted_letter"] == rec["correct_letter"]))
                    matrix[k_idx, m_idx] = np.mean(accs) if accs else 0
                else:
                    matrix[k_idx, m_idx] = fn(ml) if ml else 0

        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(n_models))
        ax.set_xticklabels([_label(m) for m in models], fontsize=9)
        ax.set_yticks(range(n_metrics))
        ax.set_yticklabels([m[0] for m in metrics_to_show], fontsize=9)
        ax.set_title(DATASET_LABELS.get(dataset, dataset))

        # Annotate cells
        for i in range(n_metrics):
            for j in range(n_models):
                val = matrix[i, j]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9,
                        color="white" if val > matrix.mean() else "black")

    fig.suptitle("Stability Overview (Higher = Less Stable, Except Accuracy & USS)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = os.path.join(figures_dir, "fig3_stability_heatmap.pdf")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


# ===================================================================
# SECTION B: Uncertainty-Flip Correlation
# ===================================================================

# --- Fig 4: LMVar vs Flip status (box plot) ---

def plot_fig4_lmvar_vs_flip(all_results: Dict, figures_dir: str):
    _ensure(figures_dir)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    models = _models_present(all_results)
    for ax_idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[ax_idx]
        positions, data, colors = [], [], []
        tick_positions, tick_labels = [], []

        for m_idx, mt in enumerate(models):
            records = all_results.get(mt, {}).get(dataset, [])
            lmv_no = [r["metrics"]["logit_margin_variance"]
                      for r in records if r.get("metrics") and not r["metrics"]["flip"]]
            lmv_yes = [r["metrics"]["logit_margin_variance"]
                       for r in records if r.get("metrics") and r["metrics"]["flip"]]
            base = m_idx * 3
            if lmv_no:
                positions.append(base); data.append(lmv_no); colors.append("steelblue")
            if lmv_yes:
                positions.append(base + 1); data.append(lmv_yes); colors.append("tomato")
            tick_positions.append(base + 0.5)
            tick_labels.append(_label(mt))

        if data:
            bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True)
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c); patch.set_alpha(0.7)

        ax.axhline(y=LMVAR_INSTABILITY_THRESHOLD, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks(tick_positions); ax.set_xticklabels(tick_labels)
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        if ax_idx == 0:
            ax.set_ylabel("Logit Margin Variance")
        ax.legend(handles=[
            Patch(facecolor="steelblue", alpha=0.7, label="No Flip"),
            Patch(facecolor="tomato", alpha=0.7, label="Flip"),
        ], fontsize=9)

    fig.suptitle("Confidence Instability by Flip Status", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(figures_dir, "fig4_lmvar_vs_flip.pdf")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


# --- Fig 5: Uncertainty-Flip Scatter ---

def plot_fig5_uncertainty_flip_scatter(all_results: Dict, figures_dir: str):
    _ensure(figures_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    models = _models_present(all_results)
    for ax_idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[ax_idx]
        for mt in models:
            records = all_results.get(mt, {}).get(dataset, [])
            x_no, y_no, x_fl, y_fl = [], [], [], []
            for r in records:
                m = r.get("metrics", {})
                if not m:
                    continue
                mean_lm = m.get("mean_logit_margin", 0)
                lmvar = m.get("logit_margin_variance", 0)
                if m.get("flip", False):
                    x_fl.append(mean_lm); y_fl.append(lmvar)
                else:
                    x_no.append(mean_lm); y_no.append(lmvar)

            c = MODEL_COLORS[mt]
            lab = _label(mt)
            ax.scatter(x_no, y_no, c=c, marker="o", alpha=0.25, s=18, label=f"{lab} (stable)")
            ax.scatter(x_fl, y_fl, c=c, marker="X", alpha=0.85, s=35, label=f"{lab} (flip)")

        ax.axhline(y=LMVAR_INSTABILITY_THRESHOLD, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Mean Logit Margin (confidence)")
        ax.set_ylabel("Logit Margin Variance (instability)")
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.legend(fontsize=7, ncol=2, loc="upper right")

    fig.suptitle("Uncertainty vs Prediction Stability", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(figures_dir, "fig5_uncertainty_flip_scatter.pdf")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


# ===================================================================
# SECTION C: Prompt Type Comparison
# ===================================================================

# --- Fig 6: Per-Variant Accuracy ---

def plot_fig6_variant_accuracy(all_results: Dict, figures_dir: str):
    _ensure(figures_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    models = _models_present(all_results)
    variants = [0, 1, 2, 3]

    for ax_idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[ax_idx]
        for mt in models:
            records = all_results.get(mt, {}).get(dataset, [])
            means, stds = [], []
            for v in variants:
                accs = [int(r["variant_results"][v]["predicted_letter"] == r["correct_letter"])
                        for r in records if len(r.get("variant_results", [])) > v
                        and r["variant_results"][v].get("predicted_letter") not in ("ERROR", None)]
                means.append(np.mean(accs) if accs else 0)
                stds.append(np.std(accs) / np.sqrt(len(accs)) if len(accs) > 1 else 0)
            ax.errorbar(variants, means, yerr=stds, marker="o", capsize=4,
                        label=_label(mt), color=MODEL_COLORS[mt], linewidth=2)

        ax.set_xticks(variants)
        ax.set_xticklabels(VARIANT_LABELS, fontsize=9)
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.set_ylim(0.35, 1.02)
        ax.legend(fontsize=9)
        if ax_idx == 0:
            ax.set_ylabel("Accuracy")

    fig.suptitle("Accuracy per Prompt Variant", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(figures_dir, "fig6_variant_accuracy.pdf")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


# --- Fig 7: Per-Variant LM Variance (grouped bar) ---

def plot_fig7_variant_lmvar(all_results: Dict, figures_dir: str):
    """Mean logit margin per variant — shows which prompt format induces
    the most/least confidence and therefore the most instability."""
    _ensure(figures_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    models = _models_present(all_results)
    variants = [0, 1, 2, 3]
    n_models = len(models)
    width = 0.22

    for ax_idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[ax_idx]
        x = np.arange(len(variants))

        for m_idx, mt in enumerate(models):
            records = all_results.get(mt, {}).get(dataset, [])
            means = []
            for v in variants:
                margins = [r["variant_results"][v]["logit_margin"]
                           for r in records if len(r.get("variant_results", [])) > v
                           and not np.isnan(r["variant_results"][v].get("logit_margin", float("nan")))]
                means.append(np.mean(margins) if margins else 0)

            offset = (m_idx - (n_models - 1) / 2) * width
            bars = ax.bar(x + offset, means, width, label=_label(mt),
                          color=MODEL_COLORS[mt], alpha=0.8)
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(VARIANT_LABELS, fontsize=9)
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.legend(fontsize=8)
        if ax_idx == 0:
            ax.set_ylabel("Mean Logit Margin")

    fig.suptitle("Confidence (Logit Margin) per Prompt Variant", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(figures_dir, "fig7_variant_logit_margin.pdf")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")


# ===================================================================
# Supplementary
# ===================================================================

# --- Fig 8: LM Variance Distribution ---

def plot_fig8_lmvar_distribution(all_results: Dict, figures_dir: str):
    _ensure(figures_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    models = _models_present(all_results)
    for ax_idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[ax_idx]
        all_vals = []
        for mt in models:
            vals = [r["metrics"]["logit_margin_variance"]
                    for r in all_results.get(mt, {}).get(dataset, []) if r.get("metrics")]
            all_vals.extend(vals)
        if not all_vals:
            continue
        bins = np.linspace(0, min(np.percentile(all_vals, 97), 30), 35)

        for mt in models:
            vals = [r["metrics"]["logit_margin_variance"]
                    for r in all_results.get(mt, {}).get(dataset, []) if r.get("metrics")]
            if not vals:
                continue
            ax.hist(vals, bins=bins, alpha=0.5, color=MODEL_COLORS[mt],
                    label=f"{_label(mt)} (mean={np.mean(vals):.2f})", density=True)

        ax.axvline(x=LMVAR_INSTABILITY_THRESHOLD, color="black", linestyle="--", linewidth=0.9)
        ax.set_xlabel("Logit Margin Variance")
        ax.set_title(DATASET_LABELS.get(dataset, dataset))
        ax.legend(fontsize=8)
        if ax_idx == 0:
            ax.set_ylabel("Density")

    fig.suptitle("Distribution of Confidence Instability", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(figures_dir, "fig8_lmvar_distribution.pdf")
    plt.savefig(out, bbox_inches="tight"); plt.close()
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

    # === Overall Stability ===
    print("\n--- Overall Stability ---")
    plot_fig1_main_results(all_results, figures_dir)
    plot_fig2_uss_distribution(all_results, figures_dir)
    plot_fig3_stability_heatmap(all_results, figures_dir)

    # === Uncertainty-Flip Correlation ===
    print("\n--- Uncertainty-Flip Correlation ---")
    plot_fig4_lmvar_vs_flip(all_results, figures_dir)
    plot_fig5_uncertainty_flip_scatter(all_results, figures_dir)

    # === Prompt Type Comparison ===
    print("\n--- Prompt Type Comparison ---")
    plot_fig6_variant_accuracy(all_results, figures_dir)
    plot_fig7_variant_lmvar(all_results, figures_dir)

    # === Supplementary ===
    print("\n--- Supplementary ---")
    plot_fig8_lmvar_distribution(all_results, figures_dir)

    print(f"\nAll figures ({8} total) saved to: {figures_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyse prompt-sensitivity experiment results")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--figures_dir", type=str, default=FIGURES_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.results_dir, args.figures_dir)
