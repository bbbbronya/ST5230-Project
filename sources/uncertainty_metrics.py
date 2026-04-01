"""
Metrics for measuring uncertainty stability across prompt variants.

Given N prompt variants for a single input x, each producing:
    - predicted_letter : str
    - entropy          : float   (predictive entropy over answer options)
    - logit_margin     : float   (top-1 minus top-2 logit)

We compute per-sample:

1. Uncertainty Stability Score (USS)
   USS(x) = 1 - std_p(U(x,p)) / (mean_p(U(x,p)) + ε)
   Range roughly [0, 1]; higher = more stable uncertainty estimates.

2. Prediction Flip (bool) and Flip Count
   Whether the predicted letter changes across any two prompt variants.

3. Logit Margin Variance
   Variance of the logit margin across prompt variants.

Aggregate functions produce per-model / per-dataset summaries.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

EPSILON = 1e-8  # numerical stability in USS denominator


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------

def compute_uss(entropies: List[float]) -> float:
    """
    Uncertainty Stability Score for a single input across prompt variants.

    Parameters
    ----------
    entropies : list of floats
        Predictive entropy U(x, p) for each prompt variant p.

    Returns
    -------
    float in (−∞, 1]; higher means more stable.
    """
    if len(entropies) < 2:
        return 1.0  # trivially stable with one variant
    arr = np.array(entropies, dtype=np.float64)
    std = float(np.std(arr, ddof=1))
    mean = float(np.mean(arr))
    return 1.0 - std / (mean + EPSILON)


def compute_flip(predictions: List[str]) -> bool:
    """True if any two prompt variants yield a different predicted letter."""
    return len(set(predictions)) > 1


def compute_flip_count(predictions: List[str]) -> int:
    """Number of distinct predicted letters observed across variants."""
    return len(set(predictions))


def compute_logit_margin_variance(margins: List[float]) -> float:
    """Variance of the logit margin across prompt variants."""
    if len(margins) < 2:
        return 0.0
    return float(np.var(margins, ddof=1))


def compute_entropy_variance(entropies: List[float]) -> float:
    """Variance of predictive entropy across prompt variants."""
    if len(entropies) < 2:
        return 0.0
    return float(np.var(entropies, ddof=1))


def compute_sample_metrics(variant_results: List[Dict]) -> Dict:
    """
    Compute all per-sample metrics from a list of per-variant inference results.

    Parameters
    ----------
    variant_results : list of dicts, each with keys:
        variant        : int
        predicted_letter : str
        entropy        : float
        logit_margin   : float
        answer_logits  : dict

    Returns
    -------
    dict with keys: uss, flip, flip_count, logit_margin_variance,
                    entropy_variance, mean_entropy, std_entropy,
                    mean_logit_margin
    """
    entropies = [r["entropy"] for r in variant_results]
    margins = [r["logit_margin"] for r in variant_results]
    preds = [r["predicted_letter"] for r in variant_results]

    return {
        "uss": compute_uss(entropies),
        "flip": compute_flip(preds),
        "flip_count": compute_flip_count(preds),
        "logit_margin_variance": compute_logit_margin_variance(margins),
        "entropy_variance": compute_entropy_variance(entropies),
        "mean_entropy": float(np.mean(entropies)),
        "std_entropy": float(np.std(entropies, ddof=1) if len(entropies) > 1 else 0.0),
        "mean_logit_margin": float(np.mean(margins)),
    }


# ---------------------------------------------------------------------------
# Aggregate metrics over a dataset split
# ---------------------------------------------------------------------------

def aggregate_metrics(sample_metrics_list: List[Dict]) -> Dict:
    """
    Compute dataset-level statistics from per-sample metric dicts.

    Returns
    -------
    dict with mean/std of each metric, plus prediction_flip_rate.
    """
    if not sample_metrics_list:
        return {}

    keys = [
        "uss", "logit_margin_variance", "entropy_variance",
        "mean_entropy", "std_entropy", "mean_logit_margin",
    ]
    agg: Dict = {}
    for k in keys:
        vals = [m[k] for m in sample_metrics_list if k in m]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))

    flips = [m["flip"] for m in sample_metrics_list]
    agg["prediction_flip_rate"] = float(np.mean(flips))
    agg["n_samples"] = len(sample_metrics_list)

    return agg


# ---------------------------------------------------------------------------
# "Hidden instability" detector
# ---------------------------------------------------------------------------

def find_hidden_instability(
    sample_records: List[Dict],
    uss_threshold: float = 0.5,
) -> List[Dict]:
    """
    Identify samples where predictions are stable (no flip) but uncertainty
    estimates fluctuate significantly (low USS).

    Parameters
    ----------
    sample_records : list of full sample result dicts (from run_experiment.py)
    uss_threshold  : samples with USS < this value are considered unstable

    Returns
    -------
    List of records flagged as hidden-instability cases.
    """
    flagged = []
    for rec in sample_records:
        metrics = rec.get("metrics", {})
        if not metrics.get("flip", True) and metrics.get("uss", 1.0) < uss_threshold:
            flagged.append(rec)
    return flagged


# ---------------------------------------------------------------------------
# Correlation analysis helpers
# ---------------------------------------------------------------------------

def correlation_uss_flip(sample_metrics_list: List[Dict]) -> Optional[float]:
    """
    Point-biserial correlation between USS and flip (0/1).
    Returns None if variance is zero.
    """
    from scipy.stats import pointbiserialr

    uss_vals = np.array([m["uss"] for m in sample_metrics_list])
    flip_vals = np.array([int(m["flip"]) for m in sample_metrics_list])

    if flip_vals.std() == 0 or uss_vals.std() == 0:
        return None

    corr, _ = pointbiserialr(flip_vals, uss_vals)
    return float(corr)


def correlation_entropy_var_flip(sample_metrics_list: List[Dict]) -> Optional[float]:
    """
    Point-biserial correlation between entropy variance and flip (0/1).
    """
    from scipy.stats import pointbiserialr

    ev_vals = np.array([m["entropy_variance"] for m in sample_metrics_list])
    flip_vals = np.array([int(m["flip"]) for m in sample_metrics_list])

    if flip_vals.std() == 0 or ev_vals.std() == 0:
        return None

    corr, _ = pointbiserialr(flip_vals, ev_vals)
    return float(corr)
