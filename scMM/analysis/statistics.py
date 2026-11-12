"""Focused tests for targeted features and numeric associations."""

import numpy as np
import pandas as pd
from scipy import stats


def differential_summary(result, *, fdr_limit=.05, log2fc_limit=1.):
    """Add fold change and explicit A-versus-B direction/significance columns."""
    summary = result.copy()
    log2fc = summary["log2_fold_change"].to_numpy(dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        fc = np.exp2(log2fc)
    valid = ~np.isnan(log2fc)
    if {"mean_a", "mean_b"} <= set(summary):
        valid &= (summary.mean_a.to_numpy() >= 0) & (summary.mean_b.to_numpy() >= 0)
    summary["FC"] = np.where(valid, fc, np.nan)
    summary["FDR"] = summary["fdr"]
    summary["direction"] = np.select([~valid, log2fc > 0, log2fc < 0], ["无法判定", "上调", "下调"], default="不变")
    tested = valid & np.isfinite(summary["FDR"].to_numpy())
    significant = tested & (summary["FDR"].to_numpy() <= fdr_limit) & (np.abs(log2fc) >= log2fc_limit) & (log2fc != 0)
    summary["regulation"] = np.select([~tested, significant & (log2fc > 0), significant & (log2fc < 0)],
                                      ["无法判定", "上调", "下调"], default="无显著变化")
    return summary


def compare_groups(samples, method):
    samples = [np.asarray(sample, dtype=float) for sample in samples]
    samples = [sample[np.isfinite(sample)] for sample in samples]
    if len(samples) < 2 or any(len(sample) == 0 for sample in samples):
        return np.nan, np.nan
    if method == "ANOVA":
        result = stats.f_oneway(*samples)
    elif method == "Kruskal-Wallis":
        if np.ptp(np.concatenate(samples)) == 0:
            return 0.0, 1.0
        result = stats.kruskal(*samples)
    elif method == "Welch":
        result = stats.ttest_ind(*samples[:2], equal_var=False)
    elif method == "Mann-Whitney":
        result = stats.mannwhitneyu(*samples[:2], alternative="two-sided")
    else:
        raise ValueError(f"Unknown test: {method}")
    return float(result.statistic), float(result.pvalue)


def numeric_association(x, y, *, permutations=999, confidence=0.95, seed=42):
    """OLS mean-response interval and two-sided permutation test of Pearson r."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError("相关分析需要至少 3 个有效观测，且两个变量均有变化。")
    fit = stats.linregress(x, y)
    centered_x, centered_y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(centered_x) * np.linalg.norm(centered_y)
    observed = abs(np.dot(centered_x, centered_y) / denominator)
    rng = np.random.default_rng(seed)
    exceed = sum(
        abs(np.dot(centered_x, rng.permutation(centered_y)) / denominator)
        >= observed - 1e-12
        for _ in range(int(permutations))
    )
    grid = np.linspace(x.min(), x.max(), 200)
    prediction = fit.intercept + fit.slope * grid
    residual = y - (fit.intercept + fit.slope * x)
    standard_error = np.sqrt(np.sum(residual ** 2) / (len(x) - 2))
    margin = stats.t.ppf((1 + confidence) / 2, len(x) - 2) * standard_error * np.sqrt(
        1 / len(x) + (grid - x.mean()) ** 2 / np.sum(centered_x ** 2)
    )
    return {
        "points": pd.DataFrame({"x": x, "y": y}),
        "line": pd.DataFrame({"x": grid, "fit": prediction, "lower": prediction - margin, "upper": prediction + margin}),
        "n": len(x),
        "r2": float(fit.rvalue ** 2),
        "permutation_p": (exceed + 1) / (int(permutations) + 1),
        "permutations": int(permutations),
        "confidence": float(confidence),
    }
