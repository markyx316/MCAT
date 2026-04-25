"""
evaluation/metrics.py — Regression evaluation metrics for financial forecasting.
==================================================================================
Primary: MAE, RMSE, R², IC (Information Coefficient), Directional Accuracy
Secondary: Skill Score (improvement over persistence baseline)
Statistical: Paired t-test, bootstrap CI, Diebold-Mariano test
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FoldMetrics:
    """Metrics for a single walk-forward fold."""
    fold_num: int
    mae: float
    rmse: float
    r2: float
    ic: float                   # Information Coefficient (Pearson corr)
    directional_accuracy: float  # Fraction with correct sign
    skill_score: float          # 1 - MSE_model / MSE_naive
    n_samples: int

    def __repr__(self):
        return (
            f"Fold {self.fold_num:2d}: MAE={self.mae:.4f} | RMSE={self.rmse:.4f} | "
            f"R²={self.r2:.4f} | IC={self.ic:.4f} | DA={self.directional_accuracy:.3f} | "
            f"Skill={self.skill_score:.4f} | N={self.n_samples}"
        )


def compute_metrics(
    predictions: np.ndarray,
    actuals: np.ndarray,
    fold_num: int = 0,
) -> FoldMetrics:
    """
    Compute all regression metrics for a single fold.

    Args:
        predictions: (N,) array of predicted returns (in percentage points).
        actuals: (N,) array of actual returns (in percentage points).
        fold_num: Fold index for labeling.

    Returns:
        FoldMetrics dataclass.
    """
    assert len(predictions) == len(actuals), "Length mismatch"
    n = len(predictions)

    # MAE
    mae = np.mean(np.abs(predictions - actuals))

    # RMSE
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    # R² (coefficient of determination)
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-10)

    # IC (Information Coefficient = Pearson correlation)
    if np.std(predictions) > 1e-10 and np.std(actuals) > 1e-10:
        ic = np.corrcoef(predictions, actuals)[0, 1]
        if np.isnan(ic):
            ic = 0.0
    else:
        ic = 0.0

    # Directional Accuracy (fraction with correct sign)
    correct_sign = np.sign(predictions) == np.sign(actuals)
    # Handle exact zeros: count as correct if both are zero, else incorrect
    da = np.mean(correct_sign)

    # Skill Score: improvement over naive baseline (predict 0 = persistence)
    mse_model = np.mean((predictions - actuals) ** 2)
    mse_naive = np.mean(actuals ** 2)  # Naive prediction: return = 0
    skill_score = 1 - mse_model / (mse_naive + 1e-10)

    return FoldMetrics(
        fold_num=fold_num,
        mae=mae,
        rmse=rmse,
        r2=r2,
        ic=ic,
        directional_accuracy=da,
        skill_score=skill_score,
        n_samples=n,
    )


def aggregate_fold_metrics(fold_metrics: list) -> Dict:
    """
    Aggregate metrics across all folds (mean ± std).

    Args:
        fold_metrics: List of FoldMetrics objects.

    Returns:
        Dict with mean and std for each metric.
    """
    metrics_dict = {
        "mae": [f.mae for f in fold_metrics],
        "rmse": [f.rmse for f in fold_metrics],
        "r2": [f.r2 for f in fold_metrics],
        "ic": [f.ic for f in fold_metrics],
        "directional_accuracy": [f.directional_accuracy for f in fold_metrics],
        "skill_score": [f.skill_score for f in fold_metrics],
    }

    result = {}
    for name, values in metrics_dict.items():
        arr = np.array(values)
        result[name] = {
            "mean": arr.mean(),
            "std": arr.std(),
            "min": arr.min(),
            "max": arr.max(),
            "values": arr,
        }

    return result


def paired_t_test(
    metrics_a: list,
    metrics_b: list,
    metric_name: str = "ic",
) -> Tuple[float, float]:
    """
    Paired t-test across walk-forward folds.

    Tests H0: model_a and model_b have equal performance.
    Uses paired differences to control for fold difficulty variation.

    Args:
        metrics_a: List of FoldMetrics for model A.
        metrics_b: List of FoldMetrics for model B.
        metric_name: Which metric to test (e.g., "ic", "mae", "r2").

    Returns:
        (t_statistic, p_value)
    """
    values_a = np.array([getattr(f, metric_name) for f in metrics_a])
    values_b = np.array([getattr(f, metric_name) for f in metrics_b])

    t_stat, p_val = stats.ttest_rel(values_a, values_b)
    return t_stat, p_val


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for a metric.

    Args:
        values: Array of metric values (e.g., per-sample predictions).
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level.

    Returns:
        (lower_bound, upper_bound)
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    boot_means = np.array([
        values[rng.choice(n, n, replace=True)].mean()
        for _ in range(n_bootstrap)
    ])
    alpha = 1 - confidence
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, upper


def diebold_mariano_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    horizon: int = 3,
) -> Tuple[float, float]:
    """
    Diebold-Mariano test for comparing forecast accuracy.
    Accounts for serial correlation in prediction errors.

    H0: Models A and B have equal predictive accuracy.

    Args:
        errors_a: Prediction errors from model A.
        errors_b: Prediction errors from model B.
        horizon: Forecast horizon (for serial correlation adjustment).

    Returns:
        (DM_statistic, p_value)
    """
    d = errors_a ** 2 - errors_b ** 2  # Loss differential

    n = len(d)
    d_mean = np.mean(d)

    # Estimate long-run variance (Newey-West with h-1 lags)
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = 0
    for k in range(1, horizon):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        gamma_sum += gamma_k

    var_d = (gamma_0 + 2 * gamma_sum) / n

    if var_d <= 0:
        return 0.0, 1.0

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return dm_stat, p_value


def print_results_table(
    results: Dict[str, Dict],
    title: str = "Results Summary",
):
    """Print a formatted results table for multiple models."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  {'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'IC':>8} {'DA':>7} {'Skill':>8}")
    print(f"  {'─'*75}")

    for model_name, metrics in results.items():
        mae = metrics["mae"]
        rmse = metrics["rmse"]
        r2 = metrics["r2"]
        ic = metrics["ic"]
        da = metrics["directional_accuracy"]
        skill = metrics["skill_score"]
        print(
            f"  {model_name:<25} "
            f"{mae['mean']:>6.4f}±{mae['std']:.3f} "
            f"{rmse['mean']:>6.4f}±{rmse['std']:.3f} "
            f"{r2['mean']:>6.4f}±{r2['std']:.3f} "
            f"{ic['mean']:>6.4f}±{ic['std']:.3f} "
            f"{da['mean']:>5.3f} "
            f"{skill['mean']:>6.4f}±{skill['std']:.3f}"
        )
    print(f"{'='*80}")


if __name__ == "__main__":
    # Test with synthetic predictions
    np.random.seed(42)
    n = 500
    actuals = np.random.randn(n) * 3  # ~3pp std
    preds_good = actuals + np.random.randn(n) * 2  # Correlated with noise
    preds_bad = np.random.randn(n) * 3  # Uncorrelated

    m_good = compute_metrics(preds_good, actuals, fold_num=0)
    m_bad = compute_metrics(preds_bad, actuals, fold_num=0)

    print("Good model:", m_good)
    print("Bad model:", m_bad)

    # Test paired t-test
    good_folds = [compute_metrics(actuals + np.random.randn(n) * 2, actuals, i) for i in range(14)]
    bad_folds = [compute_metrics(np.random.randn(n) * 3, actuals, i) for i in range(14)]

    t_stat, p_val = paired_t_test(good_folds, bad_folds, "ic")
    print(f"\nPaired t-test (IC): t={t_stat:.3f}, p={p_val:.6f}")

    # Test aggregation
    agg = aggregate_fold_metrics(good_folds)
    print(f"\nAggregated good model: IC={agg['ic']['mean']:.4f}±{agg['ic']['std']:.4f}")
