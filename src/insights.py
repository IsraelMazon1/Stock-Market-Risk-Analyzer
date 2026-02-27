from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.portfolio import normalize_weights


@dataclass(frozen=True)
class Insight:
    title: str
    severity: str  # "low" | "medium" | "high"
    message: str


def _hhi(weights: np.ndarray) -> float:
    # Herfindahl–Hirschman Index: sum(w^2)
    return float(np.sum(weights**2))


def _avg_pairwise_correlation(corr: pd.DataFrame) -> float:
    if corr.shape[0] < 2:
        return 0.0
    # average of upper triangle excluding diagonal
    vals = corr.values
    n = vals.shape[0]
    upper = []
    for i in range(n):
        for j in range(i + 1, n):
            upper.append(vals[i, j])
    return float(np.mean(upper)) if upper else 0.0


def _risk_contribution_proxy(
    log_returns: pd.DataFrame,
    weights: np.ndarray
) -> pd.Series:
    """
    Simple, interpretable proxy:
    contribution_i ∝ w_i * annualized_vol_i
    (not true marginal contribution, but stable and easy to explain)
    """
    ann_vol = log_returns.std(ddof=1) * np.sqrt(252)
    contrib = pd.Series(weights, index=log_returns.columns) * ann_vol
    # normalize to sum to 1 for display
    s = float(contrib.sum())
    return (contrib / s) if s > 0 else contrib


def generate_insights(
    tickers: List[str],
    weights_raw: np.ndarray,
    portfolio_returns: pd.Series,
    log_returns_assets: pd.DataFrame,
    confidence_level: float,
    var_dollars: float,
    cvar_dollars: float,
    backtest_norm: Dict,
    backtest_hist: Dict,
    diag: Dict,
) -> Tuple[List[Insight], Dict]:
    """
    Returns:
      - list of human-readable insights (rule-based)
      - a dict of supporting computed stats for display
    """
    weights = normalize_weights(weights_raw)

    # Supporting stats
    hhi = _hhi(weights)
    max_w = float(np.max(weights))
    max_ticker = tickers[int(np.argmax(weights))]
    corr = log_returns_assets.corr()
    avg_corr = _avg_pairwise_correlation(corr)

    cvar_var_ratio = float(cvar_dollars / var_dollars) if var_dollars > 0 else float("nan")

    contrib = _risk_contribution_proxy(log_returns_assets, weights).sort_values(ascending=False)

    # Risk level heuristic (very simple + interpretable)
    # Based on VaR as % of portfolio value using the provided dollars.
    # Caller can compute var_pct if it passes portfolio value; we infer from returns if needed.
    # Here we just categorize using typical daily VaR bands (heuristic).
    # You can tune these thresholds later.
    var_pct_est = float(var_dollars)  # placeholder; caller may overwrite in supporting dict

    insights: List[Insight] = []

    # 1) Concentration
    # HHI ranges:
    # - equal-weight N: HHI = 1/N (e.g., N=4 -> 0.25)
    # - concentrated: closer to 1.0
    if max_w >= 0.60 or hhi >= 0.45:
        insights.append(Insight(
            title="High concentration risk",
            severity="high",
            message=(
                f"Your allocation is dominated by **{max_ticker}** (largest weight: **{max_w:.0%}**). "
                f"HHI={hhi:.3f} suggests high concentration, so single-name moves can drive portfolio risk."
            ),
        ))
    elif max_w >= 0.40 or hhi >= 0.30:
        insights.append(Insight(
            title="Moderate concentration risk",
            severity="medium",
            message=(
                f"Your largest position is **{max_ticker}** at **{max_w:.0%}**. "
                f"HHI={hhi:.3f} indicates moderate concentration. Consider spreading weight if you want smoother risk."
            ),
        ))
    else:
        insights.append(Insight(
            title="Well-distributed weights",
            severity="low",
            message=(
                f"Your weights are relatively balanced (largest position: **{max_ticker}** at **{max_w:.0%}**). "
                f"HHI={hhi:.3f} suggests good diversification by allocation."
            ),
        ))

    # 2) Diversification via correlation
    if avg_corr >= 0.75:
        insights.append(Insight(
            title="Limited diversification benefit",
            severity="medium",
            message=(
                f"Assets are strongly correlated on average (avg corr ≈ **{avg_corr:.2f}**). "
                "Even with multiple tickers, drawdowns can happen together in stress periods."
            ),
        ))
    elif avg_corr <= 0.30 and len(tickers) >= 2:
        insights.append(Insight(
            title="Strong diversification benefit",
            severity="low",
            message=(
                f"Average correlation is relatively low (avg corr ≈ **{avg_corr:.2f}**), "
                "which generally improves diversification and can reduce portfolio volatility and tail risk."
            ),
        ))
    else:
        insights.append(Insight(
            title="Moderate diversification benefit",
            severity="low",
            message=(
                f"Average correlation is moderate (avg corr ≈ **{avg_corr:.2f}**). "
                "Diversification helps, but tail risk can still rise during high-volatility regimes."
            ),
        ))

    # 3) Tail severity: CVaR vs VaR
    if np.isfinite(cvar_var_ratio):
        if cvar_var_ratio >= 1.6:
            sev = "high"
            label = "heavy tail severity"
        elif cvar_var_ratio >= 1.3:
            sev = "medium"
            label = "moderate tail severity"
        else:
            sev = "low"
            label = "milder tail severity"

        insights.append(Insight(
            title="Tail loss severity (CVaR vs VaR)",
            severity=sev,
            message=(
                f"CVaR/VaR ≈ **{cvar_var_ratio:.2f}** ({label}). "
                "This ratio summarizes how much worse losses are beyond the VaR threshold."
            ),
        ))

    # 4) Model calibration via backtesting
    exp = backtest_norm.get("expected_violation_rate", 1 - confidence_level)
    obs_norm = backtest_norm.get("observed_violation_rate", float("nan"))
    obs_hist = backtest_hist.get("observed_violation_rate", float("nan"))

    # Simple interpretation thresholds
    def _calib_message(obs: float, name: str) -> Insight:
        if not np.isfinite(obs):
            return Insight(title=f"{name} backtest", severity="medium",
                           message="Not enough data after window to compute backtest reliably.")
        diff = obs - exp
        if diff > 0.02:
            return Insight(title=f"{name} VaR underestimates risk", severity="high",
                           message=f"Observed violation rate **{obs:.2%}** exceeds expected **{exp:.2%}** by **{diff:+.2%}**.")
        if diff > 0.005:
            return Insight(title=f"{name} VaR slightly underestimates risk", severity="medium",
                           message=f"Observed violation rate **{obs:.2%}** is above expected **{exp:.2%}** by **{diff:+.2%}**.")
        if diff < -0.02:
            return Insight(title=f"{name} VaR may be conservative", severity="low",
                           message=f"Observed violation rate **{obs:.2%}** is below expected **{exp:.2%}** by **{diff:+.2%}**.")
        return Insight(title=f"{name} VaR appears well-calibrated", severity="low",
                       message=f"Observed violation rate **{obs:.2%}** is close to expected **{exp:.2%}** (diff **{diff:+.2%}**).")

    insights.append(_calib_message(obs_norm, "Normal"))
    insights.append(_calib_message(obs_hist, "Historical"))

    # 5) Distribution diagnostics (skew/kurtosis)
    skew = float(diag.get("skew", 0.0))
    ex_kurt = float(diag.get("excess_kurtosis", 0.0))

    if ex_kurt > 1.0:
        insights.append(Insight(
            title="Fat tails detected (kurtosis)",
            severity="medium",
            message=(
                f"Excess kurtosis ≈ **{ex_kurt:.2f}** suggests heavier tails than a Gaussian distribution. "
                "Parametric (normal) VaR can understate extreme losses in fat-tail regimes."
            ),
        ))
    elif ex_kurt > 0.3:
        insights.append(Insight(
            title="Some tail heaviness (kurtosis)",
            severity="low",
            message=f"Excess kurtosis ≈ **{ex_kurt:.2f}** indicates mild tail heaviness vs normal."
        ))

    if skew < -0.3:
        insights.append(Insight(
            title="Downside skew",
            severity="medium",
            message=f"Skewness ≈ **{skew:.2f}** indicates a heavier downside tail (losses can be sharper than gains)."
        ))
    elif skew > 0.3:
        insights.append(Insight(
            title="Upside skew",
            severity="low",
            message=f"Skewness ≈ **{skew:.2f}** indicates a heavier upside tail."
        ))

    supporting = {
        "weights_normalized": dict(zip(tickers, weights)),
        "hhi": hhi,
        "avg_corr": avg_corr,
        "cvar_var_ratio": cvar_var_ratio,
        "risk_contrib_proxy": contrib.to_dict(),
        "skew": skew,
        "excess_kurtosis": ex_kurt,
    }

    return insights, supporting