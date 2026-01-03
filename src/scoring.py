# src/scoring.py

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd


def _zscore(s: pd.Series) -> pd.Series:
    """Safe z-score that returns 0s if variance is 0 or series is empty."""
    s = pd.to_numeric(s, errors="coerce").astype(float)
    mu = float(s.mean()) if len(s) else 0.0
    sd = float(s.std(ddof=0)) if len(s) else 0.0
    if sd == 0.0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first column in df that exists from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_business_risk_table(
    burst_df: pd.DataFrame,
    skew_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Build a single business risk table by merging signals and computing a weighted score.

    Accepts flexible column names:
      - burst:   burst_score (required)
      - skew:    rating_skew (required)
      - overlap: frac_heavy_reviewers OR overlap_score OR overlap_ratio OR reviewer_overlap (any one)

    Returns columns:
      business_id, burst_score, rating_skew, overlap_score,
      z_burst, z_skew, z_overlap, risk_score, risk_percentile
    """
    if weights is None:
        weights = {"burst": 0.45, "skew": 0.35, "overlap": 0.20}

    # ---- Validate required columns ----
    if "business_id" not in burst_df.columns or "burst_score" not in burst_df.columns:
        raise ValueError("burst_df must contain columns: ['business_id', 'burst_score']")

    if "business_id" not in skew_df.columns or "rating_skew" not in skew_df.columns:
        raise ValueError("skew_df must contain columns: ['business_id', 'rating_skew']")

    overlap_col = _pick_col(
        overlap_df,
        ["frac_heavy_reviewers", "overlap_score", "overlap_ratio", "reviewer_overlap"],
    )
    if "business_id" not in overlap_df.columns or overlap_col is None:
        raise ValueError(
            "overlap_df must contain 'business_id' and one of: "
            "['frac_heavy_reviewers', 'overlap_score', 'overlap_ratio', 'reviewer_overlap']"
        )

    # ---- Select & normalize columns ----
    b = burst_df[["business_id", "burst_score"]].copy()
    s = skew_df[["business_id", "rating_skew"]].copy()
    o = overlap_df[["business_id", overlap_col]].copy().rename(columns={overlap_col: "overlap_score"})

    # Merge (outer keeps businesses that appear in any signal)
    risk = b.merge(s, on="business_id", how="outer").merge(o, on="business_id", how="outer")
    risk[["burst_score", "rating_skew", "overlap_score"]] = risk[
        ["burst_score", "rating_skew", "overlap_score"]
    ].apply(pd.to_numeric, errors="coerce").fillna(0)

    # ---- Standardize ----
    risk["z_burst"] = _zscore(risk["burst_score"])
    risk["z_skew"] = _zscore(risk["rating_skew"])
    risk["z_overlap"] = _zscore(risk["overlap_score"])

    # ---- Weighted score ----
    w_b = float(weights.get("burst", 0.0))
    w_s = float(weights.get("skew", 0.0))
    w_o = float(weights.get("overlap", 0.0))

    risk["risk_score"] = w_b * risk["z_burst"] + w_s * risk["z_skew"] + w_o * risk["z_overlap"]

    # Percentile (0..1), higher = riskier
    risk["risk_percentile"] = risk["risk_score"].rank(pct=True)

    # Sort with most suspicious first
    risk = risk.sort_values("risk_score", ascending=False).reset_index(drop=True)

    return risk


def build_reviewer_risk_table(
    burst_df: pd.DataFrame,
    rate_df: pd.DataFrame,
    diversity_df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Build a reviewer risk table by merging reviewer signals and computing a weighted score.

    Required:
      - burst_df: user_id, burst_score
      - rate_df: user_id, pct_5star
      - diversity_df: user_id, n_unique_businesses

    Returns columns:
      user_id, burst_score, pct_5star, n_unique_businesses,
      z_burst, z_pos, z_div, risk_score, risk_percentile
    """
    if weights is None:
        weights = {"burst": 0.40, "pos": 0.25, "div": 0.35}

    # ---- Validate ----
    if "user_id" not in burst_df.columns or "burst_score" not in burst_df.columns:
        raise ValueError("burst_df must contain columns: ['user_id', 'burst_score']")

    if "user_id" not in rate_df.columns or "pct_5star" not in rate_df.columns:
        raise ValueError("rate_df must contain columns: ['user_id', 'pct_5star']")

    if "user_id" not in diversity_df.columns or "n_unique_businesses" not in diversity_df.columns:
        raise ValueError("diversity_df must contain columns: ['user_id', 'n_unique_businesses']")

    b = burst_df[["user_id", "burst_score"]].copy()
    r = rate_df[["user_id", "pct_5star"]].copy()
    d = diversity_df[["user_id", "n_unique_businesses"]].copy()

    risk = b.merge(r, on="user_id", how="outer").merge(d, on="user_id", how="outer")
    risk[["burst_score", "pct_5star", "n_unique_businesses"]] = risk[
        ["burst_score", "pct_5star", "n_unique_businesses"]
    ].apply(pd.to_numeric, errors="coerce").fillna(0)

    risk["z_burst"] = _zscore(risk["burst_score"])
    risk["z_pos"] = _zscore(risk["pct_5star"])
    risk["z_div"] = _zscore(risk["n_unique_businesses"])

    w_b = float(weights.get("burst", 0.0))
    w_p = float(weights.get("pos", 0.0))
    w_d = float(weights.get("div", 0.0))

    risk["risk_score"] = w_b * risk["z_burst"] + w_p * risk["z_pos"] + w_d * risk["z_div"]
    risk["risk_percentile"] = risk["risk_score"].rank(pct=True)

    return risk.sort_values("risk_score", ascending=False).reset_index(drop=True)
