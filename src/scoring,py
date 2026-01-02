# src/scoring.py

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def build_business_risk_table(
    burst_df: pd.DataFrame,
    skew_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    weights: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Combine trust signals into a single business risk score.
    """

    if weights is None:
        weights = {
            "burst_score": 0.45,
            "rating_skew": 0.35,
            "overlap_score": 0.20,
        }

    # Normalize overlap column name
    overlap = overlap_df.copy()
    if "overlap_score" not in overlap.columns:
        if "overlap_ratio" in overlap.columns:
            overlap = overlap.rename(columns={"overlap_ratio": "overlap_score"})
        elif "reviewer_overlap" in overlap.columns:
            overlap = overlap.rename(columns={"reviewer_overlap": "overlap_score"})
        else:
            raise ValueError(
                "overlap_df must contain one of: overlap_score, overlap_ratio, reviewer_overlap"
            )

    b = burst_df[["business_id", "burst_score"]]
    s = skew_df[["business_id", "rating_skew"]]
    o = overlap[["business_id", "overlap_score"]]

    risk = (
        b.merge(s, on="business_id", how="outer")
         .merge(o, on="business_id", how="outer")
         .fillna(0)
    )

    risk["z_burst"] = _zscore(risk["burst_score"])
    risk["z_skew"] = _zscore(risk["rating_skew"])
    risk["z_overlap"] = _zscore(risk["overlap_score"])

    risk["risk_score"] = (
        weights["burst_score"] * risk["z_burst"]
        + weights["rating_skew"] * risk["z_skew"]
        + weights["overlap_score"] * risk["z_overlap"]
    )

    risk["risk_percentile"] = risk["risk_score"].rank(pct=True)

    return risk.sort_values("risk_score", ascending=False).reset_index(drop=True)
