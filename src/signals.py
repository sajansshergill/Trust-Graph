from __future__ import annotations

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd


def business_burst_score(
    reviews: pd.DataFrame,
    window_days: int = 14
) -> pd.DataFrame:
    """
    Burst score per business:
    (recent review rate) / (overall historical review rate)

    High values indicate abnormal recent activity.
    """
    df = reviews[["business_id", "date"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["business_id", "date"])

    max_date = df["date"].max()
    window_start = max_date - pd.Timedelta(days=window_days)

    # Counts
    total_counts = (
        df.groupby("business_id")
        .size()
        .rename("total_reviews")
    )

    window_counts = (
        df[df["date"] >= window_start]
        .groupby("business_id")
        .size()
        .rename("window_reviews")
    )

    # Active span (days)
    first_last = df.groupby("business_id")["date"].agg(["min", "max"])
    active_days = (
        (first_last["max"] - first_last["min"])
        .dt.days
        .clip(lower=1)
        .rename("active_days")
    )

    out = (
        pd.concat([total_counts, window_counts, active_days], axis=1)
        .fillna(0)
        .reset_index()
    )

    out["window_rate"] = out["window_reviews"] / float(window_days)
    out["overall_rate"] = out["total_reviews"] / out["active_days"]

    out["burst_score"] = (
        out["window_rate"] /
        out["overall_rate"].replace(0, np.nan)
    ).fillna(0)

    return out.sort_values("burst_score", ascending=False)
    

def business_rating_skew(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    For each business:
      pct_5star = fraction of reviews that are 5-star
      global_pct_5star = overall fraction of 5-star reviews
      rating_skew = pct_5star - global_pct_5star
    """
    df = reviews[["business_id", "stars"]].copy()
    df = df.dropna(subset=["business_id", "stars"])

    # Ensure stars is numeric
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars"])

    # Per-business fraction of 5-star reviews
    pct_5 = df.groupby("business_id")["stars"].apply(lambda s: (s == 5).mean())
    pct_5 = pct_5.rename("pct_5star")

    # Per-business review count
    n = df.groupby("business_id").size().rename("n_reviews")

    # Global fraction of 5-star reviews
    global_5 = float((df["stars"] == 5).mean())

    out = pd.concat([pct_5, n], axis=1).reset_index()
    out["global_pct_5star"] = global_5
    out["rating_skew"] = out["pct_5star"] - out["global_pct_5star"]

    return out.sort_values("rating_skew", ascending=False)

def business_reviewer_overlap(reviews: pd.DataFrame, heavy_reviewer_threshold: int = 10) -> pd.DataFrame:
    """
    ReviewerOverlap:
      - Identify "heavy reviewers" = users who reviewed >= threshold businesses in sample
      - For each business: fraction of reviewers who are heavy reviewers
    """
    df = reviews[["user_id", "business_id"]].dropna()

    user_unique_businesses = df.groupby("user_id")["business_id"].nunique()
    heavy_users = set(user_unique_businesses[user_unique_businesses >= heavy_reviewer_threshold].index)

    biz_reviewers = df.groupby("business_id")["user_id"].apply(set)
    frac_heavy = biz_reviewers.apply(lambda users: 0.0 if len(users) == 0 else (sum(u in heavy_users for u in users) / len(users)))
    frac_heavy = frac_heavy.rename("frac_heavy_reviewers")

    n_reviewers = biz_reviewers.apply(len).rename("n_unique_reviewers")
    out = pd.concat([frac_heavy, n_reviewers], axis=1).reset_index()
    return out