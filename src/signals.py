from __future__ import annotations

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

def business_reviewer_overlap(
    reviews: pd.DataFrame,
    heavy_reviewer_threshold: int = 10
) -> pd.DataFrame:
    """
    Measures how concentrated a business's reviews are among heavy reviewers.

    frac_heavy_reviewers = (# reviews from heavy reviwers) / (total reviews)
    
    Return a richer table (useful for README tables):
    business_id, total_reviews, heavy_reviews, frac_heavy_reviewers
    """
    df = reviews[["business_id", "user_id"]].copy()
    df = df.dropna(subset=["business_id", "user_id"])

    # Identify heavy reviewers globally
    user_counts = df.groupby("user_id").size()
    heavy_users = set(user_counts[user_counts >= heavy_reviewer_threshold].index)

    # Mark heavy reviews
    df["is_heavy"] = df["user_id"].isin(heavy_users)

    total_reviews = (
        df.groupby("business_id")
        .size()
        .rename("total_reviews")
    )

    heavy_reviews = (
        df[df["is_heavy"]]
        .groupby("business_id")
        .size()
        .rename("heavy_reviews")
    )

    out = (
        pd.concat([total_reviews, heavy_reviews], axis=1)
        .fillna(0)
        .reset_index()
    )

    out["frac_heavy_reviwers"] = (out["heavy_reviews"] / out["total_reviews"]).fillna(0)

    return out.sort_values("frac_heavy_reviwers", ascending=False)

# =========================
# Reviewer-level signals
# =========================

def reviewer_burst_score(
    reviews: pd.DataFrame,
    window_days: int = 14
) -> pd.DataFrame:
    """
    Burst score per reviewer:
    (recent review rate) / (overall historical review rate)

    High values indicate abnormal recent activity for that reviewer.
    """
    df = reviews[["user_id", "date"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["user_id", "date"])

    max_date = df["date"].max()
    window_start = max_date - pd.Timedelta(days=window_days)

    total_counts = df.groupby("user_id").size().rename("total_reviews")
    window_counts = df[df["date"] >= window_start].groupby("user_id").size().rename("window_reviews")

    first_last = df.groupby("user_id")["date"].agg(["min", "max"])
    active_days = ((first_last["max"] - first_last["min"]).dt.days.clip(lower=1)).rename("active_days")

    out = (
        pd.concat([total_counts, window_counts, active_days], axis=1)
        .fillna(0)
        .reset_index()
    )

    out["window_rate"] = out["window_reviews"] / float(window_days)
    out["overall_rate"] = out["total_reviews"] / out["active_days"]

    out["burst_score"] = (out["window_rate"] / out["overall_rate"].replace(0, np.nan)).fillna(0)

    return out.sort_values("burst_score", ascending=False)


def reviewer_5star_rate(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    For each reviewer:
      pct_5star = fraction of reviews that are 5-star
      n_reviews = number of reviews
    """
    df = reviews[["user_id", "stars"]].copy()
    df = df.dropna(subset=["user_id", "stars"])
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce")
    df = df.dropna(subset=["stars"])

    pct_5 = df.groupby("user_id")["stars"].apply(lambda s: (s == 5).mean()).rename("pct_5star")
    n = df.groupby("user_id").size().rename("n_reviews")

    out = pd.concat([pct_5, n], axis=1).reset_index()
    return out.sort_values("pct_5star", ascending=False)


def reviewer_diversity(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    For each reviewer:
      n_unique_businesses = number of unique businesses reviewed
    """
    df = reviews[["user_id", "business_id"]].copy()
    df = df.dropna(subset=["user_id", "business_id"])

    out = (
        df.groupby("user_id")["business_id"]
        .nunique()
        .rename("n_unique_businesses")
        .reset_index()
    )

    return out.sort_values("n_unique_businesses", ascending=False)