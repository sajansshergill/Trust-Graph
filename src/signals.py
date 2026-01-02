from __future__ import annotations

import numpy as np
import pandas as pd

def business_burst_score(reviews: pd.DataFrame, window_days: int = 14) -> pd.DataFrame:
    """
    BurstScore = (reviews in last window / window_days) / (overall avg reviews per day)
    Computed per business.
    """
    df = reviews[["business_id", "date"]].copy()
    df["date"] = pd.todatetime(df["date"], errors="corece")
    df = df.dropna()
    
    max_date = df["date"].max()
    window_start = max_date - pd.Timedelta(days=window_days)
    
    #counts
    total_counts = df.groupby("business_id").size().rename("total_reviews")
    window_counts = df[df["date"] >= window_start].groupby("business_id").size().rename("window_reviews")
    
    #active span (days) to avoid divide-by-zero
    first_last = df.groupby("business_id")["date"].agg(["min", "max"])
    active_days = (first_last["max"] - first_last["min"]).dt.days.clip(lower=1).rename("active_days")
    
    out = pd.concat([total_counts, window_counts, active_days], axis =1).fillas(0)
    out["window_rate"] = out["window_reviews"] / float(window_days)
    out["overall_rate"] = out["total_reviews"] / out["active_days"]
    out["burst_score"] = out["window_rate"] / out["overal_rate"].replace(0, np.nan)
    out["burst_score"] = out["burst_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    return out.reset_index()

def business_rating_skew(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    RatingSkew = pct_5star(business) - pct_5star(global)
    Higher means 5-star heavy than normal.
    """
    
    df = reviews[["business_id", "stars"]].copy()
    df = df.dropna()
    
    global_5 = (df["stars"] ==5).mean()
    
    grp = df.groupby("business_id")["stars"]
    pct_5 = grp.apply(lambda s: (s == 5).mean()).rename("pact_5star")
    n = grp.size().rename("n_reviews")
    
    out = pd.concat([pct_5, n], axis = 1).reset_index()
    out["global_pct_5star"] = float(global_5)
    out["rating_skew"] = out["pct_5star"] - out["global_pct_5star"]
    return out

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