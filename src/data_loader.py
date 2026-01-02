from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def _read_jsonl(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, desc=f"Reading {path.name}")):
            if nrows is not None and i >= nrows:
                break
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_yelp_minimal(
    data_dir: Path,
    review_file: str = "review.json",
    user_file: str = "user.json",
    business_file: str = "business.json",
    max_reviews: int = 50_000,
    seed: int = 42,
    keep_text: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        reviews: columns [review_id, user_id, business_id, stars, date, text?]
        users: columns [user_id, name, review_count, average_stars, yelping_since]
        businesses: columns [business_id, name, city, state, stars, review_count, categories]
    """
    data_dir = Path(data_dir)
    review_path = data_dir / review_file
    user_path = data_dir / user_file
    business_path = data_dir / business_file

    if not review_path.exists():
        raise FileNotFoundError(
            f"Missing {review_path}. Please ensure the Yelp dataset is downloaded and placed in the specified data directory."
        )

    # -------- Load reviews --------
    review_all = _read_jsonl(review_path, nrows=None)

    if max_reviews is not None and len(review_all) > max_reviews:
        review_all = review_all.sample(n=max_reviews, random_state=seed).reset_index(drop=True)

    cols = ["review_id", "user_id", "business_id", "stars", "date"]
    if keep_text and "text" in review_all.columns:
        cols.append("text")

    reviews = review_all[cols].copy()
    reviews["date"] = (
        pd.to_datetime(reviews["date"], errors="coerce", utc=True)
        .dt.tz_convert(None)
    )
    reviews = reviews.dropna(subset=["user_id", "business_id", "stars", "date"])

    # -------- Load users --------
    users = pd.DataFrame()
    if user_path.exists():
        users_all = _read_jsonl(user_path, nrows=None)
        user_cols = [c for c in ["user_id", "name", "review_count", "average_stars", "yelping_since"] if c in users_all.columns]
        users = users_all[user_cols].copy()
        if "yelping_since" in users.columns:
            users["yelping_since"] = (
                pd.to_datetime(users["yelping_since"], errors="coerce", utc=True)
                .dt.tz_convert(None)
            )

    # -------- Load businesses --------
    businesses = pd.DataFrame()
    if business_path.exists():
        biz_all = _read_jsonl(business_path, nrows=None)
        biz_cols = [c for c in ["business_id", "name", "city", "state", "stars", "review_count", "categories"] if c in biz_all.columns]
        businesses = biz_all[biz_cols].copy()

    return reviews, users, businesses