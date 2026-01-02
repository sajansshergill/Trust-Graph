from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """
    Central configuration for the Trust-Graph project.
    Immutable (frozen=True) for reproducibility.
    """

    # Path to Yelp data directory
    data_dir: Path = Path(
        "/Users/sajanshergill/Machine-Learning-Projects/Trust-Graph/data/yelp"
    )

    # Yelp dataset filenames
    review_file: str = "review.json"
    user_file: str = "user.json"
    business_file: str = "business.json"

    # Sampling / reproducibility
    max_reviews: int = 50_000
    seed: int = 42

    # Feature engineering
    burst_window_days: int = 14
