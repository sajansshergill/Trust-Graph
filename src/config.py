from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # path to the data directory
    data_dir: Path = Path("/Users/sajanshergill/Machine-Learning-Projects/Trust-Graph/data/yelp")
    
    #Yelp Dataset Filenames
    review_file: str = "review.json"
    user_file: str = "user.json"
    business_file: str = "business.json"
    
    # Speed knobs
    max_reviews: int = 50_000 #sample size
    seed: int = 42
    
    # Burst score window
    burst_window_dats: int = 14