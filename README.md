# Trust-Graph

## Graph-Based Fake Review &amp; Fraud Ring Detection for a Marketplace

# 🔍 Problem Statement

Online marketplaces rely heavily on **user-generated reviews** to build trust and drive conversion. However **coordinated fake review behavior** - including review rings, burst activity,
and copy-pasted content - can artificially inflate reputation, mislead users, and harm long-term platform trust.
This project explores how **graph-based integrity modelling**, inspired by **ads click-fraud detection**, can be applied to identify **suspicious reviewer and listing networks** in a marketplace
setting similar to Airbnb.

# 🎯 Goals
- Detect **coordinated review behavior** using graph analytics
- Assign **interpretable risk scores** to reviewers and listings
- Demonstrate how **content integrity decisions** impact trust and revenue
- Show clear alignment with **Trust & Safety / Ads Integrity** problems

# 🧠 Key Integrity Questions
- Are certain reviwers **connected across mant unrelates listings**?
- Do some listings receive **unusually dense, fast, high-rating review bursts**?
- Can graph structure reveal **hidden fraud rings** not visible in location?
- How do integrity enforcement thresholds affect f**alse positives vs trust gain**?

# 🏗️ System Overview
## Pipeline:
Raw Reviews Data
      ↓
Graph Construction (Users ↔ Listings)
      ↓
Integrity Signal Engineering
      ↓
Graph Pattern Detection
      ↓
Risk Scoring
      ↓
Trust & Business Impact Analysis

# 🧩 Data
## Dataset Used:
- Yelp Open Dataset (used as an Airbnb-style marketplace analog)

## Core Enitities:
- Reviewer
- Business / Listing

## Edges:
- Reviwer -> Listing(review event)

## Each edge includes:
- Rating
- Timestamp

>> Note: The approach generalizes directly to Airbnb review + listing data.

# 🕸️ Graph Construction
## Node Types:
- Reviewer nodes
- Listing nodes

## Derived Graphs
- **Reviewer - Reviwer Graph**
Connedted if tow reviewers reviewed the same listing
- **Listed-Listing Graph:**
Connected if two listings share multiple reviewers

# 🚨 Fraud & Integrity Signals
The system focuses on **interpretable signals** commonly used in production integrity systems.

## 1️⃣ Review Burst Score
- Detect unusually fast spikes in reviwes
- Compares short-window activity to historical baseline

## 2️⃣ Reviewer Overlap Ratio
- Measures how often the same reviwers appear across multiple listings.

## 3️⃣ Rating Skew
- Percentage of 5-star reviews vs platform average
- Flags unnatural positivity patterns

## 4️⃣ Network Density
- Dense subgraphs of reviewers + listings
- Indicates coordinated behavior or review rings

# ⚖️ Risk Scoring
Each listing and reviewer is assigned a Fraud Risk Score using a weighted combinagtion of integrity signals:
RiskScore =
0.35 × BurstScore +
0.30 × ReviewerOverlap +
0.20 × RatingSkew +
0.15 × NetworkDensity

The goal is ranking for investigation, not binary classification.

# 📊 Example Outputs
- Top 10 highest-risk listings
- Top 10 most suspicious reviewers
- Identified review cluster with unusually dense connections
- Visual graphs highlighting potential fraud rings

# 💼 Trust & Business Impact
Fake reviews can:
- Aritifically boost booking probability
- Crowd out honest listings
- Erode long-term user trust

This project simulates:
- How different risk thresholds affect **precsion vs recall**
- The tradeoff between **removing fake reviews** and **avoiding harm to legitimate users**
- Why integrity systems must balance **revenue protection** with **fair enforcement**

# 🔗 Ads Fraud Analogy
This system mirrors **ads integrity modeling:**

| Ads Integrity    | Marketplace Integrity |
| ---------------- | --------------------- |
| Click fraud      | Fake reviews          |
| Bot networks     | Reviewer rings        |
| Advertiser abuse | Host/listing abuse    |
| Traffic bursts   | Review bursts         |

Graph-based detection is effective in both domains.

# 🛠️ Tech Stack
- Python
- NetworkX
- Pandas / Numpy
- Matplotlib / Seaborn (visualization)
- Jupyter notebook

# 🚀 How to Run
git clone https://github.com/yourusername/trustgraph

cd trustgraph

pip install -r requirements.txt

jupyter notebook

--

Run notebooks in order:

01_data_loading.ipynb

02_graph_construction.ipynb

03_integrity_signals.ipynb

04_risk_scoring.ipynb

# 📌 Key Takeaways
- Graph structure reveals fraud patterns invisible at the individual level
- Interpretable signs are critical for **trust & safety decisions**
- Integrity enforcement is a product decision, not just a modeling task
- The same ideas power both **marketplace trust** and **ads fraud defense**

🔮 Future Improvements
- Weakly-supervised ML risk model
- NLP similarity using review embeddings
- Online detection for near-real-time abuse
- Human-in-the-loop review workflows

