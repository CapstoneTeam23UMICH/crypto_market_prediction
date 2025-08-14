# Crypto Market Prediction
**Code for the DRW Crypto Price Prediction Kaggle Competition**

---

## 📌 Overview
This repository contains code developed for the **DRW Crypto Price Prediction** Kaggle competition.

In this competition, the dataset comprises **minute-level historical data** for the crypto market. The challenge is to **predict future crypto market price movements**.

A key challenge: while the training data is time series, the **test data is shuffled and timestamps are masked**, so **regular time series techniques do not directly apply**.

---

## 📦 Dataset Description

### `train.parquet`
The training dataset containing all historical market data along with the corresponding labels:

- **timestamp** — Minute-level timestamp for the record
- **bid_qty** — Total quantity buyers are willing to purchase at the best (highest) bid price
- **ask_qty** — Total quantity sellers are offering to sell at the best (lowest) ask price
- **buy_qty** — Total trading quantity executed at the best ask price
- **sell_qty** — Total trading quantity executed at the best bid price
- **volume** — Total traded volume during the minute
- **X_{1,...,780}** — Anonymized market features derived from proprietary data sources
- **label** — Target variable representing anonymized market price movement

### `test.parquet`
The test dataset has the same feature structure as `train.parquet`, with differences:

- **timestamp** — Masked, shuffled, and replaced with unique IDs
- **label** — All set to `0`

---

## 📂 Project Structure
```
crypto_market_prediction/
├── mlruns/                             # Experiment Logs
├── src/                                # Python modules for modeling, feature selection, evaluation
├── 1_feature_selection.ipynb           # Feature selection workflow
├── 2_model_refinement.ipynb            # Model tuning (LightGBM or MLP/SAE)
├── 3_competition_run.ipynb             # Final model training & prediction
├── 4_time_series_implementation.ipynb  # Time-series splitting demo
├── README.md
└── requirements.txt
```

---

## 📊 Methodology
Each notebook is **independent** and can be run inside a Kaggle Notebook with the competition dataset attached.

General workflow:
1. **Data ingestion** — Load competition data from `/input`
2. **Feature selection** — Correlation filter, Mutual Information, VIF, ARFE
3. **Model training** — LightGBM or MLP/SAE
4. **Evaluation** — RMSE/MSE, Pearson correlation, overfit gap
5. **Prediction** — Save CSV for submission

---

## 🗂 Notebook Guide

### `1_feature_selection.ipynb`
- Contains simple exploratory analysis
- Contains Initial feature selection algorithm

### `2_model_refinement.ipynb`
- Tunes a **single model type** (LightGBM or MLP/SAE)
- Outputs best hyperparameters and fold-wise metrics
- Performs hyperparameter sensitivity analysis

### `3_competition_run.ipynb`
- Trains final single model with best settings
- Generates predictions for test set

### `4_time_series_implementation.ipynb`
- Demonstrates time-series split strategies on training set

---

## 🔧 Requirements
Kaggle includes most dependencies by default. 
Each notebook contains dedicated code to pull repository and install all dependencies

---

## 📜 License
MIT License — free to use with attribution.
