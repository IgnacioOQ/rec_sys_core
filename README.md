# Recommender System Project

## Overview
This project implements a dual-strategy recommendation engine:
1.  **Collaborative Filtering:** Using the MovieLens (Small) dataset and Scikit-Surprise.
2.  **Contextual Bandits:** Using the Amazon Beauty (5-core) dataset and the ContextualBandits library.

## Project Structure
```
├── data/               # Data files (raw, interim, processed)
├── models/             # Serialized models
├── notebooks/          # Exploration notebooks
├── src/                # Source code
│   ├── data/           # Data ingestion and processing
│   └── models/         # Model training scripts
├── tests/              # Unit and integration tests
├── Makefile            # Automation commands
└── requirements.txt    # Dependencies
```

## Setup

1.  **Environment:** Python 3.8 - 3.10 is recommended (though 3.12 might work with recent library updates).
2.  **Install Dependencies:**
    ```bash
    make setup
    ```

## Usage

### 1. Data Ingestion
Download and process the datasets:
```bash
make data
```
This will download MovieLens and Amazon Beauty datasets and save processed versions to `data/interim/`.

### 2. Model Training
Train both the SVD and Bandit models:
```bash
make train
```
*   Trains SVD on MovieLens and saves to `models/svd_model.pkl`.
*   Trains LinUCB on Amazon Beauty and saves to `models/bandit_policy.pkl`.

### 3. Notebooks
Explore the logic in `notebooks/exploration.ipynb`.

### 4. Testing
Run unit tests:
```bash
make test
```
