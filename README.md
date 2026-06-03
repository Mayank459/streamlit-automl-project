# Advanced Data Analysis & AutoML Dashboard

A Streamlit application for end-to-end data analysis, visualization, and machine learning — from raw data upload to trained model download.

## Overview

This tool is designed to streamline the data science workflow without writing code. It combines exploratory data analysis (EDA), traditional supervised learning, and automated machine learning (AutoML) via [mljar-supervised](https://github.com/mljar/mljar-supervised) into a single interactive dashboard.

## Features

**Exploratory Data Analysis**
- Automatic data type detection and summary statistics
- Missing value detection and handling
- Interactive visualizations: histograms, box plots, scatter plots, correlation heatmaps
- Grouping and aggregation with customizable aggregation functions

**Traditional Machine Learning**
- Regression: Linear Regression, Random Forest Regressor (with GridSearchCV hyperparameter tuning)
- Classification: Random Forest Classifier (with GridSearchCV)
- K-Means Clustering with elbow method for optimal K selection
- Feature importance charts

**AutoML (mljar-supervised)**
- Supports `Explain`, `Perform`, `Compete`, and `Optuna` training modes
- Configurable algorithm selection: LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees, Neural Network, Linear
- Automatic validation strategy selection based on mode
- Leaderboard display with per-model metrics

**Predictions**
- Manual input form with type-aware widgets (numeric inputs for numerical features, dropdowns for categorical)
- Batch predictions from uploaded CSV/Excel files
- AutoML leaderboard model selection for individual model inference
- Upload and use previously saved `.pkl` models

**Model Export**
- Download AutoML results as a ZIP archive (reloadable via `AutoML(results_path=...)`)
- Download scikit-learn models as `.pkl` files via joblib

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`

Key packages:
| Package | Purpose |
|---|---|
| `streamlit` | Web application framework |
| `mljar-supervised` | AutoML engine |
| `scikit-learn` | Traditional ML models |
| `pandas` | Data manipulation |
| `plotly` | Interactive visualizations |
| `numpy` | Numerical computing |
| `joblib` | Model serialization |
| `openpyxl` | Excel file support |

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/Mayank459/streamlit-automl-project.git
cd streamlit-automl-project
```

**2. Create and activate a virtual environment (recommended)**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the application**
```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

This repository is configured for direct deployment on [Streamlit Community Cloud](https://share.streamlit.io).

1. Fork or connect this repository to your Streamlit Cloud account.
2. Set the main file path to `app.py`.
3. Deploy — no additional configuration is required.

> **Note:** `mljar-supervised` includes heavy optional dependencies (XGBoost, CatBoost, LightGBM). On Streamlit Cloud's free tier (1 GB RAM), use `Explain` or `Perform` mode for best stability.

## Supported Input Formats

| Format | Extension |
|---|---|
| Comma-separated values | `.csv` |
| Excel workbook | `.xlsx` |

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules (venv, automl output dirs, caches)
└── README.md           # Project documentation
```

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

## License

This project is open source. See the repository for license details.
