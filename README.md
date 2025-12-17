# Location-Based Crime Risk Prediction System

A machine learning system that predicts crime risk levels (Low/Medium/High) for a given location and time using historical crime data.

## Problem Statement

Predict Risk Level (Low/Medium/High) for a given location (latitude, longitude) and time (day, hour).

## Project Structure

```
├── data/
│   ├── crime.csv          # Raw dataset (download from Kaggle)
│   ├── clean.csv          # Cleaned data
│   ├── features.csv       # Engineered features
│   └── model_ready.csv    # Final dataset with labels
├── src/
│   ├── 01_load_clean.py   # Data loading and cleaning
│   ├── 02_features.py     # Feature engineering
│   ├── 03_labels.py       # Label creation
│   ├── 04_train_eval.py   # Model training and evaluation
│   └── 05_predict.py      # Prediction script
├── outputs/
│   ├── results.txt        # Evaluation metrics
│   └── confusion_matrix.png  # Confusion matrix visualization
├── report/
│   └── report.md          # Project report
└── requirements.txt       # Python dependencies
```

## Setup Instructions

1. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   - Download a crime dataset from Kaggle (CSV format)
   - Save it as `data/crime.csv`
   - Ensure the dataset contains columns: date/time, latitude, longitude, and optionally primary_type

## Usage

### 1. Data Pipeline

Run the scripts in order:

```bash
# Step 1: Load and clean data
python src/01_load_clean.py

# Step 2: Engineer features
python src/02_features.py

# Step 3: Create labels
python src/03_labels.py

# Step 4: Train and evaluate models
python src/04_train_eval.py
```

### 2. Make Predictions

```bash
python src/05_predict.py
```

The script will prompt you for:
- Latitude
- Longitude
- Day of week (0-6, where 0=Monday)
- Hour (0-23)

## Models

- **Main Model**: Random Forest Classifier
- **Comparison Model**: Logistic Regression

## Outputs

- `outputs/results.txt`: Evaluation metrics (Accuracy, F1-score, Confusion Matrix)
- `outputs/confusion_matrix.png`: Visualization of model performance

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib

## Author

CS549 Final Project

