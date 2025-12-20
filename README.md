# Location-Based Crime Risk Prediction System

A machine learning system that predicts crime risk levels (Low/Medium/High) and crime types for specific locations and times using historical crime data. Features an interactive web application built with Streamlit.

## Live Application

Access the web application at: https://v-location-based-crime-risk-prediction-nmgxrfwqdfuhq3wvo4qvso.streamlit.app/

## Problem Statement

This project develops a machine learning system that predicts crime risk levels for specific locations and times. The system takes a location (city and state) along with temporal information (day of week, hour, month, year) as input and outputs a predicted risk level (Low, Medium, or High) and the most likely type of crime.

## Unique Features

### Multi-City Super Dataset

The system combines 5 different city datasets into one super dataset spanning 1980-2024 (44+ years of data). This includes crime data from Chicago, Los Angeles, Boston, Philadelphia, and historical US homicide data. Most crime prediction systems use single-city datasets, but our approach trains on diverse geographic patterns, making it more generalizable.

### Dual Prediction System

The system predicts both risk level AND crime type simultaneously. While most systems only predict risk, this provides actionable insights about what type of crime is likely to occur. The system outputs:
- Risk Levels: Low, Medium, High
- Crime Types: Most likely crime type (Theft, Assault, Burglary, etc.)

### Crime Frequency Analysis

Beyond simple prediction, the system provides frequency statistics for each crime type per location. This includes:
- How often each crime type occurs
- Frequency labels (Very Common, Common, Occasional, Rare)
- Historical crime counts and percentages

### Comprehensive Web Interface

The project includes an interactive Streamlit web application with multiple visualizations:
- Risk level prediction with probability distributions
- Crime type prediction with confidence scores
- Crime frequency charts and statistics
- Interactive maps with location markers
- Location-based search with geocoding
- Temporal pattern analysis

### Robust Data Handling

The system gracefully handles missing columns across different datasets. It auto-detects column names, sets missing columns to null, continues processing, and reports missing columns at the end. This flexibility allows the system to work with datasets that have varying structures.

## Project Structure

```
├── app.py                    # Streamlit web application
├── setup.sh                  # Setup script for Streamlit Cloud
├── data/
│   ├── crime.csv             # Raw dataset
│   ├── super_dataset.csv     # Merged dataset from all cities
│   ├── clean.csv             # Cleaned data
│   ├── features.csv          # Engineered features
│   └── model_ready.csv       # Final dataset with labels
├── src/
│   ├── 00_create_super_dataset.py  # Merge multiple datasets
│   ├── 01_load_clean.py      # Data loading and cleaning
│   ├── 02_features.py        # Feature engineering
│   ├── 03_labels.py          # Label creation
│   ├── 04_train_eval.py     # Model training and evaluation
│   ├── 05_predict.py         # Command-line prediction script
│   ├── predict.py            # Prediction module (used by app)
│   └── validate_predictions.py  # Validation script
├── outputs/
│   ├── risk_model.pkl        # Risk level prediction model
│   ├── crime_model.pkl       # Crime type prediction model
│   ├── risk_encoder.pkl      # Risk level encoder
│   ├── crime_encoder.pkl     # Crime type encoder
│   ├── confusion_matrix.png  # Confusion matrix visualization
│   └── results.txt           # Evaluation metrics
├── report/
│   └── report.md             # Project report
├── requirements.txt          # Python dependencies
├── DATASET_DOWNLOAD.md       # Dataset download instructions
└── VALIDATION_GUIDE.md       # Prediction validation guide
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Git LFS (for downloading large model files)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/vemireddym/location-based-crime-risk-prediction.git
cd location-based-crime-risk-prediction
```

2. Install Git LFS and download model files:
```bash
# Install Git LFS (if not already installed)
# macOS: brew install git-lfs
# Linux: sudo apt-get install git-lfs
# Windows: Download from https://git-lfs.github.com/

# Initialize Git LFS
git lfs install

# Download large model files
git lfs pull
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Data Pipeline

Important: You must run the training pipeline before using the web app or prediction script. The model files will be generated in the `outputs/` directory.

Run the scripts in order:

```bash
# Step 0: Create super dataset (merge all city datasets)
python src/00_create_super_dataset.py

# Step 1: Load and clean data
python src/01_load_clean.py

# Step 2: Engineer features
python src/02_features.py

# Step 3: Create labels
python src/03_labels.py

# Step 4: Train and evaluate models (REQUIRED for predictions)
python src/04_train_eval.py
```

After running Step 4, you should have these files in the `outputs/` directory:
- risk_model.pkl
- crime_model.pkl
- risk_encoder.pkl
- crime_encoder.pkl
- confusion_matrix.png
- results.txt

## Usage

### Web Application (Recommended)

Run the Streamlit web application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

Features:
- Enter location (city, state) and get automatic geocoding
- Select date and time for prediction
- View risk level prediction with probabilities
- See predicted crime type with confidence
- View crime frequency statistics
- Interactive maps and visualizations

### Command Line Interface

Use the command-line prediction script:

```bash
python src/05_predict.py "Chicago, IL" 0 14 3 2024
```

Arguments:
- Location: City and State in quotes (e.g., "Chicago, IL")
- Day of week: 0-6 (0=Monday, 1=Tuesday, ..., 6=Sunday)
- Hour: 0-23 (24-hour format)
- Month: 1-12 (optional, defaults to current month)
- Year: 4-digit year (optional, defaults to current year)

Example:
```bash
python src/05_predict.py "Los Angeles, CA" 5 20 12 2023
```

This predicts risk for Los Angeles, CA on Saturday at 8 PM in December 2023.

### Python API

Import and use the prediction functions:

```python
from src.predict import predict_comprehensive

result = predict_comprehensive(
    location="Chicago, IL",
    day_of_week=0,  # Monday
    hour=14,        # 2 PM
    month=3,        # March
    year=2024
)

print(f"Risk Level: {result['risk_level']}")
print(f"Predicted Crime Type: {result['predicted_crime_type']}")
print(f"Risk Probabilities: {result['risk_probabilities']}")
```

## Model Performance

The system was trained on 1,773,262 records and tested on 443,316 records:

Risk Level Model:
- Test Accuracy: 99.92%
- Test F1-score: 99.91%

Crime Type Model:
- Test Accuracy: 99.99%
- Test F1-score: 96.75%

## Dataset

The project uses a super dataset created by merging five separate crime datasets from Kaggle:

1. US Homicide Dataset (1980-2004): https://www.kaggle.com/datasets/mrayushagrawal/us-crime-dataset/
2. Chicago Crime Data: https://www.kaggle.com/datasets/currie32/crimes-in-chicago
3. Los Angeles Crime Data: https://www.kaggle.com/datasets/cityofLA/crime-in-los-angeles
4. Boston Crime Data: https://www.kaggle.com/datasets/AnalyzeBoston/crimes-in-boston
5. Philadelphia Crime Data: https://www.kaggle.com/datasets/mchirico/philadelphiacrimedata

The merged dataset contains 2,216,578 total records after cleaning and processing. See DATASET_DOWNLOAD.md for detailed download instructions.

## System Architecture

### Data Processing Pipeline

1. Data Loading and Cleaning: Raw CSV files are loaded, missing values handled, dates standardized
2. Feature Engineering: Temporal features extracted (hour, day of week, month, year), locations encoded, historical features created
3. Label Creation: Risk levels assigned based on crime frequency (Low: bottom 70%, Medium: 70th-90th percentile, High: top 10%)
4. Model Training: Two Random Forest classifiers trained - one for risk level, one for crime type
5. Evaluation: Models evaluated using accuracy and F1-scores
6. Prediction: Trained models predict risk levels and crime types for new location-time combinations

### Key Features

- Location-based approach: Uses "City, State" format rather than coordinates for practical usability
- Temporal patterns: Captures hour, day of week, month, and year patterns
- Historical context: Includes past crime counts and 30-day rolling averages
- Dual prediction: Simultaneously predicts both risk level and crime type

### Libraries Used

- pandas: Data manipulation and CSV file handling
- numpy: Numerical computations
- scikit-learn: Machine learning models (Random Forest Classifier)
- streamlit: Web application framework
- plotly: Interactive visualizations
- folium: Interactive maps
- geopy: Geocoding services

## Deployment

### Streamlit Cloud

The model files are stored using Git LFS. For Streamlit Cloud deployment:

1. Push your code to GitHub (model files are automatically handled via Git LFS)
2. Go to https://share.streamlit.io
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Click Deploy

The app will automatically download model files via Git LFS during deployment.

Note: First deployment may take a few minutes to download large model files. Subsequent deployments will be faster.

## Validation

To verify prediction accuracy, use the validation script:

```bash
python src/validate_predictions.py "Chicago, IL"
```

This script:
- Compares predictions with historical statistics
- Tests consistency across multiple scenarios
- Validates probability sums and prediction alignment
- Checks temporal patterns (night vs day)
- Tests edge cases

See VALIDATION_GUIDE.md for detailed validation methods.

## Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn == 1.6.1
- matplotlib >= 3.6.0
- geopy >= 2.3.0
- streamlit >= 1.28.0
- plotly >= 5.18.0
- folium >= 0.15.0
- streamlit-folium >= 0.15.0

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Comparison with Existing Systems

Most existing crime prediction systems:
- Use single-city datasets
- Only predict risk level (not crime type)
- Use coordinate-based locations
- Lack comprehensive web interfaces
- Don't provide frequency analysis

Our advantages:
- Multi-city training for better generalization
- Location-based (city names) for practical use
- Dual prediction (risk + crime type)
- Frequency insights for context
- Web interface accessible to non-technical users
- Historical span of 44+ years

## Future Enhancements

Potential improvements include:
- Crime pattern recognition and anomaly detection
- Comparative city analysis
- Temporal pattern analysis with seasonal trends
- Crime type clustering (property vs violent crimes)
- Predictive trend forecasting
- Interactive crime explorer with filtering
- Safety recommendations based on predicted crime types

## Author

CS549 Final Project - Western Illinois University

## License

This project is for educational purposes.

Copyright 2025 Vemireddy. All Rights Reserved.
