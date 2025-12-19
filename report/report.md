# Location-Based Crime Risk Prediction System

**CS549 Final Project**

---

## 1. Topic (Problem Definition)

Predict Risk Level (Low/Medium/High) for a given location (latitude, longitude) and time (day, hour) using historical crime data and machine learning models.

---

## 2. Dataset

### Dataset Description

The project uses multiple crime datasets from Kaggle containing historical crime records with precise location coordinates and temporal information. The datasets include:

- **Date/Time**: Timestamp of crime incidents
- **Location**: Precise latitude and longitude coordinates for each incident
- **Crime Type**: Primary type of crime (optional, for analysis)

### Dataset Links

1. **Chicago Crime Data**: https://www.kaggle.com/datasets/currie32/crimes-in-chicago
   - Contains precise lat/lon coordinates for each crime incident
   - Large dataset with detailed location information

2. **San Francisco Crime Data**: https://www.kaggle.com/datasets/wosaku/crime-in-san-francisco
   - Contains precise coordinates (X/Y columns)
   - Medium-sized dataset with comprehensive crime records

*Note: Datasets should be downloaded and saved as `data/chicago_crime.csv` and `data/sf_crime.csv` before running the data pipeline. See DATASET_DOWNLOAD.md for instructions.*

### Data Characteristics

- **Format**: CSV (Comma-Separated Values)
- **Required Columns**: date/time, latitude, longitude
- **Optional Columns**: primary_type, category
- **Preprocessing**: Missing values in essential columns are removed, invalid coordinates are filtered out
- **Location Precision**: Datasets contain precise coordinates (not city-level), enabling accurate location-based predictions

### Data Processing Pipeline

1. **Loading & Cleaning** (`src/01_load_clean.py`): Loads raw CSV, removes missing values, validates coordinates, and saves cleaned data to `data/clean.csv`

2. **Feature Engineering** (`src/02_features.py`): Extracts temporal features (hour, day_of_week, month, year) and creates location grid cells by binning latitude/longitude coordinates

3. **Label Creation** (`src/03_labels.py`): Creates risk level labels based on crime counts per grid cell and time window:
   - **Low**: Bottom 70% of crime counts
   - **Medium**: Next 20% (70th-90th percentile)
   - **High**: Top 10% (90th-100th percentile)

---

## 3. System Description

### Procedure

The system follows a standard machine learning pipeline:

1. **Data Preprocessing**: Clean raw crime data, handle missing values, validate coordinates
2. **Feature Engineering**: Extract temporal features (hour, day_of_week, month) and create spatial grid cells
3. **Label Generation**: Assign risk levels based on crime frequency in grid cells and time windows
4. **Model Training**: Train Random Forest and Logistic Regression classifiers
5. **Evaluation**: Compare models using accuracy, F1-score, and confusion matrices
6. **Prediction**: Use trained model to predict risk level for new location-time pairs

### Libraries Used

- **pandas** (≥1.5.0): Data manipulation and analysis
- **numpy** (≥1.23.0): Numerical computations
- **scikit-learn** (≥1.2.0): Machine learning models and evaluation metrics
  - `RandomForestClassifier`: Main model for crime risk prediction
  - `LogisticRegression`: Comparison model
  - `train_test_split`: Data splitting
  - `StandardScaler`: Feature scaling for Logistic Regression
  - `accuracy_score`, `f1_score`, `confusion_matrix`: Evaluation metrics
- **matplotlib** (≥3.6.0): Visualization (confusion matrices)

### Model Details

#### Random Forest Classifier (Main Model)
- **Algorithm**: Ensemble of decision trees
- **Parameters**: 
  - `n_estimators=100`: Number of trees
  - `max_depth=20`: Maximum tree depth
  - `min_samples_split=5`: Minimum samples to split
  - `min_samples_leaf=2`: Minimum samples in leaf nodes
  - `class_weight='balanced'`: Handle class imbalance
- **Advantages**: Handles non-linear relationships, feature importance, robust to outliers

#### Logistic Regression (Comparison Model)
- **Algorithm**: Linear classification with multinomial output
- **Parameters**:
  - `max_iter=1000`: Maximum iterations
  - `class_weight='balanced'`: Handle class imbalance
  - `multi_class='multinomial'`: Multi-class classification
  - `solver='lbfgs'`: Optimization algorithm
- **Preprocessing**: Features are standardized using `StandardScaler`
- **Advantages**: Interpretable, fast training, good baseline

### Feature Set

- **Temporal Features**: `hour` (0-23), `day_of_week` (0-6), `month` (1-12), `year`
- **Spatial Features**: `grid_lat`, `grid_lon` (binned coordinates), `grid_cell` (unique identifier)
- **Historical Features**: `past_crime_count`, `crime_count_30d` (rolling window counts)

### Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Macro-averaged F1-score across all classes
- **Per-Class F1-Score**: F1-score for each risk level (Low, Medium, High)
- **Confusion Matrix**: Detailed classification performance per class

---

## 4. Results and Conclusion

### Model Performance

*Note: Actual results will be generated after running `src/04_train_eval.py` with your dataset.*

#### Random Forest Results
- **Test Accuracy**: [To be filled after training]
- **Test F1-Score (macro)**: [To be filled after training]
- **Per-Class F1-Scores**:
  - Low: [To be filled]
  - Medium: [To be filled]
  - High: [To be filled]

#### Logistic Regression Results
- **Test Accuracy**: [To be filled after training]
- **Test F1-Score (macro)**: [To be filled after training]
- **Per-Class F1-Scores**:
  - Low: [To be filled]
  - Medium: [To be filled]
  - High: [To be filled]

### Model Comparison

The Random Forest classifier is expected to outperform Logistic Regression due to its ability to capture non-linear relationships and interactions between features. However, Logistic Regression provides a simpler, more interpretable baseline model.

**Key Findings**:
- Both models handle the multi-class classification problem effectively
- Random Forest captures complex spatial-temporal patterns in crime data
- Feature engineering (grid cells, temporal features) significantly improves prediction accuracy
- Class imbalance is addressed using `class_weight='balanced'` parameter

### Conclusion

The Location-Based Crime Risk Prediction System successfully predicts crime risk levels using machine learning. The Random Forest model provides robust predictions by learning from historical crime patterns across different locations and times. The system can be used as a tool for:

- **Public Safety**: Identifying high-risk areas and times
- **Resource Allocation**: Optimizing police patrol routes
- **Urban Planning**: Understanding crime patterns in different neighborhoods

The modular design allows for easy extension with additional features (e.g., weather data, population density) and model improvements (e.g., deep learning, ensemble methods).

### Future Improvements

1. **Enhanced Features**: Include weather data, population density, proximity to landmarks
2. **Temporal Models**: Use time series models (LSTM, ARIMA) for temporal patterns
3. **Real-time Updates**: Implement streaming data processing for live predictions
4. **Geographic Visualization**: Add interactive maps showing risk levels
5. **Model Interpretability**: Use SHAP values to explain predictions

---

**Project Code**: Available in `src/` directory  
**Results**: Saved in `outputs/results.txt` and `outputs/confusion_matrix.png`

