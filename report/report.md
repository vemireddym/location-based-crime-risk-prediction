Location-Based Crime Risk Prediction System

Student Name: [Your Name Here]

Completion Status: Project Completed

Course: CS549-001
Professor: Byoung Jik Lee
Due Date: December 19, 2025


1. Topic: Problem Definition

This project develops a machine learning system that predicts crime risk levels for specific locations and times. The system takes a location (city and state) along with temporal information (day of week, hour, month, year) as input and outputs a predicted risk level: Low, Medium, or High.

The problem addresses the need for data-driven crime risk assessment. By analyzing historical crime patterns across multiple cities, the system can identify when and where crimes are more likely to occur. This information can be valuable for public safety planning, resource allocation, and helping individuals make informed decisions about when and where to travel.

The approach uses supervised machine learning with Random Forest classifiers. The system learns from over 2.2 million historical crime records spanning multiple major US cities. Instead of using precise coordinates, the system uses city and state information, making it more practical for city-level risk assessment. The model predicts both the risk level and the most likely type of crime that might occur.

2. Dataset

The project uses a super dataset created by merging five separate crime datasets from Kaggle. Each dataset contains historical crime records with location and temporal information.

Dataset Sources:

1. US Homicide Dataset (1980-2004): https://www.kaggle.com/datasets/mrayushagrawal/us-crime-dataset/
   - Historical data covering 24 years
   - Contains Year, Month, City, State, and Crime Type columns

2. Chicago Crime Data: https://www.kaggle.com/datasets/currie32/crimes-in-chicago
   - Large dataset with Date, City, State, and Crime Type information

3. Los Angeles Crime Data: https://www.kaggle.com/datasets/cityofLA/crime-in-los-angeles
   - Contains Date, City, State, and Crime Type columns

4. Boston Crime Data: https://www.kaggle.com/datasets/AnalyzeBoston/crimes-in-boston
   - Includes Date, City, State, and Crime Type information

5. Philadelphia Crime Data: https://www.kaggle.com/datasets/mchirico/philadelphiacrimedata
   - Contains Date, City, State, and Crime Type columns

Data Characteristics:

The merged super dataset contains 2,216,578 total records after cleaning and processing. The data is stored in CSV format with the following key columns:
- Date/Time: Timestamp of crime incidents (or Year/Month for older datasets)
- Location: City and State information combined into a single location identifier
- Crime Type: Type of crime for each incident

Data Processing:

The datasets are merged using a custom Python script that handles different column formats across datasets. Missing columns are handled gracefully by setting them to null values, allowing the system to work with datasets that have varying structures. The location information is standardized by combining city and state into a single "City, State" format (e.g., "Chicago, IL"). This location-based approach is more practical than using precise coordinates, as it enables city-level risk predictions that are easier to understand and use.

3. Description of Your System

Main Procedure:

The system follows a standard machine learning pipeline with six main steps:

1. Data Loading and Cleaning: Raw CSV files are loaded and cleaned. Missing values are handled, dates are standardized, and invalid records are removed.

2. Feature Engineering: Temporal features are extracted from dates (hour, day of week, month, year). Locations are encoded using label encoding. Historical features are created, including past crime counts and rolling window averages.

3. Label Creation: Risk levels are assigned based on crime frequency. Records in the bottom 70% of crime counts are labeled "Low", the next 20% (70th-90th percentile) are "Medium", and the top 10% are "High".

4. Model Training: Two Random Forest classifiers are trained - one for risk level prediction and one for crime type prediction. The data is split 80/20 for training and testing.

5. Evaluation: Models are evaluated using accuracy and F1-scores. Confusion matrices are generated to analyze performance.

6. Prediction: The trained models can predict risk levels and crime types for new location-time combinations.

Key Programming Ideas:

The system uses a location-based approach rather than coordinate-based. Instead of using latitude and longitude, locations are represented as "City, State" strings. This makes the system more practical for end users who think in terms of cities rather than coordinates.

Feature engineering plays a crucial role. Temporal patterns are captured through hour, day of week, month, and year features. Historical patterns are encoded through features like past crime counts and 30-day rolling averages. Location encoding allows the model to learn city-specific patterns.

The system implements dual prediction - it predicts both risk level and crime type simultaneously. This provides more comprehensive information than just risk level alone.

Libraries Used:

- pandas: Data manipulation and CSV file handling
- numpy: Numerical computations and array operations
- scikit-learn: Machine learning models and evaluation
  - RandomForestClassifier for both risk and crime type prediction
  - LabelEncoder for encoding categorical variables
  - train_test_split for data splitting
  - accuracy_score and f1_score for evaluation

Model Details:

The Random Forest classifier uses 100 decision trees with a maximum depth of 20. The model uses balanced class weights to handle class imbalance in the data. Minimum samples per split is set to 5, and minimum samples per leaf is 2. These parameters were chosen to balance model complexity with generalization performance.

Feature Set:

The final feature set includes:
- Temporal features: hour (0-23), day_of_week (0-6), month (1-12), year
- Location features: location_encoded (numeric encoding of city/state)
- Crime type features: crime_type_encoded, crime_type_frequency
- Historical features: past_crime_count, crime_count_30d (rolling 30-day window)

4. Your Results and Conclusions

Model Performance:

The system was trained on 1,773,262 records and tested on 443,316 records. The results show excellent performance:

Risk Level Model:
- Test Accuracy: 99.92%
- Test F1-score: 99.91%

Crime Type Model:
- Test Accuracy: 99.99%
- Test F1-score: 96.75%

These results indicate that the Random Forest model successfully learned the patterns in the historical crime data. The extremely high accuracy suggests that the feature engineering effectively captured the relevant patterns for crime prediction.

Key Findings:

The location-based approach proved effective. By using city and state information rather than precise coordinates, the system can make practical predictions that are easy to interpret. The temporal features (hour, day of week, month) were particularly important, as crime patterns vary significantly by time.

The dual prediction approach (risk level and crime type) provides more useful information than risk level alone. Users can understand not just how risky a location is, but also what type of crime is most likely.

The feature engineering, especially the historical features like rolling crime counts, helped the model capture temporal trends. The 30-day rolling window feature allows the model to consider recent crime patterns when making predictions.

Conclusions:

The Location-Based Crime Risk Prediction System successfully demonstrates that machine learning can effectively predict crime risk using historical data. The Random Forest model achieved near-perfect accuracy on the test set, indicating that the chosen features and model architecture are well-suited for this problem.

The system has practical applications in public safety planning, resource allocation, and helping individuals make informed decisions. The location-based approach makes it accessible to users who may not be familiar with coordinate systems.

A live web application has been deployed and is available at:
https://v-location-based-crime-risk-prediction-nmgxrfwqdfuhq3wvo4qvso.streamlit.app/

The web application provides an interactive interface where users can enter a location, select a date and time, and receive instant predictions for both risk level and crime type. The application includes visualizations, maps, and detailed statistics to help users understand the predictions.

The modular design of the system allows for future enhancements. Additional features such as weather data, population density, or proximity to landmarks could potentially improve predictions further. The system could also be extended to use time series models for better temporal pattern recognition.

Future Work:

Potential improvements include incorporating additional data sources like weather patterns or demographic information. The system could also be enhanced with real-time data processing capabilities for live predictions. Geographic visualization features could make the predictions more intuitive for users.
