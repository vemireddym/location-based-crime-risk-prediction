"""
Prediction Script
Loads trained model and makes predictions for user-provided location and time inputs.
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from datetime import datetime

# Add parent directory to path for imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_model_and_preprocessors(model_type='random_forest', model_dir='outputs'):
    """
    Load trained model and preprocessing components.
    
    Args:
        model_type: 'random_forest' or 'logistic_regression'
        model_dir: Directory containing saved models
    
    Returns:
        model, scaler (if needed), label_encoder
    """
    model_path = os.path.join(model_dir, f'{model_type}_model.pkl')
    encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Please run 04_train_eval.py first.")
    
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder not found: {encoder_path}. Please run 04_train_eval.py first.")
    
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load label encoder
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Load scaler if using logistic regression
    scaler = None
    if model_type == 'logistic_regression':
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}. Please run 04_train_eval.py first.")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    return model, scaler, label_encoder

def create_features_from_input(latitude, longitude, day_of_week, hour, month=None, year=None):
    """
    Create feature vector from user input.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        day_of_week: Day of week (0=Monday, 6=Sunday)
        hour: Hour of day (0-23)
        month: Month (1-12), optional (defaults to current month)
        year: Year, optional (defaults to current year)
    
    Returns:
        DataFrame with features
    """
    # Use current date if not provided
    if month is None or year is None:
        now = datetime.now()
        month = month if month is not None else now.month
        year = year if year is not None else now.year
    
    # Create grid cell (same logic as in feature engineering)
    grid_precision = 2
    grid_lat = round(latitude, grid_precision)
    grid_lon = round(longitude, grid_precision)
    
    # For past_crime_count and crime_count_30d, we'll set to 0
    # In a real system, these would be calculated from historical data
    # For prediction purposes, we'll use default values
    past_crime_count = 0
    crime_count_30d = 0
    
    # Create feature vector
    features = pd.DataFrame({
        'hour': [hour],
        'day_of_week': [day_of_week],
        'month': [month],
        'year': [year],
        'grid_lat': [grid_lat],
        'grid_lon': [grid_lon],
        'past_crime_count': [past_crime_count],
        'crime_count_30d': [crime_count_30d]
    })
    
    return features

def predict_risk_level(latitude, longitude, day_of_week, hour, 
                       month=None, year=None, model_type='random_forest', model_dir='outputs'):
    """
    Predict crime risk level for given location and time.
    
    Args:
        latitude: Latitude coordinate
        longitude: Longitude coordinate
        day_of_week: Day of week (0=Monday, 6=Sunday)
        hour: Hour of day (0-23)
        month: Month (1-12), optional
        year: Year, optional
        model_type: 'random_forest' or 'logistic_regression'
        model_dir: Directory containing saved models
    
    Returns:
        predicted_risk_level, probabilities_dict
    """
    # Validate inputs
    if not (-90 <= latitude <= 90):
        raise ValueError(f"Latitude must be between -90 and 90, got {latitude}")
    if not (-180 <= longitude <= 180):
        raise ValueError(f"Longitude must be between -180 and 180, got {longitude}")
    if not (0 <= day_of_week <= 6):
        raise ValueError(f"Day of week must be between 0 (Monday) and 6 (Sunday), got {day_of_week}")
    if not (0 <= hour <= 23):
        raise ValueError(f"Hour must be between 0 and 23, got {hour}")
    
    # Load model and preprocessors
    model, scaler, label_encoder = load_model_and_preprocessors(model_type, model_dir)
    
    # Create features
    features = create_features_from_input(latitude, longitude, day_of_week, hour, month, year)
    
    # Preprocess if needed
    if scaler is not None:
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)[0]
    else:
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)[0]
    
    # Decode prediction
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    
    # Create probabilities dictionary
    class_names = label_encoder.classes_
    probabilities_dict = dict(zip(class_names, probabilities))
    
    return predicted_class, probabilities_dict

def interactive_predict():
    """
    Interactive command-line interface for making predictions.
    """
    print("=" * 60)
    print("Crime Risk Prediction System")
    print("=" * 60)
    print("\nEnter the following information to predict crime risk level:\n")
    
    try:
        # Get user input
        latitude = float(input("Latitude (-90 to 90): "))
        longitude = float(input("Longitude (-180 to 180): "))
        
        print("\nDay of week:")
        print("  0 = Monday, 1 = Tuesday, 2 = Wednesday, 3 = Thursday")
        print("  4 = Friday, 5 = Saturday, 6 = Sunday")
        day_of_week = int(input("Day of week (0-6): "))
        
        hour = int(input("Hour of day (0-23): "))
        
        # Optional: month and year
        use_current_date = input("\nUse current date for month/year? (y/n, default=y): ").strip().lower()
        if use_current_date == 'n':
            month = int(input("Month (1-12): "))
            year = int(input("Year (e.g., 2024): "))
        else:
            month = None
            year = None
        
        # Model selection
        print("\nModel selection:")
        print("  1 = Random Forest (recommended)")
        print("  2 = Logistic Regression")
        model_choice = input("Select model (1 or 2, default=1): ").strip()
        model_type = 'logistic_regression' if model_choice == '2' else 'random_forest'
        
        # Make prediction
        print("\n" + "=" * 60)
        print("Making prediction...")
        print("=" * 60)
        
        predicted_risk, probabilities = predict_risk_level(
            latitude, longitude, day_of_week, hour, month, year, model_type
        )
        
        # Display results
        print(f"\n📍 Location: ({latitude:.4f}, {longitude:.4f})")
        print(f"🕐 Time: Day {day_of_week}, Hour {hour}")
        if month and year:
            print(f"📅 Date: {month}/{year}")
        
        print(f"\n🎯 Predicted Risk Level: {predicted_risk.upper()}")
        
        print(f"\n📊 Probability Distribution:")
        for risk_level, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            bar_length = int(prob * 40)
            bar = '█' * bar_length
            print(f"  {risk_level:8s}: {prob*100:5.2f}% {bar}")
        
        print("\n" + "=" * 60)
        
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        print("Please enter valid numeric values.")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please run 04_train_eval.py first to train the models.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

def batch_predict(input_file, output_file, model_type='random_forest', model_dir='outputs'):
    """
    Make predictions for multiple locations from a CSV file.
    
    Args:
        input_file: CSV file with columns: latitude, longitude, day_of_week, hour
        output_file: Output CSV file with predictions
        model_type: 'random_forest' or 'logistic_regression'
        model_dir: Directory containing saved models
    """
    print(f"Loading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    required_cols = ['latitude', 'longitude', 'day_of_week', 'hour']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input file must contain columns: {required_cols}")
    
    print(f"Making predictions for {len(df)} locations...")
    
    predictions = []
    probabilities_list = []
    
    for idx, row in df.iterrows():
        try:
            pred, probs = predict_risk_level(
                row['latitude'],
                row['longitude'],
                row['day_of_week'],
                row['hour'],
                row.get('month'),
                row.get('year'),
                model_type,
                model_dir
            )
            predictions.append(pred)
            probabilities_list.append(probs)
        except Exception as e:
            print(f"Error predicting row {idx}: {e}")
            predictions.append(None)
            probabilities_list.append({})
    
    # Add predictions to dataframe
    df['predicted_risk_level'] = predictions
    for risk_level in ['Low', 'Medium', 'High']:
        df[f'prob_{risk_level.lower()}'] = [p.get(risk_level, 0) for p in probabilities_list]
    
    # Save results
    df.to_csv(output_file, index=False)
    print(f"✓ Predictions saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict crime risk level for a location and time')
    parser.add_argument('--latitude', type=float, help='Latitude coordinate')
    parser.add_argument('--longitude', type=float, help='Longitude coordinate')
    parser.add_argument('--day', type=int, help='Day of week (0=Monday, 6=Sunday)')
    parser.add_argument('--hour', type=int, help='Hour of day (0-23)')
    parser.add_argument('--month', type=int, help='Month (1-12), optional')
    parser.add_argument('--year', type=int, help='Year, optional')
    parser.add_argument('--model', type=str, choices=['random_forest', 'logistic_regression'],
                       default='random_forest', help='Model to use for prediction')
    parser.add_argument('--batch', type=str, help='CSV file for batch predictions')
    parser.add_argument('--output', type=str, help='Output file for batch predictions')
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch prediction mode
        if not args.output:
            args.output = args.batch.replace('.csv', '_predictions.csv')
        batch_predict(args.batch, args.output, args.model)
    elif args.latitude and args.longitude and args.day is not None and args.hour is not None:
        # Command-line prediction mode
        try:
            predicted_risk, probabilities = predict_risk_level(
                args.latitude, args.longitude, args.day, args.hour,
                args.month, args.year, args.model
            )
            print(f"Predicted Risk Level: {predicted_risk}")
            print(f"Probabilities: {probabilities}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Interactive mode
        interactive_predict()

