import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def load_models(model_dir='outputs'):
    with open(os.path.join(model_dir, 'risk_model.pkl'), 'rb') as f:
        risk_model = pickle.load(f)
    with open(os.path.join(model_dir, 'crime_model.pkl'), 'rb') as f:
        crime_model = pickle.load(f)
    with open(os.path.join(model_dir, 'risk_encoder.pkl'), 'rb') as f:
        risk_encoder = pickle.load(f)
    with open(os.path.join(model_dir, 'crime_encoder.pkl'), 'rb') as f:
        crime_encoder = pickle.load(f)
    return risk_model, crime_model, risk_encoder, crime_encoder

def load_location_encoder():
    features_path = 'data/features.csv'
    if os.path.exists(features_path):
        df = pd.read_csv(features_path, low_memory=False)
        le = LabelEncoder()
        le.fit(df['location'].unique())
        return le
    return None

def get_crime_statistics(location, features_path='data/features.csv'):
    if not os.path.exists(features_path):
        return {}
    
    df = pd.read_csv(features_path, low_memory=False)
    location_data = df[df['location'] == location]
    
    if len(location_data) == 0:
        return {}
    
    crime_counts = location_data['crime_type'].value_counts().to_dict()
    total_crimes = len(location_data)
    
    crime_freq = {}
    for crime_type, count in crime_counts.items():
        if pd.isna(crime_type) or crime_type == 'nan':
            continue
        freq_pct = (count / total_crimes) * 100
        if freq_pct > 30:
            freq_label = 'Very Common'
        elif freq_pct > 15:
            freq_label = 'Common'
        elif freq_pct > 5:
            freq_label = 'Occasional'
        else:
            freq_label = 'Rare'
        
        crime_freq[crime_type] = {
            'count': int(count),
            'percentage': round(freq_pct, 2),
            'frequency': freq_label
        }
    
    return {
        'total_crimes': total_crimes,
        'crime_frequency': crime_freq,
        'most_common': list(crime_counts.keys())[0] if len(crime_counts) > 0 else None
    }

def create_features_from_input(location, day_of_week, hour, month, year):
    location_encoder = load_location_encoder()
    if location_encoder is None:
        raise ValueError("Location encoder not found. Run feature engineering first.")
    
    try:
        location_encoded = location_encoder.transform([location])[0]
    except:
        location_encoded = 0
    
    features = pd.DataFrame({
        'hour': [hour],
        'day_of_week': [day_of_week],
        'month': [month],
        'year': [year],
        'location_encoded': [location_encoded],
        'crime_type_encoded': [0],
        'crime_type_frequency': [0],
        'past_crime_count': [0],
        'crime_count_30d': [0]
    })
    
    return features

def predict_comprehensive(location, day_of_week, hour, month=None, year=None, model_dir='outputs'):
    if month is None or year is None:
        now = datetime.now()
        month = month if month is not None else now.month
        year = year if year is not None else now.year
    
    risk_model, crime_model, risk_encoder, crime_encoder = load_models(model_dir)
    
    features = create_features_from_input(location, day_of_week, hour, month, year)
    
    risk_pred = risk_model.predict(features)[0]
    risk_proba = risk_model.predict_proba(features)[0]
    risk_level = risk_encoder.inverse_transform([risk_pred])[0]
    risk_probabilities = dict(zip(risk_encoder.classes_, risk_proba))
    
    crime_pred = crime_model.predict(features)[0]
    crime_proba = crime_model.predict_proba(features)[0]
    crime_type = crime_encoder.inverse_transform([crime_pred])[0]
    crime_probabilities = dict(zip(crime_encoder.classes_, crime_proba))
    
    stats = get_crime_statistics(location)
    
    result = {
        'risk_level': risk_level,
        'risk_probabilities': risk_probabilities,
        'predicted_crime_type': crime_type,
        'crime_type_probability': float(crime_proba[crime_pred]),
        'crime_type_probabilities': {k: float(v) for k, v in crime_probabilities.items()},
        'location': location,
        'crime_statistics': stats
    }
    
    return result

