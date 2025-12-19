import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

def engineer_features(input_path='data/clean.csv', output_path='data/features.csv'):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return None
    
    df = pd.read_csv(input_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    
    features_df = df[['date', 'location', 'city', 'state', 'crime_type']].copy()
    
    features_df['hour'] = df['hour'] if 'hour' in df.columns else df['date'].dt.hour
    features_df['day_of_week'] = df['date'].dt.dayofweek
    features_df['month'] = df['date'].dt.month
    features_df['year'] = df['date'].dt.year
    
    location_encoder = LabelEncoder()
    features_df['location_encoded'] = location_encoder.fit_transform(features_df['location'].astype(str))
    
    crime_type_encoder = LabelEncoder()
    features_df['crime_type'] = features_df['crime_type'].astype(str)
    features_df['crime_type'] = features_df['crime_type'].replace('nan', 'Unknown')
    features_df['crime_type'] = features_df['crime_type'].fillna('Unknown')
    features_df['crime_type_encoded'] = crime_type_encoder.fit_transform(features_df['crime_type'])
    
    crime_type_counts = features_df.groupby('location')['crime_type'].value_counts().reset_index(name='crime_type_frequency')
    features_df = features_df.merge(
        crime_type_counts,
        on=['location', 'crime_type'],
        how='left'
    )
    features_df['crime_type_frequency'] = features_df['crime_type_frequency'].fillna(0)
    
    features_df = features_df.sort_values('date').reset_index(drop=True)
    features_df['past_crime_count'] = features_df.groupby('location').cumcount()
    
    features_df = features_df.sort_values('date')
    features_df['count_helper'] = 1
    
    result_dfs = []
    for location in features_df['location'].unique():
        group = features_df[features_df['location'] == location].copy()
        group = group.sort_values('date').set_index('date')
        rolling_counts = group['count_helper'].rolling('30D', closed='left').sum().fillna(0)
        group['crime_count_30d'] = rolling_counts.astype(int)
        group = group.reset_index()
        result_dfs.append(group)
    
    features_df = pd.concat(result_dfs, ignore_index=True)
    features_df = features_df.drop(columns=['count_helper'])
    
    final_features = features_df[[
        'hour', 'day_of_week', 'month', 'year',
        'location_encoded', 'crime_type_encoded', 'crime_type_frequency',
        'past_crime_count', 'crime_count_30d',
        'location', 'crime_type', 'date'
    ]].copy()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_features.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")
    
    return final_features

if __name__ == "__main__":
    np.random.seed(42)
    engineer_features()
