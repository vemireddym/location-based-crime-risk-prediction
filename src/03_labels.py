import pandas as pd
import numpy as np
import os

def create_labels(input_path='data/features.csv', output_path='data/model_ready.csv'):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return None
    
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    df['time_window'] = df['date'].dt.floor('24H')
    crime_counts = df.groupby(['location', 'time_window']).size().reset_index(name='crime_count')
    
    df = df.merge(crime_counts, on=['location', 'time_window'], how='left')
    df['crime_count'] = df['crime_count'].fillna(0).astype(int)
    
    low_threshold = np.percentile(df['crime_count'], 70)
    medium_threshold = np.percentile(df['crime_count'], 90)
    
    def assign_risk(count):
        if count <= low_threshold:
            return 'Low'
        elif count <= medium_threshold:
            return 'Medium'
        else:
            return 'High'
    
    df['risk_level'] = df['crime_count'].apply(assign_risk)
    
    most_common_crime = df.groupby('location')['crime_type'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown').reset_index()
    most_common_crime.columns = ['location', 'most_common_crime_type']
    df = df.merge(most_common_crime, on='location', how='left')
    
    model_df = df[[
        'hour', 'day_of_week', 'month', 'year',
        'location_encoded', 'crime_type_encoded', 'crime_type_frequency',
        'past_crime_count', 'crime_count_30d',
        'risk_level', 'crime_type', 'most_common_crime_type'
    ]].copy()
    
    model_df = model_df.dropna(subset=['risk_level', 'location_encoded'])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model_df.to_csv(output_path, index=False)
    print(f"Labels saved to {output_path}")
    print(f"Risk distribution: {model_df['risk_level'].value_counts()}")
    print(f"Crime types: {model_df['crime_type'].nunique()}")
    
    return model_df

if __name__ == "__main__":
    np.random.seed(42)
    create_labels()
