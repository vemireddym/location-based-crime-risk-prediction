import pandas as pd
import numpy as np
import os

def engineer_features(input_path='data/clean.csv', output_path='data/features.csv', grid_precision=3):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found")
        return None
    
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    
    features_df = df[['date', 'latitude', 'longitude']].copy()
    
    features_df['hour'] = df['date'].dt.hour
    features_df['day_of_week'] = df['date'].dt.dayofweek
    features_df['month'] = df['date'].dt.month
    features_df['year'] = df['date'].dt.year
    
    features_df['grid_lat'] = features_df['latitude'].round(grid_precision)
    features_df['grid_lon'] = features_df['longitude'].round(grid_precision)
    features_df['grid_cell'] = features_df['grid_lat'].astype(str) + '_' + features_df['grid_lon'].astype(str)
    
    features_df = features_df.sort_values('date').reset_index(drop=True)
    features_df['past_crime_count'] = features_df.groupby('grid_cell').cumcount()
    
    features_df = features_df.sort_values('date')
    features_df['count_helper'] = 1
    
    result_dfs = []
    for grid_cell in features_df['grid_cell'].unique():
        group = features_df[features_df['grid_cell'] == grid_cell].copy()
        group = group.sort_values('date').set_index('date')
        rolling_counts = group['count_helper'].rolling('30D', closed='left').sum().fillna(0)
        group['crime_count_30d'] = rolling_counts.astype(int)
        group = group.reset_index()
        result_dfs.append(group)
    
    features_df = pd.concat(result_dfs, ignore_index=True)
    features_df = features_df.drop(columns=['count_helper'])
    
    final_features = features_df[[
        'hour', 'day_of_week', 'month', 'year',
        'grid_lat', 'grid_lon', 'grid_cell',
        'past_crime_count', 'crime_count_30d',
        'latitude', 'longitude', 'date'
    ]].copy()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_features.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")
    
    return final_features

if __name__ == "__main__":
    np.random.seed(42)
    engineer_features()
