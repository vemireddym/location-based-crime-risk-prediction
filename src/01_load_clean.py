import pandas as pd
import numpy as np
import os
import glob

def detect_columns(df):
    date_cols = [c for c in df.columns if any(x in c.lower() for x in ['date', 'time', 'datetime'])]
    lat_cols = [c for c in df.columns if c.lower() in ['lat', 'latitude', 'y']]
    lon_cols = [c for c in df.columns if c.lower() in ['lon', 'longitude', 'long', 'x']]
    type_cols = [c for c in df.columns if any(x in c.lower() for x in ['type', 'primary', 'category', 'crime'])]
    
    return {
        'date': date_cols[0] if date_cols else None,
        'lat': lat_cols[0] if lat_cols else None,
        'lon': lon_cols[0] if lon_cols else None,
        'type': type_cols[0] if type_cols else None
    }

def load_dataset(filepath):
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except:
        df = pd.read_csv(filepath, encoding='latin-1', low_memory=False)
    
    cols = detect_columns(df)
    
    if not cols['date'] or not cols['lat'] or not cols['lon']:
        return None
    
    clean_df = pd.DataFrame()
    clean_df['date'] = pd.to_datetime(df[cols['date']], errors='coerce')
    clean_df['latitude'] = pd.to_numeric(df[cols['lat']], errors='coerce')
    clean_df['longitude'] = pd.to_numeric(df[cols['lon']], errors='coerce')
    
    if cols['type']:
        clean_df['primary_type'] = df[cols['type']]
    
    clean_df = clean_df.dropna(subset=['date', 'latitude', 'longitude'])
    clean_df = clean_df[
        (clean_df['latitude'] >= -90) & (clean_df['latitude'] <= 90) &
        (clean_df['longitude'] >= -180) & (clean_df['longitude'] <= 180)
    ]
    
    return clean_df

def load_and_clean_data(input_paths=None, output_path='data/clean.csv'):
    if input_paths is None:
        input_paths = ['data/chicago_crime.csv', 'data/sf_crime.csv']
    
    all_data = []
    
    for path in input_paths:
        if os.path.exists(path):
            print(f"Loading {path}...")
            df = load_dataset(path)
            if df is not None:
                all_data.append(df)
                print(f"  Loaded {len(df)} records")
        else:
            print(f"  Skipping {path} (not found)")
    
    if not all_data:
        print("Error: No valid datasets found")
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal records: {len(combined)}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return combined

if __name__ == "__main__":
    np.random.seed(42)
    load_and_clean_data()
