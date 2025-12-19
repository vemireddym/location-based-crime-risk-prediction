import pandas as pd
import numpy as np
import os

def load_and_clean_data(input_path='data/super_dataset.csv', output_path='data/clean.csv'):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run 00_create_super_dataset.py first.")
        return None
    
    df = pd.read_csv(input_path, low_memory=False)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    df = df.dropna(subset=['date', 'location'])
    df = df[df['location'].notna()]
    
    if 'time' in df.columns:
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df['time'] = df['time'].fillna(df['date'].dt.hour)
    else:
        df['time'] = df['date'].dt.hour
    
    df['hour'] = df['time'].astype(int) % 24
    
    print(f"Cleaned dataset: {len(df)} records")
    print(f"Locations: {df['location'].nunique()}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return df

if __name__ == "__main__":
    np.random.seed(42)
    load_and_clean_data()
