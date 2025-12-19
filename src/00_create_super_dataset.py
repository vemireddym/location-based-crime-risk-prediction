import pandas as pd
import numpy as np
import os

def find_column(df, keywords):
    for col in df.columns:
        if any(kw.lower() in col.lower() for kw in keywords):
            return col
    return None

def load_and_standardize_dataset(filepath):
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except:
        try:
            df = pd.read_csv(filepath, encoding='latin-1', low_memory=False)
        except:
            return None, {}
    
    missing_cols = {}
    clean_df = pd.DataFrame()
    
    date_col = find_column(df, ['date', 'datetime', 'occurred'])
    year_col = find_column(df, ['year'])
    month_col = find_column(df, ['month'])
    time_col = find_column(df, ['time', 'hour'])
    city_col = find_column(df, ['city'])
    state_col = find_column(df, ['state'])
    crime_type_col = find_column(df, ['type', 'primary', 'category', 'crime', 'offense'])
    
    if date_col:
        clean_df['date'] = pd.to_datetime(df[date_col], errors='coerce')
    elif year_col and month_col:
        clean_df['date'] = pd.to_datetime(
            df[year_col].astype(str) + '-' + df[month_col].astype(str).str.zfill(2) + '-01',
            errors='coerce'
        )
        if not time_col:
            clean_df['time'] = None
            missing_cols['time'] = True
    else:
        clean_df['date'] = None
        missing_cols['date'] = True
    
    if time_col:
        if df[time_col].dtype == 'object':
            try:
                clean_df['time'] = pd.to_datetime(df[time_col], format='%H:%M:%S', errors='coerce').dt.hour
            except:
                clean_df['time'] = None
                missing_cols['time'] = True
        else:
            clean_df['time'] = pd.to_numeric(df[time_col], errors='coerce')
    else:
        clean_df['time'] = None
        if 'time' not in missing_cols:
            missing_cols['time'] = True
    
    if city_col:
        clean_df['city'] = df[city_col].astype(str).str.strip()
    else:
        clean_df['city'] = None
        missing_cols['city'] = True
    
    if state_col:
        clean_df['state'] = df[state_col].astype(str).str.strip()
    else:
        clean_df['state'] = None
        missing_cols['state'] = True
    
    clean_df['location'] = clean_df['city'].astype(str) + ', ' + clean_df['state'].astype(str)
    clean_df['location'] = clean_df['location'].replace('nan, nan', None)
    
    if crime_type_col:
        clean_df['crime_type'] = df[crime_type_col].astype(str).str.strip()
    else:
        clean_df['crime_type'] = None
        missing_cols['crime_type'] = True
    
    clean_df = clean_df.dropna(subset=['date'])
    clean_df = clean_df[clean_df['location'] != 'nan, nan']
    clean_df = clean_df[clean_df['location'].notna()]
    
    return clean_df, missing_cols

def create_super_dataset(input_paths=None, output_path='data/super_dataset.csv'):
    if input_paths is None:
        input_paths = [
            'data/crime.csv',
            'data/chicago_crime.csv',
            'data/la_crime.csv',
            'data/boston_crime.csv',
            'data/philly_crime.csv'
        ]
    
    all_data = []
    missing_summary = {}
    
    for path in input_paths:
        if os.path.exists(path):
            print(f"Processing {path}...")
            df, missing = load_and_standardize_dataset(path)
            if df is not None and len(df) > 0:
                all_data.append(df)
                missing_summary[os.path.basename(path)] = missing
                print(f"  Loaded {len(df)} records")
            else:
                print(f"  Skipped {path} (no valid data)")
                missing_summary[os.path.basename(path)] = {'all': 'No data loaded'}
        else:
            print(f"  Skipping {path} (file not found)")
            missing_summary[os.path.basename(path)] = {'file': 'File not found'}
    
    if not all_data:
        print("Error: No valid datasets found")
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    
    print(f"\n{'='*60}")
    print("Super Dataset Created")
    print(f"{'='*60}")
    print(f"Total records: {len(combined)}")
    print(f"Unique locations: {combined['location'].nunique()}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"Unique crime types: {combined['crime_type'].nunique()}")
    
    print(f"\n{'='*60}")
    print("Missing Columns Summary")
    print(f"{'='*60}")
    for filename, missing in missing_summary.items():
        if missing:
            missing_list = [col for col in missing.keys() if missing[col]]
            if missing_list:
                print(f"{filename}:")
                for col in missing_list:
                    print(f"  - {col} column missing (filled with null)")
        else:
            print(f"{filename}: All columns present")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"\nSuper dataset saved to {output_path}")
    
    return combined

if __name__ == "__main__":
    np.random.seed(42)
    create_super_dataset()

