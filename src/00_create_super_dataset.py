import pandas as pd
import numpy as np
import os

def find_column(df, keywords):
    for col in df.columns:
        if any(kw.lower() in col.lower() for kw in keywords):
            return col
    return None

def extract_city_state_from_filename(filename):
    city_state_map = {
        'chicago': ('Chicago', 'IL'),
        'la': ('Los Angeles', 'CA'),
        'los angeles': ('Los Angeles', 'CA'),
        'boston': ('Boston', 'MA'),
        'philly': ('Philadelphia', 'PA'),
        'philadelphia': ('Philadelphia', 'PA'),
        'new york': ('New York', 'NY'),
        'ny': ('New York', 'NY')
    }
    
    filename_lower = filename.lower()
    for key, (city, state) in city_state_map.items():
        if key in filename_lower:
            return city, state
    return None, None

def load_and_standardize_dataset(filepath):
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except:
        try:
            df = pd.read_csv(filepath, encoding='latin-1', low_memory=False)
        except:
            return None, {}, {}
    
    missing_cols = {}
    auto_filled = {}
    clean_df = pd.DataFrame()
    
    filename = os.path.basename(filepath)
    default_city, default_state = extract_city_state_from_filename(filename)
    
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
        clean_df['city'] = df[city_col].astype(str).str.strip().replace('nan', pd.NA)
    else:
        if default_city:
            clean_df['city'] = default_city
            auto_filled['city'] = f'Auto-filled: {default_city}'
        else:
            clean_df['city'] = pd.NA
            missing_cols['city'] = True
    
    if state_col:
        clean_df['state'] = df[state_col].astype(str).str.strip().replace('nan', pd.NA)
    else:
        if default_state:
            clean_df['state'] = default_state
            auto_filled['state'] = f'Auto-filled: {default_state}'
        else:
            clean_df['state'] = pd.NA
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
    
    return clean_df, missing_cols, auto_filled

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
    auto_filled_summary = {}
    
    for path in input_paths:
        if os.path.exists(path):
            print(f"Processing {path}...")
            df, missing, auto_filled = load_and_standardize_dataset(path)
            if df is not None and len(df) > 0:
                all_data.append(df)
                missing_summary[os.path.basename(path)] = missing
                if auto_filled:
                    auto_filled_summary[os.path.basename(path)] = auto_filled
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
    print("Column Status Summary")
    print(f"{'='*60}")
    for filename, missing in missing_summary.items():
        if filename in auto_filled_summary:
            print(f"{filename}:")
            for col, msg in auto_filled_summary[filename].items():
                print(f"  - {col}: {msg}")
        if missing:
            missing_list = [col for col in missing.keys() if missing[col]]
            if missing_list:
                if filename not in auto_filled_summary:
                    print(f"{filename}:")
                for col in missing_list:
                    print(f"  - {col} column missing (filled with null)")
        elif filename not in auto_filled_summary:
            print(f"{filename}: All columns present")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"\nSuper dataset saved to {output_path}")
    
    return combined

if __name__ == "__main__":
    np.random.seed(42)
    create_super_dataset()

