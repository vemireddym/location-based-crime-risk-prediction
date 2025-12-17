"""
Data Loading and Cleaning Script
Loads crime dataset from CSV, cleans missing values, and saves cleaned data.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_and_clean_data(input_path='data/crime.csv', output_path='data/clean.csv'):
    """
    Load crime dataset, clean missing values, and save cleaned data.
    
    Args:
        input_path: Path to raw crime dataset CSV
        output_path: Path to save cleaned data
    """
    print("=" * 60)
    print("Step 1: Loading and Cleaning Data")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        print("Please download the crime dataset and save it as 'data/crime.csv'")
        return None
    
    # Load data
    print(f"\nLoading data from {input_path}...")
    try:
        # Try different encodings
        try:
            df = pd.read_csv(input_path, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, encoding='latin-1', low_memory=False)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Display initial info
    print(f"\nInitial data info:")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"\nMissing values per column:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    # Identify required columns
    # Common column names in crime datasets
    date_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'datetime', 'occurred'])]
    lat_cols = [col for col in df.columns if any(x in col.lower() for x in ['lat', 'latitude'])]
    lon_cols = [col for col in df.columns if any(x in col.lower() for x in ['lon', 'longitude', 'long'])]
    type_cols = [col for col in df.columns if any(x in col.lower() for x in ['type', 'primary', 'category', 'crime'])]
    
    print(f"\nDetected columns:")
    print(f"Date columns: {date_cols}")
    print(f"Latitude columns: {lat_cols}")
    print(f"Longitude columns: {lon_cols}")
    print(f"Type columns: {type_cols}")
    
    # Select the first match for each required column
    date_col = date_cols[0] if date_cols else None
    lat_col = lat_cols[0] if lat_cols else None
    lon_col = lon_cols[0] if lon_cols else None
    type_col = type_cols[0] if type_cols else None
    
    if not date_col or not lat_col or not lon_col:
        print("\nError: Required columns not found!")
        print("Dataset must contain: date/time, latitude, longitude")
        return None
    
    # Create a clean dataframe with standardized column names
    clean_df = pd.DataFrame()
    clean_df['date'] = df[date_col]
    clean_df['latitude'] = pd.to_numeric(df[lat_col], errors='coerce')
    clean_df['longitude'] = pd.to_numeric(df[lon_col], errors='coerce')
    
    if type_col:
        clean_df['primary_type'] = df[type_col]
    
    # Parse date column
    print(f"\nParsing date column '{date_col}'...")
    try:
        clean_df['date'] = pd.to_datetime(clean_df['date'], errors='coerce', infer_datetime_format=True)
    except Exception as e:
        print(f"Warning: Date parsing issue: {e}")
        # Try alternative parsing
        clean_df['date'] = pd.to_datetime(clean_df['date'], errors='coerce')
    
    # Drop rows with missing essential values
    print("\nCleaning data...")
    initial_count = len(clean_df)
    
    # Drop rows with missing lat, lon, or date
    clean_df = clean_df.dropna(subset=['latitude', 'longitude', 'date'])
    
    # Remove invalid coordinates (outside reasonable ranges)
    clean_df = clean_df[
        (clean_df['latitude'] >= -90) & (clean_df['latitude'] <= 90) &
        (clean_df['longitude'] >= -180) & (clean_df['longitude'] <= 180)
    ]
    
    final_count = len(clean_df)
    removed = initial_count - final_count
    
    print(f"Removed {removed} rows ({removed/initial_count*100:.2f}%)")
    print(f"Final dataset shape: {clean_df.shape}")
    
    # Display summary statistics
    print(f"\nCleaned data summary:")
    print(f"Date range: {clean_df['date'].min()} to {clean_df['date'].max()}")
    print(f"Latitude range: {clean_df['latitude'].min():.4f} to {clean_df['latitude'].max():.4f}")
    print(f"Longitude range: {clean_df['longitude'].min():.4f} to {clean_df['longitude'].max():.4f}")
    
    if 'primary_type' in clean_df.columns:
        print(f"\nCrime types: {clean_df['primary_type'].nunique()} unique types")
        print(f"Top 5 crime types:")
        print(clean_df['primary_type'].value_counts().head())
    
    # Save cleaned data
    print(f"\nSaving cleaned data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    clean_df.to_csv(output_path, index=False)
    print(f"✓ Cleaned data saved successfully!")
    
    return clean_df

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    clean_data = load_and_clean_data()
    
    if clean_data is not None:
        print("\n" + "=" * 60)
        print("Data cleaning completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Data cleaning failed. Please check the error messages above.")
        print("=" * 60)
        sys.exit(1)

