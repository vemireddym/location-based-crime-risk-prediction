"""
Data Loading and Cleaning Script
Loads crime dataset from CSV, cleans missing values, and saves cleaned data.
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from collections import defaultdict

# Add parent directory to path for imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import geopy for geocoding
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOCODING_AVAILABLE = True
except ImportError:
    GEOCODING_AVAILABLE = False
    print("Warning: geopy not installed. Install with: pip install geopy")

def geocode_city_state(city, state, geocoder, cache=None, delay=1.0):
    """
    Geocode a city and state to get latitude and longitude.
    
    Args:
        city: City name
        state: State name
        geocoder: Nominatim geocoder instance
        cache: Dictionary to cache results
        delay: Delay between API calls (seconds) to respect rate limits
    
    Returns:
        tuple: (latitude, longitude) or (None, None) if geocoding fails
    """
    if pd.isna(city) or pd.isna(state):
        return None, None
    
    # Check cache first
    if cache is not None:
        cache_key = f"{str(city).strip()}, {str(state).strip()}"
        if cache_key in cache:
            return cache[cache_key]
    
    try:
        # Create location string
        location_str = f"{str(city).strip()}, {str(state).strip()}, USA"
        
        # Geocode with retry logic
        location = None
        for attempt in range(3):
            try:
                location = geocoder.geocode(location_str, timeout=10)
                break
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                if attempt < 2:
                    time.sleep(delay * (attempt + 1))
                    continue
                else:
                    print(f"  Geocoding failed for {location_str}: {e}")
                    return None, None
        
        if location:
            lat, lon = location.latitude, location.longitude
            # Cache the result
            if cache is not None:
                cache_key = f"{str(city).strip()}, {str(state).strip()}"
                cache[cache_key] = (lat, lon)
            return lat, lon
        else:
            return None, None
            
    except Exception as e:
        print(f"  Error geocoding {city}, {state}: {e}")
        return None, None

def load_and_clean_data(input_path='data/crime.csv', output_path='data/clean.csv', 
                        use_geocoding=True, geocode_delay=1.0, geocode_cache_file=None):
    """
    Load crime dataset, clean missing values, and save cleaned data.
    
    Args:
        input_path: Path to raw crime dataset CSV
        output_path: Path to save cleaned data
        use_geocoding: Whether to use geocoding if lat/lon not found
        geocode_delay: Delay between geocoding API calls (seconds)
        geocode_cache_file: Path to save/load geocoding cache (JSON file)
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
    
    # Identify required columns with better detection
    # Date columns - check for explicit date columns first, then Year/Month
    date_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'datetime', 'occurred', 'incident'])]
    year_cols = [col for col in df.columns if 'year' in col.lower()]
    month_cols = [col for col in df.columns if 'month' in col.lower()]
    
    # Latitude/Longitude - be more specific to avoid false matches
    lat_cols = [col for col in df.columns if col.lower() in ['lat', 'latitude'] or 
                (col.lower().startswith('lat') and 'relationship' not in col.lower())]
    lon_cols = [col for col in df.columns if col.lower() in ['lon', 'longitude', 'long'] or 
                col.lower().startswith('lon')]
    
    # City/State for geocoding fallback
    city_cols = [col for col in df.columns if 'city' in col.lower()]
    state_cols = [col for col in df.columns if 'state' in col.lower()]
    
    type_cols = [col for col in df.columns if any(x in col.lower() for x in ['type', 'primary', 'category', 'crime'])]
    
    print(f"\nDetected columns:")
    print(f"Date columns: {date_cols}")
    print(f"Year columns: {year_cols}")
    print(f"Month columns: {month_cols}")
    print(f"Latitude columns: {lat_cols}")
    print(f"Longitude columns: {lon_cols}")
    print(f"City columns: {city_cols}")
    print(f"State columns: {state_cols}")
    print(f"Type columns: {type_cols}")
    
    # Select the first match for each required column
    date_col = date_cols[0] if date_cols else None
    year_col = year_cols[0] if year_cols else None
    month_col = month_cols[0] if month_cols else None
    lat_col = lat_cols[0] if lat_cols else None
    lon_col = lon_cols[0] if lon_cols else None
    city_col = city_cols[0] if city_cols else None
    state_col = state_cols[0] if state_cols else None
    type_col = type_cols[0] if type_cols else None
    
    # Create a clean dataframe with standardized column names
    clean_df = pd.DataFrame()
    
    # Handle date column - try explicit date first, then Year/Month combination
    if date_col:
        print(f"\nUsing date column '{date_col}'...")
        clean_df['date'] = df[date_col]
    elif year_col and month_col:
        print(f"\nCreating date from Year and Month columns...")
        # Create a date column from Year and Month (use first day of month)
        clean_df['date'] = pd.to_datetime(
            df[year_col].astype(str) + '-' + df[month_col].astype(str).str.zfill(2) + '-01',
            errors='coerce'
        )
    else:
        print("\nError: No date/time column found!")
        print("Dataset must contain either:")
        print("  - A date/time column (with 'date', 'time', 'datetime', 'occurred' in name)")
        print("  - OR both 'Year' and 'Month' columns")
        return None
    
    # Handle latitude/longitude
    if lat_col and lon_col:
        print(f"\nUsing latitude column '{lat_col}' and longitude column '{lon_col}'...")
        clean_df['latitude'] = pd.to_numeric(df[lat_col], errors='coerce')
        clean_df['longitude'] = pd.to_numeric(df[lon_col], errors='coerce')
    elif city_col and state_col and use_geocoding:
        if not GEOCODING_AVAILABLE:
            print("\nError: geopy library not installed!")
            print("Install it with: pip install geopy")
            return None
        
        print(f"\nNo latitude/longitude columns found.")
        print(f"Geocoding City and State columns to get coordinates...")
        print(f"This may take a while for large datasets...")
        
        # Initialize geocoder
        geocoder = Nominatim(user_agent="crime_risk_prediction_app")
        
        # Load geocoding cache if available
        geocode_cache = {}
        if geocode_cache_file and os.path.exists(geocode_cache_file):
            try:
                import json
                with open(geocode_cache_file, 'r') as f:
                    geocode_cache = json.load(f)
                print(f"Loaded {len(geocode_cache)} cached geocoding results")
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
        
        # Get unique city-state combinations to minimize API calls
        print("\nFinding unique city-state combinations...")
        city_state_df = df[[city_col, state_col]].drop_duplicates()
        city_state_df = city_state_df.dropna(subset=[city_col, state_col])
        unique_count = len(city_state_df)
        print(f"Found {unique_count} unique city-state combinations")
        
        # Create mapping dictionary
        city_state_to_coords = {}
        
        print("\nGeocoding city-state combinations...")
        for idx, row in city_state_df.iterrows():
            city = row[city_col]
            state = row[state_col]
            lat, lon = geocode_city_state(city, state, geocoder, geocode_cache, geocode_delay)
            city_state_to_coords[(city, state)] = (lat, lon)
            
            if (idx + 1) % 10 == 0:
                print(f"  Progress: {idx + 1}/{unique_count} geocoded...")
            
            # Respect rate limits
            time.sleep(geocode_delay)
        
        # Save cache
        if geocode_cache_file:
            try:
                import json
                os.makedirs(os.path.dirname(geocode_cache_file) if os.path.dirname(geocode_cache_file) else '.', exist_ok=True)
                with open(geocode_cache_file, 'w') as f:
                    json.dump(geocode_cache, f)
                print(f"\nSaved geocoding cache to {geocode_cache_file}")
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")
        
        # Map coordinates back to full dataframe
        print("\nMapping coordinates to all rows...")
        clean_df['latitude'] = df.apply(
            lambda row: city_state_to_coords.get((row[city_col], row[state_col]), (None, None))[0]
            if pd.notna(row[city_col]) and pd.notna(row[state_col]) else None,
            axis=1
        )
        clean_df['longitude'] = df.apply(
            lambda row: city_state_to_coords.get((row[city_col], row[state_col]), (None, None))[1]
            if pd.notna(row[city_col]) and pd.notna(row[state_col]) else None,
            axis=1
        )
        
        # Store city/state for reference
        clean_df['city'] = df[city_col]
        clean_df['state'] = df[state_col]
        
        geocoded_count = clean_df[['latitude', 'longitude']].notna().all(axis=1).sum()
        print(f"Successfully geocoded {geocoded_count} out of {len(clean_df)} rows")
        
    elif city_col and state_col:
        print("\nError: City and State columns found but geocoding is disabled!")
        print("Set use_geocoding=True or provide latitude/longitude columns")
        return None
    else:
        print("\nError: Required columns not found!")
        print("Dataset must contain:")
        print("  - Latitude and Longitude columns, OR")
        print("  - City and State columns (for geocoding)")
        return None
    
    if type_col:
        clean_df['primary_type'] = df[type_col]
    
    # Parse date column if not already datetime
    if not pd.api.types.is_datetime64_any_dtype(clean_df['date']):
        print(f"\nParsing date column...")
        try:
            clean_df['date'] = pd.to_datetime(clean_df['date'], errors='coerce', infer_datetime_format=True)
        except Exception as e:
            print(f"Warning: Date parsing issue: {e}")
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
    
    # You can customize geocoding parameters here:
    # - geocode_cache_file: Save cache to avoid re-geocoding (e.g., 'data/geocode_cache.json')
    # - geocode_delay: Delay between API calls (default 1.0 seconds to respect rate limits)
    clean_data = load_and_clean_data(
        geocode_cache_file='data/geocode_cache.json',  # Cache geocoding results
        geocode_delay=1.0  # 1 second delay between API calls
    )
    
    if clean_data is not None:
        print("\n" + "=" * 60)
        print("Data cleaning completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Data cleaning failed. Please check the error messages above.")
        print("=" * 60)
        sys.exit(1)
