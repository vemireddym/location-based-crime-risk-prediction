"""
Label Creation Script
Creates risk level labels (Low/Medium/High) based on crime counts per grid cell and time window.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_labels(input_path='data/features.csv', output_path='data/model_ready.csv', 
                  time_window_hours=24, low_threshold=0.70, medium_threshold=0.90):
    """
    Create risk level labels based on crime counts per grid cell and time window.
    
    Risk levels:
    - Low = bottom 70% (0-70th percentile)
    - Medium = next 20% (70-90th percentile)
    - High = top 10% (90-100th percentile)
    
    Args:
        input_path: Path to features CSV
        output_path: Path to save model-ready dataset
        time_window_hours: Time window in hours for counting crimes (default: 24)
        low_threshold: Percentile threshold for Low risk (default: 0.70)
        medium_threshold: Percentile threshold for Medium risk (default: 0.90)
    """
    print("=" * 60)
    print("Step 3: Creating Labels")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        print("Please run 02_features.py first.")
        return None
    
    # Load features
    print(f"\nLoading features from {input_path}...")
    try:
        df = pd.read_csv(input_path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Sort by date for time-window calculations
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"\nCreating labels based on crime counts...")
    print(f"Time window: {time_window_hours} hours")
    print(f"Risk thresholds: Low (0-{low_threshold*100:.0f}%), Medium ({low_threshold*100:.0f}-{medium_threshold*100:.0f}%), High ({medium_threshold*100:.0f}-100%)")
    
    # Count crimes per grid cell + time window
    print("\nCounting crimes per grid cell and time window...")
    
    # Create time windows
    df['time_window'] = df['date'].dt.floor(f'{time_window_hours}H')
    
    # Count crimes in each grid cell + time window combination
    crime_counts = df.groupby(['grid_cell', 'time_window']).size().reset_index(name='crime_count')
    
    print(f"  - Unique grid cell + time window combinations: {len(crime_counts)}")
    print(f"  - Crime count statistics:")
    print(f"    Min: {crime_counts['crime_count'].min()}")
    print(f"    Max: {crime_counts['crime_count'].max()}")
    print(f"    Mean: {crime_counts['crime_count'].mean():.2f}")
    print(f"    Median: {crime_counts['crime_count'].median():.2f}")
    
    # Merge crime counts back to original dataframe
    df = df.merge(crime_counts, on=['grid_cell', 'time_window'], how='left')
    df['crime_count'] = df['crime_count'].fillna(0).astype(int)
    
    # Calculate percentiles for risk level assignment
    print("\nCalculating risk level thresholds...")
    
    # Get unique crime counts and their percentiles
    unique_counts = df['crime_count'].unique()
    percentiles = np.percentile(df['crime_count'], [low_threshold * 100, medium_threshold * 100])
    
    low_threshold_value = percentiles[0]
    medium_threshold_value = percentiles[1]
    
    print(f"  - Low threshold (70th percentile): {low_threshold_value:.2f}")
    print(f"  - Medium threshold (90th percentile): {medium_threshold_value:.2f}")
    
    # Assign risk levels
    print("\nAssigning risk levels...")
    
    def assign_risk_level(count):
        if count <= low_threshold_value:
            return 'Low'
        elif count <= medium_threshold_value:
            return 'Medium'
        else:
            return 'High'
    
    df['risk_level'] = df['crime_count'].apply(assign_risk_level)
    
    # Display label distribution
    print("\nRisk level distribution:")
    label_counts = df['risk_level'].value_counts()
    label_percentages = df['risk_level'].value_counts(normalize=True) * 100
    
    for level in ['Low', 'Medium', 'High']:
        count = label_counts.get(level, 0)
        pct = label_percentages.get(level, 0)
        print(f"  - {level}: {count} ({pct:.2f}%)")
    
    # Verify distribution matches expected thresholds
    actual_low = label_percentages.get('Low', 0)
    actual_medium = label_percentages.get('Medium', 0)
    actual_high = label_percentages.get('High', 0)
    
    print(f"\nActual distribution vs Expected:")
    print(f"  Low: {actual_low:.2f}% (expected ~{low_threshold*100:.0f}%)")
    print(f"  Medium: {actual_medium:.2f}% (expected ~{(medium_threshold-low_threshold)*100:.0f}%)")
    print(f"  High: {actual_high:.2f}% (expected ~{(1-medium_threshold)*100:.0f}%)")
    
    # Prepare final dataset for modeling
    print("\nPreparing final dataset for modeling...")
    
    # Select features for modeling
    model_features = [
        'hour',
        'day_of_week',
        'month',
        'year',
        'grid_lat',
        'grid_lon',
        'past_crime_count',
        'crime_count_30d',
        'risk_level'  # This is our target label
    ]
    
    # Check which columns exist
    available_features = [col for col in model_features if col in df.columns]
    
    model_df = df[available_features].copy()
    
    # Remove rows with missing values in features
    initial_len = len(model_df)
    model_df = model_df.dropna()
    final_len = len(model_df)
    
    if initial_len != final_len:
        print(f"  - Removed {initial_len - final_len} rows with missing values")
    
    print(f"  - Final dataset shape: {model_df.shape}")
    print(f"  - Features: {len(model_df.columns) - 1}")  # -1 for target
    print(f"  - Target: risk_level")
    
    # Display feature summary
    print(f"\nFeature summary:")
    for col in model_df.columns:
        if col != 'risk_level':
            if model_df[col].dtype in ['int64', 'float64']:
                print(f"  - {col}: {model_df[col].min():.2f} to {model_df[col].max():.2f} (mean: {model_df[col].mean():.2f})")
            else:
                print(f"  - {col}: {model_df[col].nunique()} unique values")
    
    # Save model-ready dataset
    print(f"\nSaving model-ready dataset to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model_df.to_csv(output_path, index=False)
    print(f"✓ Model-ready dataset saved successfully!")
    
    return model_df

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    model_data = create_labels()
    
    if model_data is not None:
        print("\n" + "=" * 60)
        print("Label creation completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Label creation failed. Please check the error messages above.")
        print("=" * 60)
        sys.exit(1)

