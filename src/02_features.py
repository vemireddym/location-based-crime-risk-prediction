"""
Feature Engineering Script
Extracts temporal features and creates location grid cells from cleaned data.
"""

import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for imports if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def engineer_features(input_path='data/clean.csv', output_path='data/features.csv', grid_precision=2):
    """
    Engineer features from cleaned data:
    - Temporal features: hour, day_of_week, month
    - Location features: grid cell (binned lat/lon)
    - Historical features: past crime count, rolling window count
    
    Args:
        input_path: Path to cleaned data CSV
        output_path: Path to save features
        grid_precision: Decimal places for grid binning (2-3 recommended)
    """
    print("=" * 60)
    print("Step 2: Feature Engineering")
    print("=" * 60)
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        print("Please run 01_load_clean.py first.")
        return None
    
    # Load cleaned data
    print(f"\nLoading cleaned data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
        df['date'] = pd.to_datetime(df['date'])
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
    # Create features dataframe
    features_df = df[['date', 'latitude', 'longitude']].copy()
    
    # Extract temporal features
    print("\nExtracting temporal features...")
    features_df['hour'] = df['date'].dt.hour
    features_df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    features_df['month'] = df['date'].dt.month
    features_df['year'] = df['date'].dt.year
    
    print(f"  - Hour: {features_df['hour'].min()} to {features_df['hour'].max()}")
    print(f"  - Day of week: {features_df['day_of_week'].min()} to {features_df['day_of_week'].max()}")
    print(f"  - Month: {features_df['month'].min()} to {features_df['month'].max()}")
    print(f"  - Year: {features_df['year'].min()} to {features_df['year'].max()}")
    
    # Create location grid cells
    print(f"\nCreating location grid cells (precision={grid_precision})...")
    
    # Method 1: Round to specified decimal places (simple binning)
    features_df['grid_lat'] = features_df['latitude'].round(grid_precision)
    features_df['grid_lon'] = features_df['longitude'].round(grid_precision)
    features_df['grid_cell'] = (
        features_df['grid_lat'].astype(str) + '_' + 
        features_df['grid_lon'].astype(str)
    )
    
    print(f"  - Created {features_df['grid_cell'].nunique()} unique grid cells")
    print(f"  - Grid cell examples: {features_df['grid_cell'].head(3).tolist()}")
    
    # Calculate grid cell statistics
    grid_stats = features_df.groupby('grid_cell').size().describe()
    print(f"\nGrid cell statistics:")
    print(f"  - Mean crimes per cell: {grid_stats['mean']:.2f}")
    print(f"  - Median crimes per cell: {grid_stats['50%']:.2f}")
    print(f"  - Max crimes per cell: {grid_stats['max']:.0f}")
    
    # Calculate past crime count per grid cell (OPTIMIZED - vectorized)
    print("\nCalculating past crime count per grid cell...")
    
    # Sort by date for cumulative calculation
    features_df = features_df.sort_values('date').reset_index(drop=True)
    
    # Vectorized cumulative count per grid cell (much faster than row-by-row)
    features_df['past_crime_count'] = features_df.groupby('grid_cell').cumcount()
    
    print(f"  - Past crime count range: {features_df['past_crime_count'].min()} to {features_df['past_crime_count'].max()}")
    print(f"  - Mean past crime count: {features_df['past_crime_count'].mean():.2f}")
    
    # Calculate rolling window crime count (OPTIMIZED - vectorized approach)
    print("\nCalculating rolling window crime count (last 30 days)...")
    
    # Set date as index for rolling operations
    features_df = features_df.sort_values('date')
    
    # Create a helper column for counting
    features_df['count_helper'] = 1
    
    # Group by grid_cell and apply rolling count
    # This is much faster than row-by-row iteration
    def rolling_count_30d(group):
        """Calculate 30-day rolling count for a single grid cell."""
        group = group.sort_values('date')
        group = group.set_index('date')
        # Rolling count of crimes in last 30 days (excluding current row with closed='left')
        rolling_counts = group['count_helper'].rolling('30D', closed='left').sum().fillna(0)
        group['crime_count_30d'] = rolling_counts.values
        group = group.reset_index()
        return group
    
    print("  - Processing grid cells (this may take a moment)...")
    
    # Apply rolling count to each grid cell group
    result_dfs = []
    grid_cells = features_df['grid_cell'].unique()
    total_cells = len(grid_cells)
    
    for i, grid_cell in enumerate(grid_cells):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  - Progress: {i + 1}/{total_cells} grid cells processed...")
        
        group = features_df[features_df['grid_cell'] == grid_cell].copy()
        group = group.sort_values('date').set_index('date')
        
        # Rolling count with 30-day window (excluding current observation)
        rolling_counts = group['count_helper'].rolling('30D', closed='left').sum().fillna(0)
        group['crime_count_30d'] = rolling_counts.astype(int)
        group = group.reset_index()
        result_dfs.append(group)
    
    # Combine all results
    features_df = pd.concat(result_dfs, ignore_index=True)
    
    # Drop helper column
    features_df = features_df.drop(columns=['count_helper'])
    
    print(f"  - 30-day crime count range: {features_df['crime_count_30d'].min()} to {features_df['crime_count_30d'].max()}")
    print(f"  - Mean 30-day crime count: {features_df['crime_count_30d'].mean():.2f}")
    
    # Select final features for modeling
    final_features = features_df[[
        'hour',
        'day_of_week',
        'month',
        'year',
        'grid_lat',
        'grid_lon',
        'grid_cell',
        'past_crime_count',
        'crime_count_30d',
        'latitude',  # Keep original for reference
        'longitude',  # Keep original for reference
        'date'  # Keep for label creation
    ]].copy()
    
    # Display feature summary
    print(f"\nFinal features:")
    print(f"  - Temporal: hour, day_of_week, month, year")
    print(f"  - Location: grid_lat, grid_lon, grid_cell")
    print(f"  - Historical: past_crime_count, crime_count_30d")
    print(f"  - Total features: {len(final_features.columns)}")
    
    # Save features
    print(f"\nSaving features to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_features.to_csv(output_path, index=False)
    print(f"✓ Features saved successfully!")
    
    return final_features

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    features = engineer_features()
    
    if features is not None:
        print("\n" + "=" * 60)
        print("Feature engineering completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Feature engineering failed. Please check the error messages above.")
        print("=" * 60)
        sys.exit(1)
