# Dataset Download Instructions

Download datasets with City/State location information (not lat/lon coordinates).

## Required Datasets (5 Cities)

### 1. Existing Crime Dataset (1980-2004)
- **File**: `data/crime.csv` (already have)
- **Columns**: Year, Month, City, State, Crime Type
- **Note**: Already in data folder

### 2. Chicago Crime Data
- **Link**: https://www.kaggle.com/datasets/currie32/crimes-in-chicago
- **Save as**: `data/chicago_crime.csv`
- **Columns needed**: Date, City (or Location), State, Crime Type
- **Note**: Sample to 300K-400K rows for faster processing

### 3. Los Angeles Crime Data
- **Link**: https://www.kaggle.com/datasets/cityofLA/crime-in-los-angeles
- **Save as**: `data/la_crime.csv`
- **Columns needed**: Date, City, State, Crime Type
- **Note**: Sample to 300K-400K rows for faster processing

### 4. Boston Crime Data
- **Link**: https://www.kaggle.com/datasets/AnalyzeBoston/crimes-in-boston
- **Save as**: `data/boston_crime.csv`
- **Columns needed**: Date, City, State, Crime Type
- **Note**: Sample to 300K-400K rows for faster processing

### 5. Philadelphia Crime Data
- **Link**: https://www.kaggle.com/datasets/mchirico/philadelphiacrimedata
- **Save as**: `data/philly_crime.csv`
- **Columns needed**: Date, City, State, Crime Type
- **Note**: Sample to 300K-400K rows for faster processing

## Quick Download (if you have Kaggle API)

```bash
# Install kaggle API: pip install kaggle
# Set up credentials: https://www.kaggle.com/docs/api

kaggle datasets download -d currie32/crimes-in-chicago -p data/
kaggle datasets download -d cityofLA/crime-in-los-angeles -p data/
kaggle datasets download -d AnalyzeBoston/crimes-in-boston -p data/
kaggle datasets download -d mchirico/philadelphiacrimedata -p data/

# Extract and rename (adjust filenames based on actual downloaded files)
unzip data/crimes-in-chicago.zip -d data/
unzip data/crime-in-los-angeles.zip -d data/
unzip data/crimes-in-boston.zip -d data/
unzip data/philadelphiacrimedata.zip -d data/
```

## Sampling Script

After downloading, run this to sample datasets:

```python
import pandas as pd

datasets = [
    'data/chicago_crime.csv',
    'data/la_crime.csv',
    'data/boston_crime.csv',
    'data/philly_crime.csv'
]

for filepath in datasets:
    try:
        df = pd.read_csv(filepath, nrows=400000)
        df.to_csv(filepath, index=False)
        print(f"Sampled {filepath}: {len(df)} rows")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
```

## Notes

- All datasets should have City and State columns (location names, not coordinates)
- The model will predict risk based on location (city/state) and time
- Missing columns will be handled gracefully (set to null, processing continues)
- Datasets will be combined into one super dataset for training
- Total expected records: ~1.5-2M across all datasets (after sampling)

## Pipeline Order

1. Run `python src/00_create_super_dataset.py` - Creates merged super dataset
2. Run `python src/01_load_clean.py` - Cleans the super dataset
3. Run `python src/02_features.py` - Feature engineering
4. Run `python src/03_labels.py` - Create labels
5. Run `python src/04_train_eval.py` - Train models
