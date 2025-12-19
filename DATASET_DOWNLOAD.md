# Dataset Download Instructions

Download the following datasets and place them in the `data/` folder:

## Required Datasets

### 1. Chicago Crime Data
- **Link**: https://www.kaggle.com/datasets/currie32/crimes-in-chicago
- **Save as**: `data/chicago_crime.csv`
- **Columns needed**: Date, Latitude, Longitude, Primary Type
- **Note**: Sample to 300K-400K rows for faster processing

### 2. San Francisco Crime Data
- **Link**: https://www.kaggle.com/datasets/wosaku/crime-in-san-francisco
- **Save as**: `data/sf_crime.csv`
- **Columns needed**: Date, X (longitude), Y (latitude), Category
- **Note**: Sample to 300K-400K rows for faster processing

## Quick Download (if you have Kaggle API)

```bash
# Install kaggle API: pip install kaggle
# Set up credentials: https://www.kaggle.com/docs/api

kaggle datasets download -d currie32/crimes-in-chicago -p data/
kaggle datasets download -d wosaku/crime-in-san-francisco -p data/

# Extract and rename
unzip data/crimes-in-chicago.zip -d data/
unzip data/crime-in-san-francisco.zip -d data/
```

## Sampling Script

After downloading, run this to sample datasets:

```python
import pandas as pd

# Sample Chicago data
chicago = pd.read_csv('data/chicago_crime.csv', nrows=400000)
chicago.to_csv('data/chicago_crime.csv', index=False)

# Sample SF data
sf = pd.read_csv('data/sf_crime.csv', nrows=400000)
sf.to_csv('data/sf_crime.csv', index=False)
```

