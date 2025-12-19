#!/bin/bash
# Setup script for Streamlit Cloud deployment
# This script runs the training pipeline to generate model files

set -e  # Exit on error

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running training pipeline..."

# Step 0: Create super dataset
if [ -f "src/00_create_super_dataset.py" ]; then
    echo "Creating super dataset..."
    python src/00_create_super_dataset.py
fi

# Step 1: Load and clean
echo "Loading and cleaning data..."
python src/01_load_clean.py

# Step 2: Engineer features
echo "Engineering features..."
python src/02_features.py

# Step 3: Create labels
echo "Creating labels..."
python src/03_labels.py

# Step 4: Train models
echo "Training models..."
python src/04_train_eval.py

echo "Setup complete! Model files are in outputs/ directory."

