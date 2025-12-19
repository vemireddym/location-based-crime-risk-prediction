# Git LFS Setup for Large Model Files

The model files are too large for regular Git:
- `crime_model.pkl`: 816MB (exceeds GitHub's 100MB limit)
- `risk_model.pkl`: 58MB (exceeds GitHub's 50MB recommendation)

We need to use **Git Large File Storage (LFS)** to handle these files.

## Option 1: Install Git LFS and Push Models

### Step 1: Install Git LFS

**macOS:**
```bash
brew install git-lfs
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# Or download from: https://git-lfs.github.com/
```

**Windows:**
Download and install from: https://git-lfs.github.com/

### Step 2: Initialize Git LFS in Repository

```bash
# Initialize Git LFS
git lfs install

# Track .pkl files in outputs/
git lfs track "outputs/*.pkl"

# Add .gitattributes (created by git lfs track)
git add .gitattributes

# Remove the large files from the last commit
git reset HEAD~1

# Re-add files (they'll now be tracked by LFS)
git add outputs/*.pkl outputs/confusion_matrix.png outputs/results.txt .gitignore

# Commit
git commit -m "Add model files using Git LFS"

# Push
git push origin main
```

## Option 2: Don't Commit Models (Recommended for Streamlit Cloud)

Instead of committing large model files, users can train models locally:

1. **For Local Development:**
   - Train models locally: `python src/04_train_eval.py`
   - Models will be in `outputs/` folder
   - Streamlit app will work locally

2. **For Streamlit Cloud Deployment:**
   - Streamlit Cloud can run the training pipeline during deployment
   - Add a `setup.sh` script that runs training
   - Or use Streamlit's `packages.txt` to install dependencies and run training

### Create setup.sh for Streamlit Cloud

```bash
#!/bin/bash
# setup.sh - Run this during Streamlit Cloud deployment

# Install dependencies
pip install -r requirements.txt

# Run training pipeline
python src/00_create_super_dataset.py
python src/01_load_clean.py
python src/02_features.py
python src/03_labels.py
python src/04_train_eval.py
```

Then in Streamlit Cloud settings, set the setup script to `setup.sh`.

## Option 3: Use Model Storage Service

Instead of Git, store models in:
- **AWS S3** / **Google Cloud Storage** / **Azure Blob Storage**
- Download models at runtime in the Streamlit app
- More scalable for production

## Current Status

The model files are committed locally but not pushed to GitHub due to size limits. Choose one of the options above to proceed.

