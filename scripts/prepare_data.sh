#!/bin/bash
# Data Preparation Script
# Preprocesses iFogSim CSV output for ML training

set -e  # Exit on error

echo "=========================================="
echo "Data Preparation Pipeline"
echo "=========================================="

# Check if raw data exists
if [ ! -f "data/raw/sim_results.csv" ]; then
    echo "ERROR: Raw simulation data not found at data/raw/sim_results.csv"
    echo "Please run the Java simulator first to generate data."
    exit 1
fi

echo "Found raw simulation data..."

# Create output directories
mkdir -p data/processed
mkdir -p logs

# Run data preprocessing
echo "Running data preprocessing..."
python -m src.main prepare-data \
    --data-path data/raw/sim_results.csv \
    --scaler standard \
    --test-split 0.2 \
    --val-split 0.1 \
    --seed 42

echo ""
echo "=========================================="
echo "Data preparation complete!"
echo "Processed data saved to data/processed/"
echo "=========================================="

# Display statistics
if command -v python &> /dev/null; then
    python -c "
import pandas as pd
train = pd.read_csv('data/processed/train.csv')
val = pd.read_csv('data/processed/val.csv')
test = pd.read_csv('data/processed/test.csv')

print('\nDataset Statistics:')
print(f'  Training:   {len(train):,} samples')
print(f'  Validation: {len(val):,} samples')
print(f'  Test:       {len(test):,} samples')
print(f'  Total:      {len(train) + len(val) + len(test):,} samples')
print(f'\nFeatures: {list(train.columns)}')
"
fi
