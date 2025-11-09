# Data Preparation Script (PowerShell version for Windows)
# Preprocesses iFogSim CSV output for ML training

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Data Preparation Pipeline" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Check if raw data exists
if (-not (Test-Path "data/raw/sim_results.csv")) {
    Write-Host "ERROR: Raw simulation data not found at data/raw/sim_results.csv" -ForegroundColor Red
    Write-Host "Please run the Java simulator first to generate data." -ForegroundColor Yellow
    exit 1
}

Write-Host "Found raw simulation data..." -ForegroundColor Green

# Create output directories
New-Item -ItemType Directory -Force -Path "data/processed" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

# Run data preprocessing
Write-Host "Running data preprocessing..." -ForegroundColor Yellow
python -m src.main prepare-data `
    --data-path data/raw/sim_results.csv `
    --scaler standard `
    --test-split 0.2 `
    --val-split 0.1 `
    --seed 42

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Data preparation complete!" -ForegroundColor Green
Write-Host "Processed data saved to data/processed/" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan

# Display statistics
python -c @"
import pandas as pd
train = pd.read_csv('data/processed/train.csv')
val = pd.read_csv('data/processed/val.csv')
test = pd.read_csv('data/processed/test.csv')

print('\nDataset Statistics:')
print(f'  Training:   {len(train):,} samples')
print(f'  Validation: {len(val):,} samples')
print(f'  Test:       {len(test):,} samples')
print(f'  Total:      {len(train) + len(val) + len(test):,} samples')
"@
