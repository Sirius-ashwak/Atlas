# Atlas: Map. Decide. Optimize. - Web Application Setup Script
# Simple and reliable version

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Atlas - Web App Setup" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Function to check if command exists
function Test-CommandExists {
    param($Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
    return $?
}

# Check prerequisites
Write-Host "[1/7] Checking prerequisites..." -ForegroundColor Yellow

if (-not (Test-CommandExists "node")) {
    Write-Host "✗ Node.js is not installed. Please install Node.js 18+ from https://nodejs.org/" -ForegroundColor Red
    exit 1
}

if (-not (Test-CommandExists "npm")) {
    Write-Host "✗ npm is not installed. Please install npm." -ForegroundColor Red
    exit 1
}

if (-not (Test-CommandExists "python")) {
    Write-Host "✗ Python is not installed. Please install Python 3.9+." -ForegroundColor Red
    exit 1
}

$nodeVersion = node --version
$npmVersion = npm --version
$pythonVersion = python --version

Write-Host "✓ Node.js: $nodeVersion" -ForegroundColor Green
Write-Host "✓ npm: $npmVersion" -ForegroundColor Green
Write-Host "✓ Python: $pythonVersion" -ForegroundColor Green
Write-Host ""

# Navigate to web-app directory
Write-Host "[2/7] Setting up web application..." -ForegroundColor Yellow

if (-not (Test-Path "web-app")) {
    Write-Host "✗ web-app directory not found! Make sure you're in the ai_edge_allocator directory." -ForegroundColor Red
    exit 1
}

Set-Location "web-app"

# Install npm dependencies
Write-Host "[3/7] Installing npm dependencies (this may take a few minutes)..." -ForegroundColor Yellow

npm install
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Failed to install npm dependencies" -ForegroundColor Red
    Set-Location ..
    exit 1
}

Write-Host "✓ npm dependencies installed successfully" -ForegroundColor Green
Write-Host ""

# Create environment files if they don't exist
Write-Host "[4/7] Creating environment files..." -ForegroundColor Yellow

if (-not (Test-Path ".env.development")) {
    "VITE_API_URL=http://localhost:8000" | Out-File -FilePath ".env.development" -Encoding utf8
    Write-Host "✓ Created .env.development" -ForegroundColor Green
}
else {
    Write-Host "✓ .env.development already exists" -ForegroundColor Green
}

if (-not (Test-Path ".env.production")) {
    "VITE_API_URL=/api" | Out-File -FilePath ".env.production" -Encoding utf8
    Write-Host "✓ Created .env.production" -ForegroundColor Green
}
else {
    Write-Host "✓ .env.production already exists" -ForegroundColor Green
}

Write-Host ""

# Go back to parent directory
Set-Location ..

# Check if FastAPI is running
Write-Host "[5/7] Checking FastAPI backend..." -ForegroundColor Yellow

$apiRunning = $false
try {
    $null = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
    Write-Host "✓ FastAPI backend is running" -ForegroundColor Green
    $apiRunning = $true
}
catch {
    Write-Host "⚠ FastAPI backend is not running" -ForegroundColor Yellow
}

Write-Host ""

# Ask user what to do
Write-Host "[6/7] Starting services..." -ForegroundColor Yellow

if (-not $apiRunning) {
    Write-Host "Would you like to start the FastAPI backend? (Y/N)" -ForegroundColor Cyan
    $startApi = Read-Host
    
    if ($startApi -eq "Y" -or $startApi -eq "y") {
        Write-Host "Starting FastAPI backend in a new terminal..." -ForegroundColor Yellow
        $scriptPath = Join-Path $PWD "python_scripts/api/run_api.py"
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; python python_scripts/api/run_api.py --port 8000"
        Write-Host "✓ FastAPI backend started in new terminal" -ForegroundColor Green
        Write-Host "  Waiting 5 seconds for API to initialize..." -ForegroundColor Yellow
        Start-Sleep -Seconds 5
    }
}

Write-Host ""

# Start React dev server
Write-Host "[7/7] Starting React development server..." -ForegroundColor Yellow

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The React development server will start now." -ForegroundColor White
Write-Host ""
Write-Host "Services:" -ForegroundColor Yellow
Write-Host "  • FastAPI Backend: http://localhost:8000" -ForegroundColor White
Write-Host "  • API Documentation: http://localhost:8000/docs" -ForegroundColor White
Write-Host "  • React Dashboard: http://localhost:3000" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop the development server" -ForegroundColor Gray
Write-Host ""

Set-Location "web-app"
npm run dev
