# Quick start script for the web application
Write-Host "Starting Atlas Web Dashboard..." -ForegroundColor Cyan

# Check if node_modules exists
Set-Location "web-app"

if (-not (Test-Path "node_modules")) {
    Write-Host "Dependencies not installed. Running npm install..." -ForegroundColor Yellow
    npm install
}

Write-Host ""
Write-Host "Starting development server..." -ForegroundColor Green
Write-Host "Dashboard will be available at: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""

npm run dev
