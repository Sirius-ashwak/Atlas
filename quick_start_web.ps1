# Super Simple Quick Start for Web App
# Just runs the basic commands

Write-Host "`n🚀 Starting AI Edge Allocator Web Dashboard...`n" -ForegroundColor Cyan

# Check if in correct directory
if (-not (Test-Path "web-app")) {
    Write-Host "❌ Please run this from the ai_edge_allocator directory" -ForegroundColor Red
    exit 1
}

# Check if node_modules exists
Set-Location "web-app"
if (-not (Test-Path "node_modules")) {
    Write-Host "📦 Installing dependencies (first time only)...`n" -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Installation failed" -ForegroundColor Red
        Set-Location ..
        exit 1
    }
}

Write-Host "✅ Starting development server..." -ForegroundColor Green
Write-Host "📊 Dashboard will open at: http://localhost:3000`n" -ForegroundColor Cyan

npm run dev
