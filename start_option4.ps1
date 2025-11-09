# PowerShell Script to Start Option 4: Hybrid Simulation Setup
# Atlas: Map. Decide. Optimize. - MQTT-based Real-time System

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Atlas - Option 4 Setup   " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✅ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

# Check if Docker is running (for MQTT broker)
Write-Host "[2/5] Checking Docker..." -ForegroundColor Yellow
try {
    docker ps > $null 2>&1
    Write-Host "✅ Docker is running" -ForegroundColor Green
    $useDocker = $true
} catch {
    Write-Host "⚠️  Docker not running. Will need manual MQTT broker setup" -ForegroundColor Yellow
    $useDocker = $false
}

# Install MQTT dependency
Write-Host "[3/5] Installing MQTT dependencies..." -ForegroundColor Yellow
try {
    pip install paho-mqtt --quiet
    Write-Host "✅ paho-mqtt installed" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Failed to install paho-mqtt" -ForegroundColor Yellow
}

# Start MQTT Broker
Write-Host "[4/5] Starting MQTT Broker..." -ForegroundColor Yellow
if ($useDocker) {
    # Check if broker already running
    $existingBroker = docker ps --filter "name=mqtt-broker" --format "{{.Names}}"
    
    if ($existingBroker -eq "mqtt-broker") {
        Write-Host "✅ MQTT broker already running" -ForegroundColor Green
    } else {
        Write-Host "   Starting Eclipse Mosquitto in Docker..." -ForegroundColor Cyan
        docker run -d --name mqtt-broker -p 1883:1883 -p 9001:9001 eclipse-mosquitto:2.0
        Start-Sleep -Seconds 3
        Write-Host "✅ MQTT broker started on port 1883" -ForegroundColor Green
    }
} else {
    Write-Host "   Please start MQTT broker manually:" -ForegroundColor Yellow
    Write-Host "   Download from: https://mosquitto.org/download/" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "   Press Enter once broker is running, or 'q' to quit"
    if ($response -eq 'q') { exit 0 }
}

# Start components in new windows
Write-Host "[5/5] Starting components..." -ForegroundColor Yellow
Write-Host ""

# Start IoT Device Simulator
Write-Host "   Starting IoT Device Simulator..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @"
    -NoExit
    -Command `"cd '$PWD'; Write-Host '🌐 IoT Device Simulator' -ForegroundColor Green; python iot_device_simulator.py --num-devices 15 --interval 5`"
"@
Start-Sleep -Seconds 2

# Start FastAPI Server
Write-Host "   Starting FastAPI Server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @"
    -NoExit
    -Command `"cd '$PWD'; Write-Host '🚀 FastAPI Server' -ForegroundColor Green; python python_scripts/api/run_api.py --port 8000`"
"@
Start-Sleep -Seconds 3

# Start Real-time Dashboard
Write-Host "   Starting Real-time Dashboard..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @"
    -NoExit
    -Command `"cd '$PWD'; Write-Host '📊 Real-time Dashboard' -ForegroundColor Green; streamlit run python_scripts/dashboard/dashboard_realtime.py --server.port 8502`"
"@
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  ✅ All components started!           " -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Access points:" -ForegroundColor Cyan
Write-Host "  • MQTT Broker:      mqtt://localhost:1883" -ForegroundColor White
Write-Host "  • FastAPI:          http://localhost:8000" -ForegroundColor White
Write-Host "  • API Docs:         http://localhost:8000/docs" -ForegroundColor White
Write-Host "  • Real-time Dashboard: http://localhost:8502" -ForegroundColor White
Write-Host ""
Write-Host "Components running in separate windows:" -ForegroundColor Yellow
Write-Host "  1. IoT Device Simulator (publishing telemetry)" -ForegroundColor White
Write-Host "  2. FastAPI Server (REST API)" -ForegroundColor White
Write-Host "  3. Streamlit Dashboard (visualization)" -ForegroundColor White
Write-Host ""
Write-Host "To stop all services:" -ForegroundColor Yellow
Write-Host "  1. Close all PowerShell windows" -ForegroundColor White
Write-Host "  2. Stop MQTT broker: docker stop mqtt-broker" -ForegroundColor White
Write-Host ""
Write-Host "Press any key to exit this window..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
