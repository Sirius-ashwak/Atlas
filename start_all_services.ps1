# Complete Option 4 Startup Script
# Starts all services in correct order with proper delays

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  Starting All Option 4 Services       " -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Function to check if port is listening
function Test-Port {
    param($Port)
    $connection = Test-NetConnection -ComputerName localhost -Port $Port -WarningAction SilentlyContinue
    return $connection.TcpTestSucceeded
}

# Step 1: Check MQTT Broker
Write-Host "[1/4] Checking MQTT Broker..." -ForegroundColor Yellow
$brokerRunning = docker ps --filter "name=mqtt-broker" --format "{{.Names}}" 2>$null
if ($brokerRunning -eq "mqtt-broker") {
    Write-Host "✅ MQTT broker already running" -ForegroundColor Green
} else {
    Write-Host "   Starting MQTT broker..." -ForegroundColor Cyan
    docker run -d --name mqtt-broker -p 1883:1883 -p 9001:9001 eclipse-mosquitto:2.0 mosquitto -c /mosquitto-no-auth.conf 2>$null
    Write-Host "   Waiting for broker to start..." -ForegroundColor Cyan
    Start-Sleep -Seconds 3
    Write-Host "✅ MQTT broker started" -ForegroundColor Green
}

# Step 2: Start FastAPI
Write-Host "`n[2/4] Starting FastAPI Server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host '🚀 FastAPI Server (Port 8000)' -ForegroundColor Green; python python_scripts/api/run_api.py --port 8000"
Write-Host "   Waiting for API to initialize..." -ForegroundColor Cyan
Start-Sleep -Seconds 4
Write-Host "✅ FastAPI started" -ForegroundColor Green

# Step 3: Start IoT Simulator
Write-Host "`n[3/4] Starting IoT Device Simulator..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host '🌐 IoT Device Simulator (15 devices)' -ForegroundColor Green; python iot_device_simulator.py --num-devices 15 --interval 5"
Write-Host "   Waiting for simulator to connect..." -ForegroundColor Cyan
Start-Sleep -Seconds 3
Write-Host "✅ Simulator started" -ForegroundColor Green

# Step 4: Start Real-time Dashboard
Write-Host "`n[4/4] Starting Real-time Dashboard..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; Write-Host '📊 Real-time Dashboard (Port 8502)' -ForegroundColor Green; streamlit run python_scripts/dashboard/dashboard_realtime.py --server.port 8502"
Write-Host "   Waiting for dashboard to load..." -ForegroundColor Cyan
Start-Sleep -Seconds 5
Write-Host "✅ Dashboard started" -ForegroundColor Green

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "  ✅ All Services Started Successfully! " -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

Write-Host "Access your services:" -ForegroundColor Cyan
Write-Host "  • Real-time Dashboard: " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8502" -ForegroundColor Yellow
Write-Host "  • FastAPI:            " -NoNewline -ForegroundColor White  
Write-Host "http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host "  • API Health:         " -NoNewline -ForegroundColor White
Write-Host "http://localhost:8000/health" -ForegroundColor Yellow

Write-Host "`nIn the Dashboard (http://localhost:8502):" -ForegroundColor Cyan
Write-Host "  1. Look at the sidebar" -ForegroundColor White
Write-Host "  2. Broker: localhost" -ForegroundColor White
Write-Host "  3. Port: 1883" -ForegroundColor White
Write-Host "  4. Click '🔌 Connect'" -ForegroundColor White
Write-Host "  5. You should see '🟢 LIVE' indicator" -ForegroundColor Green

Write-Host "`nTo stop all services:" -ForegroundColor Yellow
Write-Host "  1. Close all PowerShell windows" -ForegroundColor White
Write-Host "  2. Run: docker stop mqtt-broker" -ForegroundColor White

Write-Host "`nPress any key to exit this window..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
