# Train baseline RL models (DQN and PPO) - PowerShell version

param(
    [string]$Model = "both",  # "dqn", "ppo", or "both"
    [int]$Timesteps = 100000,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Training Baseline Models" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Train DQN
if ($Model -eq "dqn" -or $Model -eq "both") {
    Write-Host ""
    Write-Host ">>> Training DQN Baseline..." -ForegroundColor Yellow
    Write-Host ""
    
    python -m src.main train-dqn `
        --env-config configs/env_config.yaml `
        --hybrid-config configs/hybrid_config.yaml `
        --timesteps $Timesteps `
        --seed $Seed
    
    Write-Host ""
    Write-Host "DQN training complete!" -ForegroundColor Green
}

# Train PPO
if ($Model -eq "ppo" -or $Model -eq "both") {
    Write-Host ""
    Write-Host ">>> Training PPO Baseline..." -ForegroundColor Yellow
    Write-Host ""
    
    python -m src.main train-ppo `
        --env-config configs/env_config.yaml `
        --hybrid-config configs/hybrid_config.yaml `
        --timesteps $Timesteps `
        --seed $Seed
    
    Write-Host ""
    Write-Host "PPO training complete!" -ForegroundColor Green
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Baseline training finished!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To view training progress:" -ForegroundColor Yellow
Write-Host "  tensorboard --logdir logs/" -ForegroundColor White
Write-Host ""
Write-Host "Models saved to:" -ForegroundColor Yellow
Write-Host "  models/dqn/" -ForegroundColor White
Write-Host "  models/ppo/" -ForegroundColor White
