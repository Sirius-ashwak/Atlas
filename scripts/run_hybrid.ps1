# Train hybrid DQN-PPO-GNN model - PowerShell version

param(
    [int]$Timesteps = 100000,
    [int]$EvalFreq = 5000,
    [int]$Seed = 42
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Training Hybrid DQN-PPO-GNN Model" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

Write-Host ""
Write-Host "Configuration:" -ForegroundColor Yellow
Write-Host "  Total timesteps: $Timesteps" -ForegroundColor White
Write-Host "  Eval frequency:  $EvalFreq" -ForegroundColor White
Write-Host "  Random seed:     $Seed" -ForegroundColor White
Write-Host ""

# Train hybrid model
python -m src.main train-hybrid `
    --env-config configs/env_config.yaml `
    --hybrid-config configs/hybrid_config.yaml `
    --timesteps $Timesteps `
    --eval-freq $EvalFreq `
    --n-eval 10 `
    --log-dir logs/hybrid `
    --model-dir models/hybrid `
    --seed $Seed

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Hybrid training complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To monitor training in real-time:" -ForegroundColor Yellow
Write-Host "  tensorboard --logdir logs/hybrid" -ForegroundColor White
Write-Host ""
Write-Host "Model saved to:" -ForegroundColor Yellow
Write-Host "  models/hybrid/final_model.pt" -ForegroundColor White
Write-Host ""
Write-Host "To evaluate the model:" -ForegroundColor Yellow
Write-Host "  python -m src.main evaluate ``" -ForegroundColor White
Write-Host "    --model-type hybrid ``" -ForegroundColor White
Write-Host "    --model-path models/hybrid/final_model.pt ``" -ForegroundColor White
Write-Host "    --n-eval 100" -ForegroundColor White
