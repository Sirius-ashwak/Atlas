#!/bin/bash
# Train baseline RL models (DQN and PPO)

set -e

echo "=========================================="
echo "Training Baseline Models"
echo "=========================================="

# Parse arguments
MODEL=${1:-"both"}  # "dqn", "ppo", or "both"
TIMESTEPS=${2:-100000}
SEED=${3:-42}

# Train DQN
if [ "$MODEL" = "dqn" ] || [ "$MODEL" = "both" ]; then
    echo ""
    echo ">>> Training DQN Baseline..."
    echo ""
    
    python -m src.main train-dqn \
        --env-config configs/env_config.yaml \
        --hybrid-config configs/hybrid_config.yaml \
        --timesteps $TIMESTEPS \
        --seed $SEED
    
    echo ""
    echo "DQN training complete!"
fi

# Train PPO
if [ "$MODEL" = "ppo" ] || [ "$MODEL" = "both" ]; then
    echo ""
    echo ">>> Training PPO Baseline..."
    echo ""
    
    python -m src.main train-ppo \
        --env-config configs/env_config.yaml \
        --hybrid-config configs/hybrid_config.yaml \
        --timesteps $TIMESTEPS \
        --seed $SEED
    
    echo ""
    echo "PPO training complete!"
fi

echo ""
echo "=========================================="
echo "Baseline training finished!"
echo "=========================================="
echo ""
echo "To view training progress:"
echo "  tensorboard --logdir logs/"
echo ""
echo "Models saved to:"
echo "  models/dqn/"
echo "  models/ppo/"
