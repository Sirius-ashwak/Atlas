#!/bin/bash
# Train hybrid DQN-PPO-GNN model

set -e

echo "=========================================="
echo "Training Hybrid DQN-PPO-GNN Model"
echo "=========================================="

# Parse arguments
TIMESTEPS=${1:-100000}
EVAL_FREQ=${2:-5000}
SEED=${3:-42}

echo ""
echo "Configuration:"
echo "  Total timesteps: $TIMESTEPS"
echo "  Eval frequency:  $EVAL_FREQ"
echo "  Random seed:     $SEED"
echo ""

# Train hybrid model
python -m src.main train-hybrid \
    --env-config configs/env_config.yaml \
    --hybrid-config configs/hybrid_config.yaml \
    --timesteps $TIMESTEPS \
    --eval-freq $EVAL_FREQ \
    --n-eval 10 \
    --log-dir logs/hybrid \
    --model-dir models/hybrid \
    --seed $SEED

echo ""
echo "=========================================="
echo "Hybrid training complete!"
echo "=========================================="
echo ""
echo "To monitor training in real-time:"
echo "  tensorboard --logdir logs/hybrid"
echo ""
echo "Model saved to:"
echo "  models/hybrid/final_model.pt"
echo ""
echo "To evaluate the model:"
echo "  python -m src.main evaluate \\"
echo "    --model-type hybrid \\"
echo "    --model-path models/hybrid/final_model.pt \\"
echo "    --n-eval 100"
