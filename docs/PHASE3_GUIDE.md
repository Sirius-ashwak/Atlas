# 🔬 Phase 3: Research & Experimentation Guide

## 🎯 Overview

Phase 3 focuses on improving your models through:
- Advanced GNN architectures (GAT, GraphSAGE)
- Attention-based fusion mechanisms
- Hyperparameter optimization
- Ablation studies

## 📚 What's New

### 1. Advanced GNN Encoders (`src/gnn/advanced_encoders.py`)

**GAT (Graph Attention Networks)**
- Uses attention to weight neighbor importance
- Multi-head attention for robust learning
- Better for heterogeneous graphs

**GraphSAGE**
- Scalable neighborhood aggregation
- Multiple aggregator types (mean, max, LSTM)
- Efficient for large graphs

**Hybrid GNN**
- Combines GCN + GAT + GraphSAGE
- Ensemble approach for robustness
- Fusion layer merges outputs

**Attention Fusion**
- Learns dynamic weights for DQN/PPO outputs
- Replaces fixed weighted_sum
- Adapts to different scenarios

### 2. Hyperparameter Optimization (`src/experiments/hyperparameter_tuning.py`)

**Automated tuning using Optuna:**
- GNN architecture (hidden_dim, num_layers, conv_type)
- Fusion strategy (weighted_sum vs attention)
- Learning rates for DQN and PPO
- Buffer sizes, batch sizes, gamma values

**Modes:**
- **Quick**: 10 trials (~30 min)
- **Full**: 50 trials (~2-3 hours)

### 3. Ablation Studies (`src/experiments/ablation_study.py`)

**Systematic testing of:**
- GNN encoder impact (GCN vs GAT vs GraphSAGE)
- Fusion strategy impact (weighted_sum vs attention)
- Model components (DQN-only vs PPO-only vs Hybrid)

**Output:**
- CSV with detailed results
- Visualization comparing variants
- Statistical analysis across seeds

## 🚀 Quick Start

### Option 1: Interactive Mode

```bash
python python_scripts/training/run_phase3.py --interactive
```

**Menu:**
```
1. Test Advanced GNN Encoders
2. Hyperparameter Optimization (Quick)
3. Hyperparameter Optimization (Full)
4. Ablation Study
5. Train with GAT Encoder
6. Train with Attention Fusion
7. Run All Experiments
```

### Option 2: Command Line

```bash
# Test encoders
python python_scripts/training/run_phase3.py --experiment test-encoders

# Quick hyperparameter tuning (10 trials)
python python_scripts/training/run_phase3.py --experiment tune-quick

# Full hyperparameter tuning (50 trials)
python python_scripts/training/run_phase3.py --experiment tune-full

# Run ablation study
python python_scripts/training/run_phase3.py --experiment ablation

# Train with GAT
python python_scripts/training/run_phase3.py --experiment train-gat

# Train with attention fusion
python python_scripts/training/run_phase3.py --experiment train-attention

# Run everything
python python_scripts/training/run_phase3.py --experiment all
```

### Option 3: Direct Script Execution

```bash
# Test encoders only
cd src/gnn
python advanced_encoders.py

# Hyperparameter tuning
cd src/experiments
python hyperparameter_tuning.py --mode quick

# Ablation study
python ablation_study.py
```

## 📊 Expected Outputs

### Hyperparameter Optimization
```
reports/optimization/
├── best_params.yaml              # Optimal hyperparameters
├── optimization_history.png      # Progress over trials
└── param_importances.png         # Which params matter most
```

### Ablation Study
```
reports/ablation/
├── ablation_results.csv          # Detailed results
└── ablation_comparison.png       # Visual comparison
```

### Trained Models
```
models/
├── hybrid_gat/                   # GAT-based hybrid
│   ├── best_model.pt
│   └── final_model_step_*.pt
├── hybrid_attention/             # Attention fusion hybrid
│   └── ...
└── optuna/                       # Hyperparameter tuning trials
    └── trial_*/
```

## 🧪 Experiment Examples

### Example 1: Compare GNN Architectures

```python
from src.experiments.ablation_study import AblationStudy
import yaml

# Load configs
with open('configs/env_config.yaml', 'r') as f:
    env_cfg = yaml.safe_load(f)['environment']

with open('configs/hybrid_config.yaml', 'r') as f:
    hybrid_cfg = yaml.safe_load(f)['hybrid']

# Run GNN comparison only
study = AblationStudy(env_cfg, hybrid_cfg, n_timesteps=10000, n_seeds=3)
study.test_gnn_architectures()
```

### Example 2: Train with GAT and Attention

```python
from src.agent.hybrid_trainer import HybridTrainer
import yaml

# Load configs
with open('configs/env_config.yaml', 'r') as f:
    env_cfg = yaml.safe_load(f)['environment']

with open('configs/hybrid_config.yaml', 'r') as f:
    hybrid_cfg = yaml.safe_load(f)['hybrid']

# Modify config
hybrid_cfg['architecture']['gnn_conv_type'] = 'GAT'
hybrid_cfg['fusion']['strategy'] = 'attention'

# Train
trainer = HybridTrainer(env_cfg, hybrid_cfg, 
                       log_dir="logs/gat_attention",
                       model_dir="models/gat_attention")
trainer.train(total_timesteps=20000)
metrics = trainer.evaluate(n_episodes=50)

print(f"Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
```

### Example 3: Quick Hyperparameter Tuning

```python
from src.experiments.hyperparameter_tuning import HyperparameterTuner
import yaml

with open('configs/env_config.yaml', 'r') as f:
    env_cfg = yaml.safe_load(f)['environment']

tuner = HyperparameterTuner(
    env_config=env_cfg,
    n_trials=10,
    n_timesteps=5000,
    study_name="quick_test"
)

study = tuner.optimize()
print(f"Best params: {study.best_params}")
print(f"Best reward: {study.best_value:.2f}")
```

## 📈 Analyzing Results

### Load and Compare Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load ablation results
df = pd.read_csv('reports/ablation/ablation_results.csv')

# Calculate mean across seeds
summary = df.groupby(['experiment', 'variant']).agg({
    'mean_reward': ['mean', 'std']
}).round(2)

print(summary)

# Plot
for exp in df['experiment'].unique():
    exp_data = df[df['experiment'] == exp]
    exp_data.boxplot(column='mean_reward', by='variant', figsize=(10, 6))
    plt.title(f'{exp} - Performance Comparison')
    plt.ylabel('Mean Reward')
    plt.show()
```

### Load Optimized Hyperparameters

```python
import yaml

with open('reports/optimization/best_params.yaml', 'r') as f:
    best_params = yaml.safe_load(f)

print("Optimal hyperparameters:")
for key, value in best_params.items():
    print(f"  {key}: {value}")
```

## 🎓 Research Questions to Explore

1. **Does GAT outperform GCN?**
   - Run: `python python_scripts/training/run_phase3.py --experiment ablation`
   - Compare GNN architectures

2. **Is attention fusion better than weighted sum?**
   - Run ablation study
   - Check fusion strategy results

3. **What's the optimal learning rate?**
   - Run: `python python_scripts/training/run_phase3.py --experiment tune-quick`
   - Check param_importances.png

4. **How many GNN layers are needed?**
   - Run hyperparameter tuning
   - Analyze layer count vs performance

5. **Does hybrid really help vs single methods?**
   - Run ablation study
   - Compare DQN-only, PPO-only, Hybrid

## 📝 Tips for Running Experiments

### Resource Management
```python
# For quick testing, reduce timesteps
n_timesteps = 5000  # Instead of 20000

# Use fewer seeds for faster results
n_seeds = 2  # Instead of 5

# Quick hyperparameter tuning
n_trials = 10  # Instead of 50
```

### Parallel Execution
```bash
# Run multiple experiments in parallel (different terminals)
# Terminal 1:
python python_scripts/training/run_phase3.py --experiment train-gat

# Terminal 2:
python python_scripts/training/run_phase3.py --experiment train-attention

# Terminal 3:
python python_scripts/training/run_phase3.py --experiment tune-quick
```

### Monitoring Progress
```bash
# Watch TensorBoard during training
tensorboard --logdir logs/

# Check GPU usage (if using GPU)
nvidia-smi -l 1
```

## 🐛 Troubleshooting

### Issue: Optuna not found
```bash
pip install optuna
```

### Issue: PyTorch Geometric errors with GAT
```bash
# Reinstall PyG with correct CUDA version
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Issue: Out of memory
```python
# Reduce batch size in config
'batch_size': 32  # Instead of 64
```

### Issue: Training too slow
```python
# Reduce number of GNN layers
'gnn_num_layers': 2  # Instead of 3

# Use smaller hidden dimension
'gnn_hidden_dim': 32  # Instead of 64
```

## 🎯 Next Steps After Phase 3

Based on your results:

1. **If GAT performs best:**
   - Use GAT as default encoder
   - Update configs
   - Retrain final models

2. **If attention fusion helps:**
   - Make it default strategy
   - Analyze attention weights
   - Visualize learned importance

3. **If you find optimal hyperparameters:**
   - Update configs with best params
   - Train final model with 100K+ steps
   - Publish results

4. **Move to Phase 4: Deployment**
   - Create inference API
   - Build monitoring dashboard
   - Deploy to production

## 📚 Further Reading

- **GAT Paper**: [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
- **GraphSAGE Paper**: [Inductive Representation Learning](https://arxiv.org/abs/1706.02216)
- **Optuna Docs**: [https://optuna.org](https://optuna.org)
- **PyTorch Geometric**: [https://pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io)

---

**Good luck with your experiments!** 🚀

For questions or issues, check the main README or open a GitHub issue.
