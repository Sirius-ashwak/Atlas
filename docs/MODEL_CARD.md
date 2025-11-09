---
license: mit
tags:
- reinforcement-learning
- iot
- edge-computing
- resource-allocation
- graph-neural-network
- dqn
- ppo
- hybrid-model
library_name: pytorch
pipeline_tag: reinforcement-learning
---

# 🌐 AI Edge Allocator: Hybrid DQN-PPO-GNN for IoT Resource Allocation

## Model Description

**AI Edge Allocator** is a reinforcement learning model that optimizes task placement in IoT edge computing environments. It combines Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO) with Graph Neural Networks (GNN) to make intelligent resource allocation decisions.

### Key Features

- **Hybrid Architecture**: Combines value-based (DQN) and policy-based (PPO) approaches
- **Graph-Aware Learning**: Uses PyTorch Geometric GNN to capture network topology
- **Multiple Encoders**: Supports GCN, GAT, GraphSAGE, and Hybrid GNN architectures
- **Production Ready**: Includes FastAPI server and Streamlit dashboard
- **High Performance**: Achieves **246.02 ± 8.57** mean reward

## Model Variants

This repository includes multiple trained models:

| Model | Type | Mean Reward | Std Dev | Description |
|-------|------|-------------|---------|-------------|
| **Hybrid (Best)** | DQN+PPO+GNN | **246.02** | **8.57** | Best performing model |
| DQN Baseline | DQN | 244.15 | 9.20 | Value-based baseline |
| PPO Baseline | PPO | 241.87 | 11.84 | Policy-based baseline |
| Hybrid GAT | DQN+PPO+GAT | TBD | TBD | Graph Attention Network encoder |
| Hybrid Attention | DQN+PPO+GNN | TBD | TBD | Attention fusion mechanism |

## Intended Use

### Direct Use

This model is designed for:
- **IoT Resource Allocation**: Optimize task placement in edge computing networks
- **Research**: Benchmark for graph-based reinforcement learning
- **Education**: Learn hybrid RL architectures with GNN integration

### Out-of-Scope Use

This model is NOT intended for:
- Production deployment without proper testing in your specific environment
- Safety-critical systems without additional validation
- Real-time systems without latency testing

## How to Use

### Installation

```bash
git clone https://github.com/Sirius-ashwak/DeepSea-IoT.git
cd DeepSea-IoT/ai_edge_allocator
pip install -r requirements.txt
```

### Quick Inference

```python
import torch
from src.agent.hybrid_trainer import HybridTrainer
from src.env.iot_env import IoTEnv
from src.utils.data_loader import DataLoader

# Load environment
data_loader = DataLoader()
env = IoTEnv(data_loader, config={})

# Load model
checkpoint = torch.load('models/hybrid/best_model.pt')
policy = checkpoint['policy']

# Make prediction
obs = env.reset()
action, _ = policy.predict(obs, deterministic=True)
print(f"Selected node: {action}")
```

### Using REST API

```bash
# Start API server
python python_scripts/api/run_api.py

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"network_state": {...}, "model_type": "hybrid"}'
```

### Using Dashboard

```bash
# Start dashboard
python -m streamlit run python_scripts/dashboard/dashboard_app.py
# Open http://localhost:8501
```

## Training Procedure

### Training Hyperparameters

**Hybrid Model (Best):**
- **Training Steps**: 5,000
- **DQN Learning Rate**: 0.0003
- **PPO Learning Rate**: 0.0003
- **Batch Size**: 64
- **Gamma (Discount)**: 0.99
- **GNN Hidden Dim**: 64
- **GNN Layers**: 3
- **Fusion Strategy**: Weighted sum (DQN: 0.6, PPO: 0.4)

### Training Data

- **Data Source**: Simulated IoT network traces
- **Network Size**: 17 nodes (sensors, fog, cloud)
- **Episodes**: 50 per evaluation
- **Environment**: Custom Gymnasium environment

### Training Process

1. Pre-train DQN for 10,000 steps
2. Pre-train PPO for 10,000 steps
3. Fine-tune hybrid model for 5,000 steps
4. Evaluate on 100 episodes

## Evaluation Results

### Performance Metrics

| Model | Mean Reward | Std Dev | Min | Max | Training Steps |
|-------|-------------|---------|-----|-----|----------------|
| Hybrid (Best) | **246.02** | **8.57** | - | - | 5,000 |
| DQN Baseline | 244.15 | 9.20 | 211.94 | 255.84 | 10,000 |
| PPO Baseline | 241.87 | 11.84 | 187.48 | 254.61 | 10,000 |
| Hybrid (Final) | 242.64 | 10.12 | 201.43 | 257.14 | 20,000 |

### Key Findings

- ✅ **Hybrid model achieved best performance** with lowest variance
- ✅ **8.57 std dev** - most stable predictions
- ✅ **Significantly outperforms** random baseline (~0 reward)
- ✅ **DQN competitive** with 244.15 mean reward

## Environmental Impact

- **Hardware**: CPU training (no GPU required)
- **Training Time**: ~2 hours for all models
- **Carbon Emissions**: Minimal (CPU-only training)

## Technical Specifications

### Model Architecture

```
Observation (Network State: 17 nodes × 7 features)
         ↓
   GNN Encoder (PyG)
    [3 GCN Layers: 6 → 64 → 64 → 128]
         ↓
   Graph Embedding (128-dim)
       /   \
      /     \
  DQN Head  PPO Head
  [Q-values] [Policy + Value]
      \     /
       \   /
    Fusion Layer
    (Weighted: 0.6 DQN + 0.4 PPO)
         ↓
   Action Selection
   (Node ID: 0-16)
```

### Input Format

```python
{
    "nodes": [
        {
            "cpu_util": float,      # 0-1
            "mem_util": float,      # 0-1
            "energy": float,        # Joules
            "latency": float,       # ms
            "bandwidth": float,     # Mbps
            "queue_len": float,     # tasks
            "node_type": int        # 0=sensor, 1=fog, 2=cloud
        },
        ...
    ],
    "edges": [[source, target], ...]  # Connectivity
}
```

### Output Format

```python
{
    "selected_node": int,           # Node ID (0-16)
    "confidence": float,            # 0-1
    "q_values": [float, ...],       # Optional: Q-values per node
    "processing_time_ms": float     # Inference time
}
```

## Limitations and Biases

### Limitations

- **Simulated Data**: Trained on synthetic IoT network traces
- **Fixed Topology**: Assumes 17-node network structure
- **CPU Only**: Models trained without GPU acceleration
- **No Transfer Learning**: Requires retraining for different topologies

### Potential Biases

- May favor fog nodes over cloud due to training data distribution
- Performance may degrade on networks significantly different from training data
- Assumes reliable network connectivity

## Citation

If you use this model in your research, please cite:

```bibtex
@software{deepsea_iot_2025,
  title={DeepSea-IoT: Hybrid DQN-PPO-GNN for IoT Edge Resource Allocation},
  author={Mohamed Ashwak},
  year={2025},
  url={https://github.com/Sirius-ashwak/DeepSea-IoT},
  version={1.0.0}
}
```

## Model Card Authors

**Mohamed Ashwak** ([@Sirius-ashwak](https://github.com/Sirius-ashwak))

## Model Card Contact

- **GitHub**: [Sirius-ashwak/DeepSea-IoT](https://github.com/Sirius-ashwak/DeepSea-IoT)
- **Issues**: [GitHub Issues](https://github.com/Sirius-ashwak/DeepSea-IoT/issues)

## Additional Resources

- **GitHub Repository**: https://github.com/Sirius-ashwak/DeepSea-IoT
- **Documentation**: See README.md
- **API Guide**: API_GUIDE.md
- **Dashboard Guide**: DASHBOARD_GUIDE.md
- **Docker Guide**: DOCKER_GUIDE.md

## Acknowledgments

Built with:
- **PyTorch** - Deep learning framework
- **Stable-Baselines3** - RL algorithms
- **PyTorch Geometric** - Graph neural networks
- **FastAPI** - API framework
- **Streamlit** - Dashboard framework

## License

MIT License - See [LICENSE](https://github.com/Sirius-ashwak/DeepSea-IoT/blob/main/ai_edge_allocator/LICENSE) for details.

---

**Last Updated**: October 2025  
**Model Version**: 1.0.0  
**Framework**: PyTorch 2.0+, Stable-Baselines3 2.0+
