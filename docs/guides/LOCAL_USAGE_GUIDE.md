# 🏠 Local Model Usage Guide

Complete guide for using your trained AI Edge Allocator models locally on your system.

---

## 🎯 Quick Start (Simplest Method)

### 1. Use the Simple Inference Script

We've created a standalone script for easy local inference:

```bash
# Using default hybrid model
python python_scripts/inference/local_inference.py --model-type hybrid

# Using DQN model
python python_scripts/inference/local_inference.py --model-type dqn

# Using PPO model
python python_scripts/inference/local_inference.py --model-type ppo

# Using custom model path
python python_scripts/inference/local_inference.py --model-type hybrid --model-path models/hybrid/checkpoint_step_5000.pt

# Using custom network state from JSON
python python_scripts/inference/local_inference.py --model-type hybrid --input-json my_network.json
```

**Output Example:**
```
======================================================================
🚀 AI EDGE ALLOCATOR - LOCAL INFERENCE
======================================================================

🔧 Initializing hybrid model...
✅ Loaded Hybrid model from models/hybrid/best_model.pt

🌐 Network State:
   Total Nodes: 4
   - cloud_0: CPU=0.25, Mem=0.18, Latency=30.0ms
   - fog_1: CPU=0.45, Mem=0.38, Latency=15.2ms
   - fog_2: CPU=0.62, Mem=0.55, Latency=12.5ms
   - edge_3: CPU=0.78, Mem=0.71, Latency=5.3ms

🤖 Running HYBRID model inference...

✅ PREDICTION RESULTS:
======================================================================
📍 Selected Node: fog_1 (index: 1)
💯 Confidence: 0.8734
🔧 Model Type: hybrid
======================================================================

💾 Results saved to: prediction_result_hybrid.json
```

---

## 📦 Method 1: Simple Python Script

### Create Your Own Script

```python
from local_inference import LocalModelInference

# Initialize model
model = LocalModelInference(model_type='hybrid')

# Define network state
network_state = {
    'nodes': [
        {
            'node_id': 'fog_1',
            'cpu_util': 0.45,
            'mem_util': 0.38,
            'energy': 120.5,
            'latency': 15.2,
            'bandwidth': 150.0,
            'queue_len': 5,
            'node_type': 1  # 0=cloud, 1=fog, 2=edge
        },
        # ... more nodes
    ]
}

# Make prediction
result = model.predict(network_state)

print(f"Selected Node: {result['selected_node_id']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 🎮 Method 2: Using Python Interactively

### In Python REPL or Jupyter Notebook

```python
import torch
import numpy as np
from stable_baselines3 import DQN, PPO

# ==================== OPTION A: DQN Model ====================
# Load DQN model
dqn_model = DQN.load("models/dqn/best_model/best_model.zip")

# Create observation (flatten node features)
# For 4 nodes with 7 features each = 28 values
observation = np.array([
    # Node 0 (cloud)
    0.25, 0.18, 100.0, 30.0, 200.0, 2.0, 0.0,
    # Node 1 (fog)
    0.45, 0.38, 120.5, 15.2, 150.0, 5.0, 1.0,
    # Node 2 (fog)
    0.62, 0.55, 135.8, 12.5, 140.0, 8.0, 1.0,
    # Node 3 (edge)
    0.78, 0.71, 145.2, 5.3, 100.0, 12.0, 2.0,
], dtype=np.float32)

# Predict action
action, _states = dqn_model.predict(observation, deterministic=True)
print(f"Selected Node: {action}")

# Get Q-values
q_values = dqn_model.q_net(torch.FloatTensor(observation)).detach().numpy()[0]
print(f"Q-values: {q_values}")


# ==================== OPTION B: PPO Model ====================
# Load PPO model
ppo_model = PPO.load("models/ppo/best_model/best_model.zip")

# Predict
action, _states = ppo_model.predict(observation, deterministic=True)
print(f"Selected Node: {action}")


# ==================== OPTION C: Hybrid Model ====================
# Load Hybrid model
checkpoint = torch.load("models/hybrid/best_model.pt", map_location='cpu')
print("Checkpoint keys:", checkpoint.keys())

# Note: Hybrid model needs proper graph structure
# See src/agent/hybrid_trainer.py for full implementation
```

---

## 🔧 Method 3: Using the Existing API

### Start Local API Server

```bash
# Terminal 1: Start API server
python python_scripts/api/run_api.py

# API will be available at http://localhost:8000
```

### Make Requests to API

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/predict"

# Network state
network_state = {
    "nodes": [
        {
            "node_id": "fog_1",
            "cpu_util": 0.45,
            "mem_util": 0.38,
            "energy": 120.5,
            "latency": 15.2,
            "bandwidth": 150.0,
            "queue_len": 5,
            "node_type": 1
        }
    ],
    "model_type": "hybrid"
}

# Make request
response = requests.post(url, json=network_state)
result = response.json()

print(f"Selected Node: {result['selected_node']}")
print(f"Confidence: {result['confidence']}")
```

**Or use curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @example_network_state.json
```

---

## 📊 Method 4: Using the Dashboard

### Start Dashboard Locally

```bash
# Windows PowerShell
.\run_dashboard.ps1

# Or directly
python -m streamlit run python_scripts/dashboard/dashboard_app.py

# Dashboard will open at http://localhost:8501
```

The dashboard provides:
- 🎮 **Interactive UI** for network configuration
- 📊 **Real-time visualization** of network state
- 🤖 **Model selection** (DQN, PPO, Hybrid)
- 📈 **Performance metrics** and confidence scores
- 📥 **Export results** to JSON

---

## 🐍 Method 5: Direct Python Integration

### Load and Use Models Directly

```python
# === Complete Example: Load and Use Model ===

import torch
import numpy as np
from stable_baselines3 import DQN
from pathlib import Path

class SimpleEdgeAllocator:
    """Simple wrapper for model inference."""
    
    def __init__(self, model_path="models/dqn/best_model/best_model.zip"):
        """Load model."""
        self.model = DQN.load(model_path)
        print(f"✅ Model loaded from {model_path}")
    
    def allocate_task(self, nodes_info: list) -> dict:
        """
        Allocate task to best node.
        
        Args:
            nodes_info: List of dicts with node metrics
        
        Returns:
            Dict with selected node and confidence
        """
        # Convert to observation
        obs = []
        for node in nodes_info:
            obs.extend([
                node['cpu_util'],
                node['mem_util'],
                node['energy'],
                node['latency'],
                node['bandwidth'],
                node['queue_len'],
                float(node.get('node_type', 1))
            ])
        obs = np.array(obs, dtype=np.float32)
        
        # Predict
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Get confidence (Q-value)
        q_values = self.model.q_net(torch.FloatTensor(obs)).detach().numpy()[0]
        confidence = float(np.max(q_values))
        
        return {
            'selected_node': int(action),
            'node_id': nodes_info[action]['node_id'],
            'confidence': confidence,
            'all_q_values': q_values.tolist()
        }

# Usage
allocator = SimpleEdgeAllocator()

nodes = [
    {'node_id': 'fog_1', 'cpu_util': 0.4, 'mem_util': 0.3, 
     'energy': 120, 'latency': 15, 'bandwidth': 150, 'queue_len': 5, 'node_type': 1},
    {'node_id': 'fog_2', 'cpu_util': 0.6, 'mem_util': 0.5,
     'energy': 135, 'latency': 12, 'bandwidth': 140, 'queue_len': 8, 'node_type': 1},
]

result = allocator.allocate_task(nodes)
print(f"Deploy to: {result['node_id']} (confidence: {result['confidence']:.2f})")
```

---

## 📥 Method 6: Load from Hugging Face

If you've uploaded your models to Hugging Face:

```python
from huggingface_hub import hf_hub_download
import torch

# Download model from Hugging Face
model_path = hf_hub_download(
    repo_id="your-username/DeepSea-IoT",
    filename="models/hybrid/best_model.pt"
)

# Load model
checkpoint = torch.load(model_path, map_location='cpu')
model = checkpoint['policy']

print("✅ Model loaded from Hugging Face!")
```

---

## 📋 Network State Format

### Required Fields

```json
{
  "nodes": [
    {
      "node_id": "string (unique identifier)",
      "cpu_util": 0.0-1.0,        // CPU utilization (0-100%)
      "mem_util": 0.0-1.0,        // Memory utilization (0-100%)
      "energy": float,            // Energy consumption (Joules)
      "latency": float,           // Network latency (milliseconds)
      "bandwidth": float,         // Available bandwidth (Mbps)
      "queue_len": int,           // Current queue length
      "node_type": 0|1|2          // 0=cloud, 1=fog, 2=edge
    }
  ]
}
```

### Example Files

- **`example_network_state.json`** - Sample 4-node network
- **`data/raw/sim_results.csv`** - Historical simulation data

---

## 🔍 Understanding Model Outputs

### DQN Model Output
```python
{
    'selected_node': 1,              # Node index
    'selected_node_id': 'fog_1',     # Node identifier
    'confidence': 0.8734,            # Max Q-value
    'node_scores': {                 # Q-values for each node
        0: 0.7234,
        1: 0.8734,
        2: 0.6123,
        3: 0.5891
    }
}
```

### PPO Model Output
```python
{
    'selected_node': 1,
    'selected_node_id': 'fog_1',
    'confidence': 0.85,              # Policy confidence
    'model_type': 'ppo'
}
```

### Hybrid Model Output
```python
{
    'selected_node': 1,
    'selected_node_id': 'fog_1',
    'confidence': 0.90,              # Combined DQN+PPO+GNN
    'model_type': 'hybrid',
    'architecture': 'DQN-PPO-GNN'
}
```

---

## ⚙️ Configuration

### Customize Model Behavior

Edit `configs/hybrid_config.yaml`:

```yaml
inference:
  deterministic: true        # Deterministic vs stochastic
  temperature: 1.0           # Softmax temperature (if using)
  top_k: 3                   # Return top-K nodes
  
evaluation:
  confidence_threshold: 0.7  # Minimum confidence
  fallback_strategy: "random"  # Or "round_robin", "least_loaded"
```

---

## 🧪 Testing Your Setup

### Quick Test Script

```python
# test_local_model.py
from local_inference import LocalModelInference, create_sample_network_state

print("🧪 Testing local model inference...\n")

# Test each model type
for model_type in ['dqn', 'ppo', 'hybrid']:
    try:
        print(f"Testing {model_type.upper()}...")
        model = LocalModelInference(model_type=model_type)
        network_state = create_sample_network_state()
        result = model.predict(network_state)
        print(f"✅ {model_type.upper()} works! Selected: {result['selected_node_id']}\n")
    except Exception as e:
        print(f"❌ {model_type.upper()} failed: {e}\n")
```

Run with:
```bash
python test_local_model.py
```

---

## 🐛 Troubleshooting

### Issue: Model file not found

**Solution:**
```bash
# Check if models exist
ls models/dqn/best_model/
ls models/ppo/best_model/
ls models/hybrid/

# If missing, train models first
python -m src.main experiment --methods dqn ppo hybrid --timesteps 10000
```

### Issue: Import errors

**Solution:**
```bash
# Install dependencies
pip install -r requirements.txt

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Out of memory

**Solution:**
```python
# Use CPU instead of GPU
checkpoint = torch.load(model_path, map_location='cpu')

# Or reduce observation size
```

### Issue: Wrong observation shape

**Solution:**
```python
# Ensure observation matches training:
# - Number of nodes must match training config
# - 7 features per node: [cpu, mem, energy, latency, bandwidth, queue, type]
# - Total observation size = num_nodes * 7

# Example for 20 nodes (default training config):
assert len(observation) == 20 * 7  # 140 features
```

---

## 📚 Next Steps

1. ✅ **Test basic inference** with sample data
2. 📊 **Integrate with your IoT system** - Send real network metrics
3. 🔧 **Customize for your network** - Adjust node features
4. 🚀 **Deploy API** for production use
5. 📈 **Monitor performance** with TensorBoard/dashboard

---

## 💡 Tips & Best Practices

### Performance Tips
- ✅ Load model once, reuse for multiple predictions
- ✅ Use `deterministic=True` for production
- ✅ Batch predictions when possible
- ✅ Cache model in memory

### Production Tips
- ✅ Add error handling and logging
- ✅ Implement fallback strategies
- ✅ Monitor prediction latency
- ✅ Set confidence thresholds
- ✅ Validate input data

### Debugging Tips
- ✅ Start with sample data first
- ✅ Check observation shape matches training
- ✅ Verify model checkpoints are complete
- ✅ Use `deterministic=True` for reproducibility

---

## 📖 Related Documentation

- **[README.md](../README.md)** - Main project documentation
- **[QUICKSTART.md](../QUICKSTART.md)** - Complete setup guide
- **[API_GUIDE.md](../API_GUIDE.md)** - REST API reference
- **[DASHBOARD_GUIDE.md](../DASHBOARD_GUIDE.md)** - Dashboard usage
- **[HUGGINGFACE_GUIDE.md](../HUGGINGFACE_GUIDE.md)** - Share your models

---

## 🤝 Need Help?

- 💬 **Open an issue**: [GitHub Issues](https://github.com/Sirius-ashwak/DeepSea-IoT/issues)
- 📧 **Contact**: See README for contact information
- 📚 **Documentation**: Check `docs/` directory

---

**Happy Inferencing!** 🚀
