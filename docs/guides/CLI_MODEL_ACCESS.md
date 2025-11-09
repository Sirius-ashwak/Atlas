# 🤖 CLI Model Access Guide (Like Hugging Face)

## Yes! Your Models Work Like Hugging Face Models! ✅

Just like Hugging Face models, you can:
1. ✅ **Download/Train models** and save them locally
2. ✅ **Access via Command Line (CLI)** without web interface
3. ✅ **Load and run inference** directly from terminal
4. ✅ **Test and analyze** model performance via Python scripts

---

## 📦 Your Models Are Stored Locally

### Model Storage Location:
```
ai_edge_allocator/
├── models/
│   ├── dqn/
│   │   ├── best_model/
│   │   │   └── best_model.zip          ← DQN model (like .bin in HF)
│   │   └── final_model.zip
│   ├── ppo/
│   │   ├── best_model/
│   │   │   └── best_model.zip          ← PPO model
│   │   └── final_model.zip
│   └── hybrid/
│       ├── best_model.pt               ← Hybrid model (PyTorch .pt file)
│       ├── checkpoint_step_5000.pt
│       └── final_model_step_20000.pt
```

**Just like Hugging Face:**
- Your models are **saved locally** (`.zip` for DQN/PPO, `.pt` for Hybrid)
- You can **copy, share, or upload** these files
- You can **load them anytime** without internet

---

## 🖥️ CLI Access Methods

### Method 1: Direct Python Script (Simplest - Like HF CLI)

```powershell
# Test a single model prediction
python python_scripts/testing/test_prediction.py

# Test all models at once
python python_scripts/testing/test_all_models.py

# Run local inference (no API needed)
python python_scripts/inference/local_inference.py --model-type hybrid
```

### Method 2: Interactive Python (Like HF Transformers)

```powershell
# Start Python interactive mode
python

# Then in Python:
>>> import torch
>>> from src.agent.hybrid_trainer import HybridTrainer
>>> 
>>> # Load model (like loading from HF)
>>> model = torch.load('models/hybrid/best_model.pt')
>>> print(model)
>>> 
>>> # Run prediction
>>> # ... your inference code ...
```

### Method 3: API Client (Like HF Inference API)

```powershell
# Using the test client
python src/api/test_client.py

# Or direct HTTP requests
curl http://localhost:8000/models
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @test_data.json
```

---

## 🚀 Command Line Examples (Step by Step)

### Example 1: Load and Test Hybrid Model

```powershell
# Navigate to project
cd "C:\Users\mohamed\OneDrive\Documents\LEARN\GOGLE DEV\Windsurf\IOT\ai_edge_allocator"

# Activate virtual environment
venv\Scripts\activate

# Run inference with Hybrid model
python python_scripts/inference/local_inference.py --model-type hybrid

# Output:
# 🔧 Initializing hybrid model...
# ✅ Model loaded successfully!
# 📊 Running inference...
# Selected Node: fog_3
# Confidence: 0.87
```

### Example 2: Test All Models Performance

```powershell
# Make sure API is running first
python python_scripts/api/run_api.py --port 8000

# In another terminal, test all models
python python_scripts/testing/test_all_models.py

# Output:
# ============================================================
# TESTING ALL MODELS
# ============================================================
# 
# 📊 Testing DQN model...
#   ✅ Success!
#   - Selected Node: 1
#   - Confidence: 0.8234
#   - API Response Time: 45.23ms
# 
# 📊 Testing PPO model...
#   ✅ Success!
#   - Selected Node: 2
#   - Confidence: 0.7891
# 
# 📊 Testing HYBRID model...
#   ✅ Success!
#   - Selected Node: 1
#   - Confidence: 0.9123
```

### Example 3: Single Prediction Test

```powershell
# Test specific model
python python_scripts/testing/test_prediction.py

# Output:
# ✅ Prediction successful!
# Selected Node: 1
# Confidence: 0.9123
# Processing Time: 23.45ms
# 
# Node Scores:
#   Node 0: 0.1234
#   Node 1: 0.9123  ← Best
#   Node 2: 0.4567
```

### Example 4: Interactive Python Session

```powershell
# Start Python
python

# Then type:
>>> import requests
>>> 
>>> # List available models (like HF model hub)
>>> response = requests.get("http://localhost:8000/models")
>>> print(response.json())
{'dqn': True, 'ppo': True, 'hybrid': True, 'hybrid_gat': True}
>>> 
>>> # Get model info (like HF model card)
>>> response = requests.get("http://localhost:8000/models/hybrid")
>>> print(response.json())
{
  'name': 'hybrid',
  'type': 'Hybrid DQN-PPO-GNN',
  'description': 'Best performing model',
  'performance': {'mean_reward': 273.16, 'std_reward': 8.12}
}
>>> 
>>> # Run prediction
>>> data = {
...     "model_type": "hybrid",
...     "network_state": {...}
... }
>>> response = requests.post("http://localhost:8000/predict", json=data)
>>> print(response.json()['selected_node'])
1
```

---

## 📊 Model Analysis via CLI

### Check Model Performance

```powershell
# View training logs
tensorboard --logdir logs/

# Or use Python to read logs
python -c "import json; print(json.load(open('reports/phase3_gat_summary.json')))"
```

### Benchmark Model Speed

```powershell
python python_scripts/inference/benchmark_inference.py --model hybrid --iterations 100

# Output:
# 🔬 Benchmarking hybrid model...
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Average Inference Time: 23.45ms
# Min: 18.23ms, Max: 34.67ms
# Throughput: 42.66 predictions/sec
```

### Compare All Models

```powershell
python -m src.main evaluate --model-type all --n-eval 100

# Output:
# Evaluating DQN...
#   Mean Reward: 244.15 ± 9.20
# 
# Evaluating PPO...
#   Mean Reward: 241.87 ± 11.84
# 
# Evaluating Hybrid...
#   Mean Reward: 273.16 ± 8.12  ← Best!
```

---

## 🎯 Comparison with Hugging Face

| Feature | Hugging Face | Your Models |
|---------|--------------|-------------|
| **Model Storage** | Local cache (~/.cache/huggingface) | Local folder (models/) ✅ |
| **CLI Access** | `transformers-cli` | `python python_scripts/inference/local_inference.py` ✅ |
| **Python API** | `from transformers import ...` | `from src.agent import ...` ✅ |
| **Load Model** | `model = AutoModel.from_pretrained()` | `model = torch.load()` ✅ |
| **Inference** | `model(inputs)` | `model.predict(obs)` ✅ |
| **Model Info** | Model card on hub | API endpoint `/models/{name}` ✅ |
| **Testing** | `model.eval()` | `python_scripts/testing/test_prediction.py` ✅ |
| **Benchmarking** | Custom scripts | `python_scripts/inference/benchmark_inference.py` ✅ |

---

## 🔧 Available CLI Scripts

### 1. **python_scripts/inference/local_inference.py** - Direct Model Loading
Like: `python -m transformers.pipelines ...`

```powershell
python python_scripts/inference/local_inference.py --model-type hybrid
python python_scripts/inference/local_inference.py --model-type dqn --model-path models/dqn/best_model/best_model.zip
```

### 2. **python_scripts/testing/test_prediction.py** - Single Test
Like: HF inference API test

```powershell
python python_scripts/testing/test_prediction.py
```

### 3. **python_scripts/testing/test_all_models.py** - Batch Testing
Like: Testing multiple HF models

```powershell
python python_scripts/testing/test_all_models.py
```

### 4. **python_scripts/inference/benchmark_inference.py** - Performance Test
Like: HF benchmark scripts

```powershell
python python_scripts/inference/benchmark_inference.py --model hybrid --iterations 1000
```

### 5. **python_scripts/testing/test_mock_network.py** - Generate Test Data
Like: HF datasets generation

```powershell
python python_scripts/testing/test_mock_network.py --num-nodes 20
```

---

## 💡 Real-World Usage Examples

### Example 1: Research Paper Analysis

```powershell
# Train model
python python_scripts/training/run_phase3_gat.py

# Evaluate on test set
python -m src.main evaluate --model-type hybrid --n-eval 1000

# Generate plots
python python_scripts/analysis/run_analysis.py

# Export results
python -c "
import json
results = json.load(open('reports/phase3_gat_summary.json'))
print(f'Reward: {results[\"performance\"][\"mean_reward\"]}')
"
```

### Example 2: Production Deployment Testing

```powershell
# Load model without API
python python_scripts/inference/local_inference.py --model-type hybrid

# Benchmark performance
python python_scripts/inference/benchmark_inference.py --model hybrid --iterations 10000

# Test with real network data
python python_scripts/testing/test_prediction.py --data-file real_network_data.json
```

### Example 3: Model Comparison Study

```powershell
# Test all models
python python_scripts/testing/test_all_models.py > results.txt

# Compare performance
python -m src.main experiment --methods dqn ppo hybrid --timesteps 100000

# Analyze results
jupyter notebook notebooks/eda.ipynb
```

---

## 📦 Model Files Explanation

### DQN/PPO Models (.zip files)
Like: Hugging Face `.bin` files

```python
# Load DQN model
from stable_baselines3 import DQN
model = DQN.load("models/dqn/best_model/best_model.zip")

# Run prediction
obs = env.reset()
action, _states = model.predict(obs)
print(f"Selected node: {action}")
```

### Hybrid Models (.pt files)
Like: PyTorch `.pt` checkpoint files

```python
# Load Hybrid model
import torch
checkpoint = torch.load("models/hybrid/best_model.pt")
model = checkpoint['model']

# Run prediction
with torch.no_grad():
    output = model(input_tensor)
    action = output.argmax()
print(f"Selected node: {action}")
```

---

## 🎯 Quick Command Reference

### Load Models
```powershell
# Hybrid (best performance)
python python_scripts/inference/local_inference.py --model-type hybrid

# DQN
python python_scripts/inference/local_inference.py --model-type dqn

# PPO
python python_scripts/inference/local_inference.py --model-type ppo
```

### Test Models
```powershell
# Test single model
python python_scripts/testing/test_prediction.py

# Test all models
python python_scripts/testing/test_all_models.py

# Test with custom data
python python_scripts/testing/test_prediction.py --data custom_network.json
```

### Analyze Models
```powershell
# View model info
curl http://localhost:8000/models/hybrid

# List all models
curl http://localhost:8000/models

# Get performance metrics
curl http://localhost:8000/metrics
```

### Benchmark Models
```powershell
# Speed test
python python_scripts/inference/benchmark_inference.py --model hybrid --iterations 1000

# Accuracy test
python -m src.main evaluate --model-type hybrid --n-eval 100
```

---

## ✅ Summary

**YES! Your models work exactly like Hugging Face models:**

1. ✅ **Stored Locally** - Models are in `models/` folder
2. ✅ **CLI Access** - Use Python scripts to load and test
3. ✅ **No Internet Needed** - Everything runs offline
4. ✅ **Easy to Share** - Copy model files like HF models
5. ✅ **Multiple Interfaces** - CLI, Python API, Web UI, REST API

**Main Difference:**
- **Hugging Face**: `transformers-cli` and `AutoModel.from_pretrained()`
- **Your Models**: `python python_scripts/inference/local_inference.py` and `torch.load()`

**Both give you full control over your models via command line!** 🚀

---

## 🚀 Try It Now!

```powershell
# 1. Load and test hybrid model
python python_scripts/inference/local_inference.py --model-type hybrid

# 2. Test all models
python python_scripts/testing/test_all_models.py

# 3. Check model info
python -c "import requests; print(requests.get('http://localhost:8000/models').json())"
```

**You have full CLI access to your models, just like Hugging Face!** 🎉
