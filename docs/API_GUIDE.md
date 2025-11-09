# 🚀 API Guide: AI Edge Allocator

Complete guide for using the FastAPI inference server.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Running the Server](#running-the-server)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Client SDKs](#client-sdks)
- [Production Deployment](#production-deployment)

---

## 🏁 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Start the Server

```bash
python python_scripts/api/run_api.py
```

The server will start at `http://localhost:8000`

### 3. Test the API

```bash
# In another terminal
python -m src.api.test_client
```

### 4. View Documentation

Open your browser to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📦 Installation

### Option 1: Pip Install

```bash
# Install API dependencies
pip install fastapi uvicorn[standard] pydantic

# Or use requirements file
pip install -r requirements_api.txt
```

### Option 2: With Virtual Environment

```bash
python -m venv venv_api
source venv_api/bin/activate  # On Windows: venv_api\Scripts\activate
pip install -r requirements_api.txt
```

---

## 🚀 Running the Server

### Development Mode (with auto-reload)

```bash
python python_scripts/api/run_api.py --reload
```

### Production Mode

```bash
python python_scripts/api/run_api.py --host 0.0.0.0 --port 8000 --workers 4
```

### Custom Configuration

```bash
python python_scripts/api/run_api.py \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 4 \
    --log-level info
```

### Using Uvicorn Directly

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔌 API Endpoints

### Root Endpoint

**GET /** - API information

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "service": "AI Edge Allocator API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "health": "/health"
}
```

---

### Health Check

**GET /health** - Server health status

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": {
    "dqn": true,
    "ppo": true,
    "hybrid": true,
    "hybrid_gat": false,
    "hybrid_attention": false
  },
  "uptime_seconds": 3600.5
}
```

---

### Single Prediction

**POST /predict** - Get optimal node placement for a task

**Request Body:**
```json
{
  "network_state": {
    "nodes": [
      {
        "cpu_util": 0.3,
        "mem_util": 0.5,
        "energy": 40.0,
        "latency": 10.0,
        "bandwidth": 150.0,
        "queue_len": 2.0,
        "node_type": 0
      },
      {
        "cpu_util": 0.6,
        "mem_util": 0.7,
        "energy": 80.0,
        "latency": 5.0,
        "bandwidth": 200.0,
        "queue_len": 5.0,
        "node_type": 1
      }
    ],
    "edges": [[0, 1]]
  },
  "model_type": "hybrid"
}
```

**Response:**
```json
{
  "selected_node": 1,
  "confidence": 0.87,
  "q_values": [0.45, 0.87],
  "node_scores": {
    "0": 0.45,
    "1": 0.87
  },
  "processing_time_ms": 12.5
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "network_state": {
      "nodes": [
        {"cpu_util": 0.3, "mem_util": 0.5, "energy": 40.0, 
         "latency": 10.0, "bandwidth": 150.0, "queue_len": 2.0, "node_type": 0}
      ],
      "edges": []
    },
    "model_type": "hybrid"
  }'
```

---

### Batch Prediction

**POST /batch-predict** - Predict for multiple network states

**Request Body:**
```json
{
  "network_states": [
    {
      "nodes": [...],
      "edges": [...]
    },
    {
      "nodes": [...],
      "edges": [...]
    }
  ],
  "model_type": "hybrid"
}
```

**Response:**
```json
{
  "predictions": [
    {
      "selected_node": 1,
      "confidence": 0.87,
      "node_scores": {...},
      "processing_time_ms": 10.2
    },
    ...
  ],
  "total_processing_time_ms": 45.8
}
```

---

### List Models

**GET /models** - List all available models

```bash
curl http://localhost:8000/models
```

**Response:**
```json
{
  "dqn": true,
  "ppo": true,
  "hybrid": true,
  "hybrid_gat": false,
  "hybrid_attention": false
}
```

---

### Get Model Info

**GET /models/{model_type}** - Get detailed model information

```bash
curl http://localhost:8000/models/hybrid
```

**Response:**
```json
{
  "model_type": "hybrid",
  "model_path": "models/hybrid/best_model.pt",
  "loaded": true,
  "architecture": {
    "gnn_type": "GCN",
    "hidden_dim": 64,
    "num_layers": 3,
    "fusion_strategy": "weighted_sum"
  },
  "training_info": {
    "mean_reward": 246.02,
    "std_reward": 8.57,
    "training_steps": 5000
  }
}
```

---

### Load Model

**POST /models/{model_type}/load** - Load a specific model

```bash
curl -X POST http://localhost:8000/models/hybrid_gat/load
```

**Response:**
```json
{
  "status": "success",
  "message": "Model 'hybrid_gat' loaded successfully"
}
```

---

## 💻 Usage Examples

### Python with Requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make prediction
payload = {
    "network_state": {
        "nodes": [
            {
                "cpu_util": 0.3,
                "mem_util": 0.5,
                "energy": 40.0,
                "latency": 10.0,
                "bandwidth": 150.0,
                "queue_len": 2.0,
                "node_type": 0
            }
        ],
        "edges": []
    },
    "model_type": "hybrid"
}

response = requests.post("http://localhost:8000/predict", json=payload)
result = response.json()
print(f"Selected Node: {result['selected_node']}")
print(f"Confidence: {result['confidence']}")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

async function predict() {
  const payload = {
    network_state: {
      nodes: [
        {
          cpu_util: 0.3,
          mem_util: 0.5,
          energy: 40.0,
          latency: 10.0,
          bandwidth: 150.0,
          queue_len: 2.0,
          node_type: 0
        }
      ],
      edges: []
    },
    model_type: 'hybrid'
  };

  const response = await axios.post('http://localhost:8000/predict', payload);
  console.log('Selected Node:', response.data.selected_node);
  console.log('Confidence:', response.data.confidence);
}

predict();
```

### Using the Python Client

```python
from src.api.test_client import EdgeAllocatorClient

# Create client
client = EdgeAllocatorClient("http://localhost:8000")

# Health check
health = client.health_check()
print(health)

# Make prediction
nodes = [
    {
        "cpu_util": 0.3,
        "mem_util": 0.5,
        "energy": 40.0,
        "latency": 10.0,
        "bandwidth": 150.0,
        "queue_len": 2.0,
        "node_type": 0
    }
]
edges = []

result = client.predict(nodes, edges, model_type="hybrid")
print(f"Selected Node: {result['selected_node']}")
```

---

## 📝 Request Schema

### Node Features

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `cpu_util` | float | 0-1 | CPU utilization |
| `mem_util` | float | 0-1 | Memory utilization |
| `energy` | float | ≥0 | Energy consumption (Joules) |
| `latency` | float | ≥0 | Network latency (ms) |
| `bandwidth` | float | ≥0 | Available bandwidth (Mbps) |
| `queue_len` | float | ≥0 | Task queue length |
| `node_type` | int | 0-2 | Node type (0=sensor, 1=fog, 2=cloud) |

### Model Types

- `dqn` - Deep Q-Network
- `ppo` - Proximal Policy Optimization
- `hybrid` - Hybrid DQN-PPO-GNN
- `hybrid_gat` - Hybrid with GAT encoder
- `hybrid_attention` - Hybrid with attention fusion

---

## 🐳 Production Deployment

### Using Gunicorn

```bash
gunicorn src.api.main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Using Docker (see Dockerfile in next section)

```bash
docker build -t edge-allocator-api .
docker run -p 8000:8000 edge-allocator-api
```

### Environment Variables

```bash
export API_HOST=0.0.0.0
export API_PORT=8000
export MODEL_DIR=./models
export LOG_LEVEL=info
```

---

## 🔒 Security Considerations

### Production Checklist

- [ ] Configure CORS properly (don't use `allow_origins=["*"]`)
- [ ] Add API authentication (JWT tokens, API keys)
- [ ] Enable HTTPS (use reverse proxy like Nginx)
- [ ] Implement rate limiting
- [ ] Add request validation
- [ ] Set up monitoring and logging
- [ ] Use environment variables for sensitive config

### Example: Adding API Key Auth

```python
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY = "your-secret-api-key"
api_key_header = APIKeyHeader(name="X-API-Key")

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Add to endpoints
@app.post("/predict", dependencies=[Security(get_api_key)])
async def predict(request: PredictionRequest):
    ...
```

---

## 🐛 Troubleshooting

### Issue: Models not loading

**Solution:** Check model paths exist:
```bash
ls -la models/
```

Ensure models are trained:
```bash
python -m src.main train-hybrid --timesteps 10000
```

### Issue: Port already in use

**Solution:** Use a different port:
```bash
python python_scripts/api/run_api.py --port 8080
```

Or kill the process using the port:
```bash
# Find process
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Kill process
kill <PID>  # Linux/Mac
taskkill /PID <PID> /F  # Windows
```

### Issue: Import errors

**Solution:** Install dependencies:
```bash
pip install -r requirements_api.txt
```

---

## 📚 Additional Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Source Code**: `src/api/`

---

**Need Help?** Open an issue on GitHub or check the main README.md

Happy Deploying! 🚀
