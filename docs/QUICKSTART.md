# 🚀 Quick Start Guide

Get up and running with AI Edge Allocator in 5 minutes.

## ⚡ Fast Track (For Impatient Users)

```powershell
# 1. Clone and install
git clone https://github.com/Sirius-ashwak/DeepSea-IoT.git
cd DeepSea-IoT/ai_edge_allocator
pip install -r requirements.txt

# 2. Generate data
python -m src.main prepare-data

# 3. Train hybrid model
python -m src.main train-hybrid --timesteps 10000

# 4. Start API and Dashboard
python python_scripts/api/run_api.py                          # Terminal 1
python -m streamlit run python_scripts/dashboard/dashboard_app.py   # Terminal 2
```

**Done!** 🎉 Visit:
- Dashboard: http://localhost:8501
- API Docs: http://localhost:8000/docs

---

## 📋 Step-by-Step

### 1. Prerequisites

- Python 3.11+
- 4GB+ RAM
- ~500MB disk space

### 2. Installation

```powershell
pip install -r requirements.txt
```

### 3. Generate Mock Data

```powershell
python -m src.main prepare-data
```

### 4. Train Models

**Quick (10K steps, ~20 min):**
```powershell
python -m src.main train-hybrid --timesteps 10000
```

**Full (20K steps, ~40 min):**
```powershell
python -m src.main train-hybrid --timesteps 20000
```

### 5. Use Trained Models

**Option A: Python API**
```python
from src.agent.hybrid_trainer import HybridTrainer
import torch

checkpoint = torch.load('models/hybrid/best_model.pt')
policy = checkpoint['policy']
```

**Option B: REST API**
```powershell
# Start server
python python_scripts/api/run_api.py

# Make prediction
curl -X POST http://localhost:8000/predict ...
```

**Option C: Dashboard**
```powershell
python -m streamlit run python_scripts/dashboard/dashboard_app.py
# Open http://localhost:8501
```

---

## 🐳 Docker Quick Start

```powershell
docker compose up -d
```

Access:
- API: http://localhost:8000
- Dashboard: http://localhost:8501

---

## 🎯 What's Next?

- **Experiment**: See [Phase 3 Guide](PHASE3_GUIDE.md)
- **Deploy**: See [Phase 4 Summary](PHASE4_SUMMARY.md)
- **API Details**: See [API Guide](API_GUIDE.md)
- **Dashboard**: See [Dashboard Guide](DASHBOARD_GUIDE.md)

---

## 🆘 Issues?

Check [Troubleshooting](TROUBLESHOOTING.md) or [FAQ](FAQ.md)
