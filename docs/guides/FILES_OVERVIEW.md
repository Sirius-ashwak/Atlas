# 📦 Project Files Overview

## What You Actually Need

### ✅ Essential Files (Must Use)

```
ai_edge_allocator/
├── SIMPLE_START.md          ⭐ START HERE - 4 commands only
├── README.md                 📚 Main documentation
│
├── web-app/                  🎨 React Dashboard (Modern UI)
│   ├── package.json          - Dependencies
│   ├── src/                  - React components
│   └── ...
│
├── python_scripts/           🧰 Organized command-line tools
│   ├── api/run_api.py        - FastAPI backend launcher
│   ├── analysis/run_analysis.py - Phase 2 analysis suite
│   ├── inference/            - Local inference + benchmarking
│   ├── training/             - Model training utilities
│   ├── testing/              - Quick validation scripts
│   └── ...
├── requirements_api.txt      - API dependencies
│
├── models/                   🤖 Trained ML models
│   ├── dqn/                  - DQN checkpoint
│   ├── ppo/                  - PPO checkpoint
│   └── hybrid/               - Hybrid GAT models
│
└── src/                      🧠 Core ML code
    ├── agent/                - Training agents
    ├── env/                  - RL environments
    └── ...
```

---

## 📝 Documentation Files (Reference Only)

### Main Guides
- `SIMPLE_START.md` ⭐ **USE THIS FIRST** - 4 simple commands
- `README.md` - Complete project overview
- `WEB_APP_GUIDE.md` - Detailed React setup (if you need more info)

### Optional/Advanced Guides (Can Ignore for Basic Setup)
- `QUICKSTART_WEB.md` - Alternative quick start (similar to SIMPLE_START.md)
- `WEB_APP_README.md` - Technical web app details
- `WEB_APP_SUMMARY.md` - What was built
- `SCRIPT_FIXED_README.md` - Troubleshooting scripts
- `PROJECT_100_PERCENT_COMPLETE.md` - Project completion checklist
- `docs/QUICKSTART.md` - ML training quick start
- `docs/PHASE3_GUIDE.md` - Advanced experiments
- All other files in `docs/` folder

---

## 🔧 Setup Scripts (Optional - Can Ignore)

**You don't need any of these scripts!** Just use the manual commands in `SIMPLE_START.md`.

Scripts are provided for automation convenience, but manual setup is **simpler and more reliable**.

### Scripts You Can Ignore:
- ❌ `setup_web_app.ps1` - Automated setup (optional)
- ❌ `setup_web_app_v2.ps1` - Alternative setup (optional)
- ❌ `setup_web_app_old.ps1` - Legacy setup (ignore)
- ❌ `quick_start_web.ps1` - Quick automation (optional)
- ❌ `start_web_app.ps1` - Start automation (optional)
- ❌ `start_all_services.ps1` - Multi-service automation (optional)
- ❌ `run_dashboard.ps1` - Streamlit dashboard (legacy)
- ❌ `setup_project.ps1` - Project setup automation (optional)

**Bottom line:** Use manual commands from `SIMPLE_START.md` instead of scripts.

---

## 🐳 Docker Files (Optional - For Production)

### Docker Setup (Only if you want containers)
- `Dockerfile` - API container
- `Dockerfile.dashboard` - Streamlit container
- `Dockerfile.simulator` - IoT simulator container
- `docker-compose.yml` - Multi-service orchestration
- `docker-compose-mqtt.yml` - MQTT simulation

**Note:** Docker is **not required** for development. Only use for production deployment.

---

## 🧪 Python Scripts (ML Training & Experiments)

### Core Scripts
- `python_scripts/api/run_api.py` ⭐ **ESSENTIAL** - Start FastAPI server
- `python_scripts/analysis/run_analysis.py` - Data analysis
- `python_scripts/training/run_phase3_gat.py` - Train GAT model
- `python_scripts/training/run_phase3.py` - Run Phase 3 experiments

### Training Scripts
- `python_scripts/training/train_gat_*.py` - Various GAT training approaches
- `python_scripts/training/simple_gat_train.py` - Simplified training
- `python_scripts/utilities/deploy_gat_model.py` - Model deployment
- `python_scripts/training/enable_gat.py` - Enable GAT features

### Utilities
- `python_scripts/inference/local_inference.py` - Run inference locally
- `python_scripts/inference/local_inference_optimized.py` - Optimized inference
- `python_scripts/inference/benchmark_inference.py` - Performance benchmarking
- `python_scripts/simulation/iot_device_simulator.py` - IoT device simulation
- `python_scripts/utilities/upload_to_huggingface.py` - Share models on HF Hub

### Dashboard Scripts
- `python_scripts/dashboard/dashboard_app.py` - Streamlit dashboard (legacy)
- `python_scripts/dashboard/dashboard_realtime.py` - Real-time Streamlit
- `python_scripts/dashboard/streamlit_inference_app.py` - Inference UI

---

## 📊 Configuration Files

### Essential Configs
- `configs/env_config.yaml` - Environment parameters
- `configs/hybrid_config.yaml` - Hybrid model settings
- `configs/phase3_gat_config.yaml` - GAT architecture

### Project Configs
- `pyproject.toml` - Python project metadata
- `pytest.ini` - Testing configuration
- `setup.py` - Package installation
- `requirements.txt` - All dependencies
- `requirements_api.txt` - API-only dependencies
- `requirements_dashboard.txt` - Dashboard dependencies

### Web App Configs
- `web-app/package.json` - Node.js dependencies
- `web-app/tsconfig.json` - TypeScript config
- `web-app/vite.config.ts` - Vite build config
- `web-app/nginx.conf` - Nginx production config

---

## 🗂️ Data & Outputs

### Directories
- `data/raw/` - Raw simulation data
- `data/processed/` - Preprocessed ML data
- `models/` - Trained model checkpoints
- `logs/` - TensorBoard training logs
- `reports/` - Analysis reports and plots
- `notebooks/` - Jupyter notebooks for analysis

### Sample Files
- `IOT-temp.csv` - Sample IoT data
- `example_network_state.json` - Sample network state

---

## 📝 Development Files

### Testing
- `tests/` - Unit tests
- `pytest.ini` - Test configuration

### Scripts
- `scripts/` - Utility bash/PowerShell scripts

### Documentation
- `docs/` - Complete documentation folder
- `DOCS_INDEX.md` - Documentation index
- `DOCUMENTATION_STRUCTURE.md` - Docs organization

---

## 🎯 Quick Decision Tree

### "I want to run the web dashboard"
→ Read `SIMPLE_START.md` (4 commands)

### "I want to train ML models"
→ Read `docs/QUICKSTART.md`

### "I want to understand the architecture"
→ Read `README.md` (Architecture section)

### "I want to deploy to production"
→ Read `WEB_APP_GUIDE.md` (Docker section)

### "I want to run MQTT simulation"
→ Read `OPTION4_MQTT_GUIDE.md`

### "I'm confused about scripts"
→ **Ignore all scripts!** Use manual commands from `SIMPLE_START.md`

---

## 🎓 Learning Path

**For Beginners:**
1. Start with `SIMPLE_START.md` - Get dashboard running
2. Read `README.md` - Understand project overview
3. Explore the dashboard at http://localhost:3000
4. Check `WEB_APP_GUIDE.md` if you want more details

**For ML Engineers:**
1. Read `docs/QUICKSTART.md` - ML training basics
2. Check `docs/PHASE3_GUIDE.md` - Advanced experiments
3. Run `python_scripts/training/run_phase3_gat.py` - Train your own model
4. Analyze results in Jupyter notebooks

**For DevOps:**
1. Read `docs/DOCKER_GUIDE.md` - Container deployment
2. Review `docker-compose.yml` - Service orchestration
3. Check `WEB_APP_GUIDE.md` - Production deployment
4. Test MQTT with `OPTION4_MQTT_GUIDE.md`

---

## 🚫 What NOT to Do

❌ Don't try to run PowerShell scripts with `python` command
❌ Don't try to use all the scripts at once
❌ Don't read all documentation files before starting
❌ Don't use Docker unless you need production deployment
❌ Don't skip `SIMPLE_START.md` - it's the easiest way!

✅ DO follow the 4 manual commands in `SIMPLE_START.md`
✅ DO keep the API and React terminals open
✅ DO check the browser console for errors
✅ DO ask questions if stuck!

---

## 📞 Need Help?

1. Check `SIMPLE_START.md` - Troubleshooting section
2. Read `README.md` - Common issues section
3. Review `WEB_APP_GUIDE.md` - Advanced troubleshooting
4. Open an issue on GitHub

---

**Remember:** Start with `SIMPLE_START.md` and ignore everything else until you need it!
