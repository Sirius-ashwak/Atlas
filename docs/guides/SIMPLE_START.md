# 🚀 Simple Start Guide - AI Edge Allocator

**This is the simplest way to get the React web dashboard running. No Docker, no automation scripts - just 4 manual commands.**

---

## ✅ Prerequisites

Before starting, make sure you have:

1. **Node.js 18+** - [Download from nodejs.org](https://nodejs.org/)
2. **Python 3.9+** - [Download from python.org](https://www.python.org/)
3. A terminal (PowerShell on Windows, bash on Linux/Mac)

---

## 🎯 5 Steps to Run the Dashboard

### Step 1: Install Python API Dependencies

Open PowerShell in the project root and run:

```powershell
# Install Python packages for the API
pip install -r requirements_api.txt
```

**What this does:** Installs FastAPI, uvicorn, and other Python packages needed for the backend (~2 minutes).

---

### Step 2: Install Node.js Dependencies

```powershell
cd web-app
npm install
```

**What this does:** Downloads all React, Material-UI, and visualization libraries (~5 minutes first time).

---

### Step 3: Start the Python API Backend

Open a terminal (PowerShell) and run:

```powershell
cd ..  # Go back to project root if you're in web-app
python python_scripts/api/run_api.py --port 8000
```

**What this does:** Starts FastAPI server on http://localhost:8000

**Keep this terminal open!** The API must run in the background.

---

### Step 4: Start the React Development Server

Open a **SECOND terminal** (new PowerShell window) and run:

```powershell
cd web-app
npm run dev
```

**What this does:** Starts Vite dev server on http://localhost:3000

**Keep this terminal open too!**

---

### Step 5: Open Your Browser

Navigate to:

```
http://localhost:3000
```

**You should see:**
- Modern Material-UI dashboard
- Network topology visualization (D3.js)
- Performance charts
- Model management interface

---

## 🎨 What You Can Do

1. **Dashboard** - View network topology and real-time metrics
2. **Models** - Browse trained models (DQN, PPO, GAT)
3. **Inference** - Test allocations with sample networks
4. **Monitoring** - Track system health and performance
5. **Settings** - Configure thresholds and preferences

---

## 🛑 Stopping the App

To stop the dashboard:

1. Press `Ctrl+C` in the React terminal (Step 4)
2. Press `Ctrl+C` in the API terminal (Step 3)

---

## ❓ Troubleshooting

### Problem: "npm: command not found"
**Solution:** Install Node.js from [nodejs.org](https://nodejs.org/) and restart your terminal.

### Problem: "python: command not found"
**Solution:** Install Python from [python.org](https://www.python.org/) or use `python3` instead of `python`.

### Problem: "Module not found" errors in Python
**Solution:** Install Python dependencies first:
```powershell
pip install -r requirements_api.txt
```

### Problem: Port 3000 or 8000 already in use
**Solution:** Stop other services using those ports, or change the port:
```powershell
# For API (change port)
python python_scripts/api/run_api.py --port 8080

# For React (change port in vite.config.ts)
```

### Problem: API connection errors in React
**Solution:** Make sure the FastAPI backend (Step 3) is running. Check http://localhost:8000/health in your browser.

---

## 📦 Optional: Production Build

To create an optimized production build:

```powershell
cd web-app
npm run build
```

This creates a `dist/` folder with production-ready files. You can serve these with any web server (Nginx, Apache, etc.).

---

## 🆚 Alternative: Use Docker

If you prefer Docker:

```powershell
docker-compose up --build web api
```

This starts both the API and web app in containers. Access at http://localhost:3000.

---

## 📚 Need More Details?

- **Full React Guide:** [WEB_APP_GUIDE.md](WEB_APP_GUIDE.md)
- **API Documentation:** [docs/API_GUIDE.md](../API_GUIDE.md)
- **Main README:** [README.md](../README.md)

---

## 🎯 Summary

**Total Time:** ~10 minutes (including installations)

**Commands:**
1. `pip install -r requirements_api.txt` (Install Python packages)
2. `cd web-app && npm install` (Install Node packages)
3. `python python_scripts/api/run_api.py --port 8000` (Terminal 1 - API)
4. `cd web-app && npm run dev` (Terminal 2 - React)
5. Open http://localhost:3000

**That's it!** No scripts needed, no Docker required. Just simple manual commands.

---

**Questions?** Check the [README.md](../README.md) or [WEB_APP_GUIDE.md](WEB_APP_GUIDE.md) for comprehensive documentation.
