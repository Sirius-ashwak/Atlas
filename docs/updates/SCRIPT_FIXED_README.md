# ✅ SCRIPT FIXED - Ready to Run!

## 🛠️ What Was Fixed

The PowerShell script had a syntax error in the `try-catch` block. It has been fixed and is now ready to use!

---

## 🚀 Three Ways to Start

### Option 1: Full Setup Script (Recommended)
```powershell
.\setup_web_app.ps1
```
**Features:**
- ✅ Checks all prerequisites
- ✅ Installs dependencies
- ✅ Creates environment files
- ✅ Offers to start FastAPI
- ✅ Starts React dev server

### Option 2: Quick Start Script (Simplest)
```powershell
.\quick_start_web.ps1
```
**Features:**
- ✅ Minimal checks
- ✅ Installs dependencies if needed
- ✅ Starts React dev server immediately

### Option 3: Manual Steps
```powershell
cd web-app
npm install
npm run dev
```
**For developers who want full control**

---

## 📋 Prerequisites

Before running any script, make sure you have:

1. **Node.js 18+** installed
   ```powershell
   node --version
   ```

2. **npm** installed
   ```powershell
   npm --version
   ```

3. **Python 3.9+** installed (for FastAPI backend)
   ```powershell
   python --version
   ```

---

## ⚡ Quick Start (30 Seconds)

```powershell
# Step 1: Navigate to project
cd ai_edge_allocator

# Step 2: Run the quick start
.\quick_start_web.ps1

# Step 3: Wait for "Local: http://localhost:3000"
# Step 4: Open browser to http://localhost:3000
```

---

## 🐛 If You Get Errors

### Error: "web-app directory not found"
**Solution:** Make sure you're in the `ai_edge_allocator` directory
```powershell
cd "c:\Users\mohamed\OneDrive\Documents\LEARN\GOGLE DEV\Windsurf\IOT\ai_edge_allocator"
.\setup_web_app.ps1
```

### Error: "npm: command not found"
**Solution:** Install Node.js from https://nodejs.org/
```powershell
# After installation, restart PowerShell and try again
```

### Error: "Port 3000 already in use"
**Solution:** Kill the process on port 3000
```powershell
# Find process
netstat -ano | findstr :3000

# Kill it (replace PID with actual number)
taskkill /PID <PID> /F

# Try again
.\quick_start_web.ps1
```

### Error: npm install fails
**Solution:** Clear cache and reinstall
```powershell
cd web-app
Remove-Item -Recurse -Force node_modules
Remove-Item package-lock.json
npm cache clean --force
npm install
```

---

## 🎯 What Happens When You Run the Script

1. ✅ Checks for Node.js, npm, and Python
2. ✅ Navigates to `web-app` directory
3. ✅ Installs npm dependencies (first time only, ~2-3 minutes)
4. ✅ Creates environment files
5. ✅ Checks if FastAPI is running
6. ✅ Offers to start FastAPI in new terminal (optional)
7. ✅ Starts React development server
8. ✅ Opens dashboard at http://localhost:3000

---

## 📊 Expected Output

```
================================================
  AI Edge Allocator - Web App Setup
================================================

[1/7] Checking prerequisites...
✓ Node.js: v18.x.x
✓ npm: 9.x.x
✓ Python: 3.11.x

[2/7] Setting up web application...
✓ Found web-app directory

[3/7] Installing npm dependencies...
✓ npm dependencies installed successfully

[4/7] Creating environment files...
✓ Created .env.development
✓ Created .env.production

[5/7] Checking FastAPI backend...
⚠ FastAPI backend is not running

[6/7] Starting services...
Would you like to start the FastAPI backend? (Y/N)

[7/7] Starting React development server...
================================================
  Setup Complete!
================================================

Services:
  • FastAPI Backend: http://localhost:8000
  • API Documentation: http://localhost:8000/docs
  • React Dashboard: http://localhost:3000

  VITE v5.0.8  ready in 1234 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

---

## 🔧 Alternative: Start FastAPI Separately

If you want to start the FastAPI backend separately:

```powershell
# Terminal 1: Start FastAPI
cd ai_edge_allocator
python python_scripts/api/run_api.py --port 8000

# Terminal 2: Start React
cd ai_edge_allocator
.\quick_start_web.ps1
```

---

## 🐳 Or Use Docker (Easiest)

```powershell
cd ai_edge_allocator
docker-compose up -d
```

Then visit:
- React Web: http://localhost:3000
- FastAPI: http://localhost:8000

---

## ✅ Success Indicators

You'll know it's working when you see:

1. ✅ No error messages in terminal
2. ✅ Message: "Local: http://localhost:3000"
3. ✅ Browser opens to dashboard
4. ✅ Dashboard shows network topology
5. ✅ No console errors in browser (F12)

---

## 📞 Still Having Issues?

1. **Check the full guide:** `WEB_APP_GUIDE.md`
2. **Read troubleshooting:** Section 🐛 Troubleshooting
3. **Verify Node.js version:** Must be 18 or higher
4. **Check npm version:** Must be 9 or higher
5. **Restart PowerShell:** Sometimes helps with PATH issues

---

## 🎉 Ready to Go!

Choose your preferred method and run it:

```powershell
# Recommended for first-time users
.\setup_web_app.ps1

# OR for quick starts
.\quick_start_web.ps1

# OR manual
cd web-app && npm install && npm run dev
```

**Your dashboard will be ready in 2-3 minutes!** 🚀

---

**Need more help?** See `WEB_APP_GUIDE.md` for comprehensive troubleshooting.
