# 🚀 AI Edge Allocator - Modern Web Application Setup Guide

Complete guide to set up and run the new React-based web dashboard.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (Development)](#quick-start-development)
3. [Docker Deployment](#docker-deployment)
4. [Manual Setup](#manual-setup)
5. [Configuration](#configuration)
6. [Features](#features)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

- **Node.js 18+** and npm
- **Python 3.9+** (for FastAPI backend)
- **Docker** (optional, for containerized deployment)
- **Git**

### Check Versions

```powershell
node --version    # Should be v18.0.0 or higher
npm --version     # Should be 9.0.0 or higher
python --version  # Should be 3.9 or higher
docker --version  # Optional
```

---

## 🚀 Quick Start (Development)

### Option 1: Automated Setup (Recommended)

Run the automated setup script:

```powershell
cd ai_edge_allocator
.\setup_web_app.ps1
```

This script will:
1. Install Node.js dependencies
2. Create environment files
3. Start the FastAPI backend
4. Start the React development server

### Option 2: Manual Setup

#### Step 1: Install Backend Dependencies

```powershell
cd ai_edge_allocator

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install Python dependencies
pip install -r requirements_api.txt
```

#### Step 2: Start FastAPI Backend

```powershell
# In terminal 1
python python_scripts/api/run_api.py --port 8000
```

The API will be available at: http://localhost:8000

#### Step 3: Install Frontend Dependencies

```powershell
# In terminal 2
cd web-app
npm install
```

#### Step 4: Start React Development Server

```powershell
npm run dev
```

The web dashboard will be available at: http://localhost:3000

---

## 🐳 Docker Deployment

### Full Stack Deployment

Deploy all services (API + Web Dashboard) with Docker Compose:

```powershell
cd ai_edge_allocator

# Build and start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f web
```

**Services Running:**
- FastAPI: http://localhost:8000
- React Web: http://localhost:3000
- API Docs: http://localhost:8000/docs

### Individual Service Deployment

#### Build React Web App Only

```powershell
cd web-app
docker build -t edge-allocator-web .
docker run -d -p 3000:3000 edge-allocator-web
```

### Stop Services

```powershell
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

---

## 📝 Configuration

### Environment Variables

#### Development (.env.development)

```env
VITE_API_URL=http://localhost:8000
```

#### Production (.env.production)

```env
VITE_API_URL=/api
```

### API Proxy Configuration

The Vite dev server proxies `/api` requests to the FastAPI backend.

**vite.config.ts:**
```typescript
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/api/, '')
    }
  }
}
```

### Change Port

To run on a different port:

```powershell
# Method 1: Edit vite.config.ts
# Change port: 3000 to your desired port

# Method 2: Use environment variable
$env:PORT=3001; npm run dev
```

---

## 🎯 Features

### 1. Interactive Dashboard
- Real-time network topology visualization using D3.js
- Performance metrics cards (latency, cost, bandwidth)
- Allocation table with confidence scores
- Responsive charts with Recharts

### 2. Model Management
- View all trained models
- Compare model performance
- Load/unload models dynamically

### 3. Custom Inference
- Configure network parameters
- Select models (DQN, PPO, Hybrid)
- Run predictions
- Visualize allocation results

### 4. Real-time Monitoring
- WebSocket-based live updates
- Historical performance tracking
- Alert system for anomalies

### 5. Settings
- API configuration
- Theme preferences
- Export/import configurations

---

## 📊 Available Scripts

```powershell
# Development
npm run dev              # Start dev server with hot reload

# Production
npm run build            # Build for production
npm run preview          # Preview production build

# Code Quality
npm run lint             # Run ESLint
npm run type-check       # TypeScript type checking

# Docker
docker-compose up -d     # Start all services
docker-compose logs -f   # View logs
docker-compose down      # Stop all services
```

---

## 🔍 API Integration

The web app communicates with the FastAPI backend using these endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/models` | List available models |
| GET | `/models/{name}` | Get model details |
| POST | `/predict` | Run inference |
| POST | `/predict/batch` | Batch predictions |
| GET | `/metrics` | Get system metrics |
| POST | `/generate-mock-network` | Generate test network |

### Example API Call

```typescript
import ApiService from './services/api'

// Run prediction
const result = await ApiService.predict({
  network_state: {
    devices: [...],
    num_devices: 15,
    num_fog_nodes: 3,
    num_cloud_nodes: 2
  },
  model_name: 'hybrid'
})
```

---

## 🎨 UI Components

### Material-UI Components Used

- **Layout**: AppBar, Drawer, Box, Container
- **Data Display**: Table, Card, Chip, Typography
- **Inputs**: Button, TextField, Select
- **Feedback**: CircularProgress, Alert, Snackbar
- **Navigation**: Tabs, Breadcrumbs

### Custom Components

- `NetworkTopology`: D3.js network graph
- `MetricsCards`: KPI dashboard cards
- `AllocationTable`: Device allocation results
- `PerformanceChart`: Recharts bar charts

---

## 🐛 Troubleshooting

### Issue 1: Port 3000 Already in Use

**Solution:**
```powershell
# Find and kill process on port 3000
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Or use a different port
$env:PORT=3001; npm run dev
```

### Issue 2: API Connection Failed

**Symptoms:**
- Dashboard shows "Failed to load data"
- Network errors in browser console

**Solutions:**
1. Verify FastAPI is running:
   ```powershell
   curl http://localhost:8000/health
   ```

2. Check CORS settings in FastAPI (`python_scripts/api/run_api.py`):
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["http://localhost:3000"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

3. Verify proxy configuration in `vite.config.ts`

### Issue 3: npm install Fails

**Solutions:**
```powershell
# Clear npm cache
npm cache clean --force

# Delete node_modules and package-lock.json
Remove-Item -Recurse -Force node_modules
Remove-Item package-lock.json

# Reinstall
npm install
```

### Issue 4: TypeScript Errors

**Solution:**
```powershell
# Install missing type definitions
npm install --save-dev @types/react @types/react-dom @types/node

# Restart TypeScript server in VS Code
# Ctrl+Shift+P -> "TypeScript: Restart TS Server"
```

### Issue 5: Docker Build Fails

**Solutions:**
```powershell
# Clean Docker build cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache

# Check Docker logs
docker-compose logs web
```

### Issue 6: White Screen on Production Build

**Causes:**
- Missing environment variables
- Routing issues
- Build errors

**Solutions:**
```powershell
# Check build output
npm run build

# Test production build locally
npm run preview

# Verify environment variables
cat .env.production
```

---

## 🔧 Advanced Configuration

### Enable HTTPS (Development)

Create `vite.config.ts`:
```typescript
import fs from 'fs'
import path from 'path'

export default defineConfig({
  server: {
    https: {
      key: fs.readFileSync('path/to/key.pem'),
      cert: fs.readFileSync('path/to/cert.pem'),
    }
  }
})
```

### Custom Nginx Configuration

For production deployment, modify `nginx.conf`:

```nginx
# Add custom headers
add_header X-Custom-Header "value";

# Increase client body size
client_max_body_size 10M;

# Add rate limiting
limit_req_zone $binary_remote_addr zone=mylimit:10m rate=10r/s;
limit_req zone=mylimit burst=20;
```

### Environment-Specific Builds

```powershell
# Development build
npm run build -- --mode development

# Staging build
npm run build -- --mode staging

# Production build
npm run build -- --mode production
```

---

## 📱 Mobile Development

Test on mobile devices:

```powershell
# Find your local IP
ipconfig

# Start dev server accessible from network
npm run dev -- --host

# Access from mobile browser
http://<your-ip>:3000
```

---

## 🚀 Deployment Checklist

- [ ] Environment variables configured
- [ ] API backend is accessible
- [ ] Build runs without errors (`npm run build`)
- [ ] All tests pass (`npm run test`)
- [ ] Linting passes (`npm run lint`)
- [ ] Docker image builds successfully
- [ ] Health check endpoint works
- [ ] CORS configured correctly
- [ ] Security headers added
- [ ] Performance optimization enabled (gzip, caching)

---

## 📚 Additional Resources

- [React Documentation](https://react.dev/)
- [Material-UI Docs](https://mui.com/)
- [Vite Guide](https://vitejs.dev/)
- [D3.js Gallery](https://d3-graph-gallery.com/)
- [Recharts Examples](https://recharts.org/)

---

## 🤝 Support

For issues or questions:
1. Check this troubleshooting guide
2. Review GitHub Issues
3. Contact project maintainers

---

**Built with ❤️ for IoT Edge Computing Resource Allocation**
