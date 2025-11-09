# AI Edge Allocator - React Web Application

Modern web dashboard for the AI Edge Allocator IoT resource allocation system, built with React, TypeScript, Material-UI, and D3.js.

## 🎯 Features

- **Interactive Dashboard**: Real-time visualization of network topology and resource allocation
- **Model Management**: View and manage trained AI models (DQN, PPO, Hybrid)
- **Live Inference**: Run predictions on custom network configurations
- **Performance Monitoring**: Track metrics (latency, cost, bandwidth) with charts
- **Responsive Design**: Mobile-friendly interface with Material-UI components
- **Modern Stack**: React 18, TypeScript, Vite, Material-UI v5

## 📁 Project Structure

```
web-app/
├── src/
│   ├── components/          # Reusable React components
│   │   ├── Layout/         # Navbar, Sidebar
│   │   └── Dashboard/      # Dashboard-specific components
│   ├── pages/              # Route pages
│   │   ├── Dashboard.tsx   # Main dashboard
│   │   ├── Models.tsx      # Model management
│   │   ├── Inference.tsx   # Custom inference
│   │   ├── Monitoring.tsx  # Real-time monitoring
│   │   └── Settings.tsx    # Application settings
│   ├── services/           # API integration
│   │   └── api.ts          # FastAPI client
│   ├── store/              # State management (Zustand)
│   │   └── useAppStore.ts  # Global app state
│   ├── types/              # TypeScript definitions
│   │   └── index.ts        # Type interfaces
│   ├── App.tsx             # Main app component
│   ├── main.tsx            # Entry point
│   └── theme.ts            # Material-UI theme
├── public/                 # Static assets
├── Dockerfile              # Multi-stage Docker build
├── nginx.conf              # Nginx configuration
├── package.json            # Dependencies
├── tsconfig.json           # TypeScript config
├── vite.config.ts          # Vite bundler config
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm
- Python FastAPI backend running on port 8000

### Development Setup

1. **Install Dependencies**:
```powershell
cd web-app
npm install
```

2. **Start Development Server**:
```powershell
npm run dev
```

3. **Access the Dashboard**:
Open http://localhost:3000 in your browser

### Production Build

1. **Build for Production**:
```powershell
npm run build
```

2. **Preview Production Build**:
```powershell
npm run preview
```

## 🐳 Docker Deployment

### Build Docker Image

```powershell
docker build -t edge-allocator-web .
```

### Run Container

```powershell
docker run -d -p 3000:3000 --name web-dashboard edge-allocator-web
```

### Docker Compose (Recommended)

Update the main `docker-compose.yml` to include the web service (see Docker section below).

## 🔧 Configuration

### Environment Variables

Create `.env.development` and `.env.production` files:

```env
# API Backend URL
VITE_API_URL=http://localhost:8000
```

### Vite Configuration

The `vite.config.ts` includes a proxy to forward `/api` requests to the FastAPI backend:

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

## 📦 Tech Stack

### Core
- **React 18**: UI framework
- **TypeScript**: Type safety
- **Vite**: Fast build tool and dev server

### UI Libraries
- **Material-UI v5**: Component library
- **@mui/icons-material**: Icon components
- **@emotion**: CSS-in-JS styling

### Visualization
- **D3.js**: Network topology graphs
- **Recharts**: Performance charts
- **React Flow**: Node-based visualizations

### State Management
- **Zustand**: Lightweight state management

### API & Networking
- **Axios**: HTTP client
- **Socket.IO Client**: WebSocket support (for real-time monitoring)

### Utilities
- **React Router v6**: Client-side routing
- **React Toastify**: Toast notifications
- **date-fns**: Date formatting

## 🎨 Key Components

### Dashboard
- **NetworkTopology**: D3.js visualization of IoT network
- **MetricsCards**: KPI cards (latency, cost, bandwidth)
- **AllocationTable**: Device allocation results
- **PerformanceChart**: Bar charts for metrics comparison

### Layout
- **Navbar**: Top navigation with app branding
- **Sidebar**: Left navigation menu

### Services
- **ApiService**: Centralized API client for FastAPI backend

## 📊 Available Scripts

```powershell
# Development server with hot reload
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Run linter
npm run lint
```

## 🔌 API Integration

The web app connects to the FastAPI backend with these endpoints:

```typescript
// Health check
GET /health

// List models
GET /models

// Get model info
GET /models/{model_name}

// Run prediction
POST /predict

// Batch prediction
POST /predict/batch

// Get metrics
GET /metrics

// Generate mock network
POST /generate-mock-network

// Training history
GET /training-history/{model_name}
```

## 🐳 Docker Integration

### Updated docker-compose.yml

Add this service to your main `docker-compose.yml`:

```yaml
services:
  # ... existing services (api, dashboard) ...

  # React Web Application
  web:
    build:
      context: ./web-app
      dockerfile: Dockerfile
    container_name: edge-allocator-web
    ports:
      - "3000:3000"
    depends_on:
      - api
    environment:
      - VITE_API_URL=/api
    networks:
      - edge-allocator-network
    restart: unless-stopped
```

### Full Stack Deployment

```powershell
# Start all services
docker-compose up -d

# Services:
# - FastAPI: http://localhost:8000
# - React Web: http://localhost:3000
# - Streamlit (legacy): http://localhost:8501
```

## 🎯 Usage Examples

### 1. View Dashboard

Navigate to http://localhost:3000 to see:
- Real-time network topology
- Current allocations
- Performance metrics

### 2. Run Custom Inference

1. Go to the **Inference** page
2. Configure network parameters
3. Select a model (DQN, PPO, Hybrid)
4. Click "Run Inference"
5. View results and allocation decisions

### 3. Monitor Performance

The **Monitoring** page provides:
- Real-time metric updates
- Historical performance charts
- WebSocket-based live data

## 🔒 Security Features

- CORS configuration for API access
- Environment-based API URL configuration
- Nginx security headers (X-Frame-Options, X-Content-Type-Options)
- Input validation on all forms

## 🚀 Performance Optimizations

- **Code Splitting**: Automatic route-based code splitting
- **Lazy Loading**: Components loaded on demand
- **Caching**: Static assets cached for 1 year
- **Compression**: Gzip enabled in Nginx
- **Production Build**: Minified and optimized bundle

## 📱 Responsive Design

The dashboard is fully responsive:
- **Desktop**: Full-width dashboard with sidebar
- **Tablet**: Collapsible sidebar
- **Mobile**: Bottom navigation, stacked components

## 🐛 Troubleshooting

### Port Already in Use

```powershell
# Change port in vite.config.ts or use environment variable
PORT=3001 npm run dev
```

### API Connection Issues

1. Verify FastAPI is running on port 8000
2. Check CORS settings in FastAPI
3. Ensure proxy configuration in `vite.config.ts` is correct

### Build Errors

```powershell
# Clear cache and rebuild
rm -rf node_modules dist
npm install
npm run build
```

## 🔄 Migration from Streamlit

To replace the Streamlit dashboard:

1. **Stop Streamlit container**:
```powershell
docker-compose stop dashboard
```

2. **Start React web app**:
```powershell
docker-compose up -d web
```

3. **Update links** in documentation to point to `http://localhost:3000`

## 📝 Development Roadmap

- [ ] WebSocket integration for real-time updates
- [ ] Advanced model comparison page
- [ ] Custom training configuration UI
- [ ] Export/import network configurations
- [ ] Dark mode theme toggle
- [ ] Multi-language support (i18n)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

Part of the AI Edge Allocator project. See main LICENSE file.

## 📧 Support

For issues or questions, open an issue on GitHub or contact the project maintainers.

---

**Built with ❤️ using React, TypeScript, and Material-UI**
