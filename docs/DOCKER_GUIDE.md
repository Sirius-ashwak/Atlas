# 🐳 Docker Deployment Guide

Complete guide for containerized deployment of AI Edge Allocator.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Building Images](#building-images)
- [Running Containers](#running-containers)
- [Docker Compose](#docker-compose)
- [Configuration](#configuration)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### Install Docker

**Windows:**
- Download [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Install and restart your computer
- Verify: `docker --version`

**Linux:**
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Verify
docker --version
docker compose version
```

**macOS:**
- Download [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
- Install and start Docker Desktop
- Verify: `docker --version`

---

## 🏁 Quick Start

### Option 1: Docker Compose (Recommended)

**Start everything with one command:**

```bash
# Build and start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

**Access the services:**
- **API Server**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Dashboard**: http://localhost:8501

### Option 2: Individual Containers

**Build and run API:**
```bash
docker build -t edge-allocator-api .
docker run -d -p 8000:8000 --name api edge-allocator-api
```

**Build and run Dashboard:**
```bash
docker build -f Dockerfile.dashboard -t edge-allocator-dashboard .
docker run -d -p 8501:8501 --name dashboard edge-allocator-dashboard
```

---

## 🏗️ Building Images

### Build API Image

```bash
docker build -t edge-allocator-api:latest .
```

**With custom tag:**
```bash
docker build -t edge-allocator-api:v1.0.0 .
```

### Build Dashboard Image

```bash
docker build -f Dockerfile.dashboard -t edge-allocator-dashboard:latest .
```

### Build Both with Docker Compose

```bash
docker compose build
```

**No cache (force rebuild):**
```bash
docker compose build --no-cache
```

---

## 🚀 Running Containers

### Run API Container

**Basic:**
```bash
docker run -d \
  --name edge-allocator-api \
  -p 8000:8000 \
  edge-allocator-api
```

**With volume mounts:**
```bash
docker run -d \
  --name edge-allocator-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/logs:/app/logs \
  edge-allocator-api
```

**With environment variables:**
```bash
docker run -d \
  --name edge-allocator-api \
  -p 8000:8000 \
  -e LOG_LEVEL=debug \
  -e MODEL_DIR=/app/models \
  edge-allocator-api
```

### Run Dashboard Container

```bash
docker run -d \
  --name edge-allocator-dashboard \
  -p 8501:8501 \
  --link edge-allocator-api:api \
  -e API_BASE_URL=http://api:8000 \
  edge-allocator-dashboard
```

---

## 🎭 Docker Compose

### docker-compose.yml

The project includes a complete `docker-compose.yml` that orchestrates:
- API server (port 8000)
- Dashboard (port 8501)
- Network configuration
- Volume management
- Health checks

### Common Commands

**Start services:**
```bash
docker compose up -d
```

**View logs:**
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
docker compose logs -f dashboard
```

**Stop services:**
```bash
docker compose stop
```

**Stop and remove:**
```bash
docker compose down
```

**Rebuild and restart:**
```bash
docker compose up -d --build
```

**Check status:**
```bash
docker compose ps
```

**Execute commands in container:**
```bash
docker compose exec api python -c "print('Hello from API')"
```

---

## ⚙️ Configuration

### Environment Variables

**API Container:**
| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_DIR` | Path to models directory | `/app/models` |
| `LOG_LEVEL` | Logging level | `info` |
| `PYTHONUNBUFFERED` | Unbuffered Python output | `1` |

**Dashboard Container:**
| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | API server URL | `http://api:8000` |
| `STREAMLIT_SERVER_PORT` | Dashboard port | `8501` |
| `STREAMLIT_SERVER_ADDRESS` | Bind address | `0.0.0.0` |

### Volume Mounts

**Models volume:**
```yaml
volumes:
  - ./models:/app/models:ro  # Read-only models
```

**Logs volume:**
```yaml
volumes:
  - ./logs:/app/logs  # Read-write logs
```

**Custom data:**
```yaml
volumes:
  - ./custom_data:/app/data
```

---

## 🌐 Production Deployment

### Best Practices

1. **Use specific image tags (not `latest`):**
   ```bash
   docker build -t edge-allocator-api:1.0.0 .
   ```

2. **Set resource limits:**
   ```yaml
   services:
     api:
       deploy:
         resources:
           limits:
             cpus: '2'
             memory: 4G
           reservations:
             cpus: '1'
             memory: 2G
   ```

3. **Use health checks:**
   ```yaml
   healthcheck:
     test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
     interval: 30s
     timeout: 10s
     retries: 3
   ```

4. **Enable logging:**
   ```yaml
   logging:
     driver: "json-file"
     options:
       max-size: "10m"
       max-file: "3"
   ```

### Security

**Don't run as root:**
```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

**Use secrets for sensitive data:**
```yaml
secrets:
  api_key:
    file: ./secrets/api_key.txt
```

**Scan images for vulnerabilities:**
```bash
docker scan edge-allocator-api
```

### Reverse Proxy (Nginx)

**nginx.conf example:**
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### HTTPS with Let's Encrypt

```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certbot/conf:/etc/letsencrypt
      - ./certbot/www:/var/www/certbot
```

---

## 📊 Monitoring

### Container Stats

```bash
# Real-time stats
docker stats

# Specific container
docker stats edge-allocator-api
```

### Logs

```bash
# Follow logs
docker logs -f edge-allocator-api

# Last 100 lines
docker logs --tail 100 edge-allocator-api

# With timestamps
docker logs -t edge-allocator-api
```

### Health Checks

```bash
# Check health status
docker inspect --format='{{.State.Health.Status}}' edge-allocator-api

# View health check logs
docker inspect --format='{{json .State.Health}}' edge-allocator-api | jq
```

---

## 🐛 Troubleshooting

### Issue: Container won't start

**Check logs:**
```bash
docker logs edge-allocator-api
```

**Inspect container:**
```bash
docker inspect edge-allocator-api
```

**Check if port is in use:**
```bash
# Windows
netstat -ano | findstr :8000

# Linux/Mac
lsof -i :8000
```

### Issue: Models not loading

**Verify volume mount:**
```bash
docker exec edge-allocator-api ls -la /app/models
```

**Check permissions:**
```bash
ls -la ./models
chmod -R 755 ./models  # If needed
```

### Issue: Can't connect to API from dashboard

**Check network:**
```bash
docker network inspect edge-allocator-network
```

**Test connectivity:**
```bash
docker exec edge-allocator-dashboard curl http://api:8000/health
```

### Issue: Out of disk space

**Clean up:**
```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove everything unused
docker system prune -a --volumes
```

### Issue: Build fails

**Clear build cache:**
```bash
docker builder prune
```

**Build without cache:**
```bash
docker build --no-cache -t edge-allocator-api .
```

---

## 🔄 Updates and Maintenance

### Update Images

```bash
# Pull latest base images
docker compose pull

# Rebuild with latest code
docker compose up -d --build
```

### Backup

**Backup models:**
```bash
docker cp edge-allocator-api:/app/models ./models_backup
```

**Backup logs:**
```bash
docker cp edge-allocator-api:/app/logs ./logs_backup
```

### Restore

```bash
docker cp ./models_backup/. edge-allocator-api:/app/models
```

---

## 🚢 Registry & Distribution

### Push to Docker Hub

```bash
# Tag image
docker tag edge-allocator-api username/edge-allocator-api:1.0.0

# Login
docker login

# Push
docker push username/edge-allocator-api:1.0.0
```

### Private Registry

```bash
# Tag for private registry
docker tag edge-allocator-api myregistry.com/edge-allocator-api:1.0.0

# Push
docker push myregistry.com/edge-allocator-api:1.0.0
```

---

## 📝 Example Workflows

### Development

```bash
# Start with live reload
docker compose -f docker-compose.dev.yml up

# Watch logs
docker compose logs -f

# Restart after code changes
docker compose restart api
```

### Testing

```bash
# Run tests in container
docker compose run --rm api pytest

# With coverage
docker compose run --rm api pytest --cov
```

### Production

```bash
# Deploy to production
docker compose -f docker-compose.prod.yml up -d

# Monitor
docker compose ps
docker compose logs -f

# Scale services
docker compose up -d --scale api=3
```

---

## 📚 Additional Resources

- **Docker Documentation**: https://docs.docker.com
- **Docker Compose**: https://docs.docker.com/compose/
- **Docker Hub**: https://hub.docker.com
- **Best Practices**: https://docs.docker.com/develop/dev-best-practices/

---

## 🆘 Getting Help

- **Docker Community**: https://forums.docker.com
- **Stack Overflow**: Tag `docker` or `docker-compose`
- **Project Issues**: [GitHub Issues](https://github.com/Sirius-ashwak/DeepSea-IoT/issues)

---

**Happy Containerizing!** 🐳 Questions? Check the main README or open an issue on GitHub!
