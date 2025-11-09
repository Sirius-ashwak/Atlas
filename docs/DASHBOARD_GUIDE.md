# 🎨 Dashboard Guide: AI Edge Allocator

Interactive Streamlit dashboard for monitoring, visualization, and real-time predictions.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Troubleshooting](#troubleshooting)

---

## 🏁 Quick Start

### 1. Install Dependencies

```bash
pip install streamlit plotly requests pandas numpy
# Or use requirements file
pip install -r requirements_dashboard.txt
```

### 2. Start the API Server (Required)

**Terminal 1:**
```bash
python python_scripts/api/run_api.py
```

Wait for: `✅ Server started at http://0.0.0.0:8000`

### 3. Start the Dashboard

**Terminal 2:**
```bash
streamlit run python_scripts/dashboard/dashboard_app.py
```

Or use the launcher:
```powershell
.\run_dashboard.ps1  # Windows
./run_dashboard.sh    # Linux/Mac
```

### 4. Open in Browser

The dashboard will automatically open at: **http://localhost:8501**

---

## ✨ Features

### 📊 **Overview Tab**
- **System Status**: Real-time API health monitoring
- **Models Status**: Check which models are loaded
- **Uptime Tracking**: Monitor server uptime
- **Quick Metrics**: Key performance indicators

### 🔮 **Prediction Tab**
- **Network Generation**: Create random IoT networks
- **Interactive Topology**: Visualize network structure
- **Real-time Predictions**: Get optimal node placement
- **Node Metrics**: Detailed resource utilization charts
- **Confidence Scores**: See prediction confidence

### 📈 **Analytics Tab**
- **Model Comparison**: Compare different model performances
- **Training History**: View training metrics (coming soon)
- **Performance Charts**: Interactive Plotly visualizations

### ℹ️ **About Tab**
- Project information
- Model performance summary
- Links to documentation
- Author and license info

---

## 📦 Installation

### Option 1: Pip Install

```bash
# Install dashboard dependencies
pip install streamlit plotly requests pandas numpy

# Or use requirements file
pip install -r requirements_dashboard.txt
```

### Option 2: With Virtual Environment

```bash
# Create and activate virtual environment
python -m venv venv_dashboard
source venv_dashboard/bin/activate  # Linux/Mac
# venv_dashboard\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements_dashboard.txt
```

---

## 🚀 Usage

### Basic Usage

**Step 1: Start API Server**
```bash
python python_scripts/api/run_api.py
```

**Step 2: Start Dashboard** (in new terminal)
```bash
streamlit run python_scripts/dashboard/dashboard_app.py
```

### Custom Port

```bash
streamlit run python_scripts/dashboard/dashboard_app.py --server.port 8502
```

### With Custom API URL

Edit `python_scripts/dashboard/dashboard_app.py` and change:
```python
API_BASE_URL = "http://your-api-server:8000"
```

---

## 🎯 Using the Dashboard

### Making Predictions

1. Navigate to the **🔮 Prediction** tab
2. Configure number of nodes in the sidebar
3. Click **🎲 Generate Random Network**
4. Click **🚀 Run Prediction**
5. View the results:
   - Selected node (highlighted in red)
   - Confidence score
   - Processing time
   - Node metrics charts

### Selecting Models

Use the sidebar to choose different models:
- **hybrid** - Best overall performance (recommended)
- **dqn** - Deep Q-Network
- **ppo** - Proximal Policy Optimization
- **hybrid_gat** - Hybrid with GAT encoder
- **hybrid_attention** - Hybrid with attention fusion

### Auto-Refresh Mode

Enable auto-refresh in the sidebar to automatically update the dashboard:
1. Check "Auto-refresh"
2. Set refresh interval (1-10 seconds)
3. Dashboard will update automatically

---

## 📸 Dashboard Components

### Network Topology Visualization

- **Blue nodes**: Sensors (IoT devices)
- **Green nodes**: Fog servers (edge computing)
- **Orange nodes**: Cloud servers
- **Red node**: Selected node for task placement

Hover over nodes to see:
- Node ID
- Node type
- CPU utilization
- Queue length

### Resource Metrics Charts

**Chart 1: Resource Utilization**
- CPU utilization per node
- Memory utilization per node
- Grouped bar chart

**Chart 2: Latency & Queue**
- Network latency (line chart)
- Task queue length (line chart)
- Dual-axis visualization

---

## 🔧 Configuration

### Sidebar Options

| Option | Description | Default |
|--------|-------------|---------|
| Model Selection | Choose RL model | hybrid |
| Number of Nodes | Network size | 10 |
| Auto-refresh | Enable auto-update | Off |
| Refresh Interval | Update frequency | 5s |

### Environment Variables

You can set these before running:

```bash
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=localhost
export API_BASE_URL=http://localhost:8000
```

---

## 🐛 Troubleshooting

### Issue: "API Server: Disconnected"

**Cause**: API server is not running

**Solution**:
```bash
# In another terminal, start the API server
python python_scripts/api/run_api.py
```

### Issue: "Model not loaded"

**Cause**: Model files don't exist in `models/` directory

**Solution**:
```bash
# Train models first
python -m src.main train-hybrid --timesteps 10000
```

### Issue: Port already in use

**Solution**: Use a different port
```bash
streamlit run python_scripts/dashboard/dashboard_app.py --server.port 8502
```

### Issue: Streamlit not found

**Solution**: Install dependencies
```bash
pip install streamlit plotly
```

### Issue: Prediction fails

**Possible causes**:
1. API server not running → Start it with `python python_scripts/api/run_api.py`
2. Models not loaded → Check API health at `http://localhost:8000/health`
3. Network issues → Check firewall settings

---

## 🎨 Customization

### Change Theme

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Dark Mode

```toml
[theme]
base = "dark"
primaryColor = "#4da6ff"
backgroundColor = "#0e1117"
secondaryBackgroundColor = "#262730"
textColor = "#fafafa"
```

### Add Custom Pages

Create new tabs by modifying `python_scripts/dashboard/dashboard_app.py`:

```python
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🔮 Prediction", 
    "📈 Analytics",
    "🎯 Custom Tab",  # Your new tab
    "ℹ️ About"
])

with tab5:
    st.subheader("Custom Content")
    # Your code here
```

---

## 📊 Performance Tips

### For Large Networks (100+ nodes)

1. Disable auto-refresh
2. Use batch predictions
3. Reduce visualization complexity

### For Production Deployment

1. Run behind Nginx reverse proxy
2. Enable authentication
3. Set appropriate resource limits
4. Use HTTPS

---

## 🔗 Integration with API

The dashboard communicates with the FastAPI server:

```python
# Health check
GET http://localhost:8000/health

# List models
GET http://localhost:8000/models

# Make prediction
POST http://localhost:8000/predict
{
    "network_state": {...},
    "model_type": "hybrid"
}
```

---

## 📚 Additional Resources

- **Streamlit Documentation**: https://docs.streamlit.io
- **Plotly Charts**: https://plotly.com/python/
- **API Documentation**: http://localhost:8000/docs
- **Project README**: ../README.md

---

## 🚀 Advanced Features

### Export Predictions

Add export functionality:

```python
import json

if st.button("Export Results"):
    results = {
        "timestamp": datetime.now().isoformat(),
        "prediction": st.session_state.get('prediction_result'),
        "network": st.session_state.get('nodes')
    }
    
    json_str = json.dumps(results, indent=2)
    st.download_button(
        "Download JSON",
        json_str,
        "prediction_results.json",
        "application/json"
    )
```

### Real-time Monitoring

Stream predictions:

```python
placeholder = st.empty()

for i in range(100):
    with placeholder.container():
        result = make_prediction(nodes, edges, model)
        st.metric("Current Selection", result['selected_node'])
    time.sleep(1)
```

---

## 💡 Tips & Best Practices

1. **Keep API Running**: Dashboard needs API server to function
2. **Use Auto-refresh Sparingly**: Can increase load on API
3. **Check Model Status**: Ensure models are loaded before predictions
4. **Monitor Performance**: Watch processing times in metrics
5. **Save Configurations**: Use session state for persistence

---

## 🤝 Contributing

Want to add features to the dashboard?

1. Fork the repository
2. Modify `python_scripts/dashboard/dashboard_app.py`
3. Test thoroughly
4. Submit pull request

**Ideas for contributions**:
- Real-time training monitoring
- Comparison mode (side-by-side models)
- Historical predictions log
- Advanced filtering options
- Export to PDF reports

---

## 📞 Support

Need help?

- **GitHub Issues**: [Report issues](https://github.com/Sirius-ashwak/DeepSea-IoT/issues)
- **Documentation**: Check README.md and API_GUIDE.md
- **API Health**: Visit http://localhost:8000/health

---

**Happy Monitoring!** 🎨 Questions? Check the main README or open an issue on GitHub!
