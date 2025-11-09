"""
Streamlit Dashboard for AI Edge Allocator
Real-time monitoring, visualization, and prediction interface.

Usage:
    streamlit run dashboard_app.py
    streamlit run dashboard_app.py --server.port 8501
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="AI Edge Allocator Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health() -> Optional[Dict]:
    """Check if API server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_models_status() -> Dict:
    """Get status of all models."""
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=2)
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}


def create_sample_network(num_nodes: int = 10) -> tuple:
    """Create a sample network for testing."""
    np.random.seed(42)
    
    nodes = []
    for i in range(num_nodes):
        # Determine node type based on position
        if i < num_nodes * 0.5:
            node_type = 0  # Sensor
        elif i < num_nodes * 0.8:
            node_type = 1  # Fog
        else:
            node_type = 2  # Cloud
        
        nodes.append({
            "cpu_util": np.random.uniform(0.2, 0.9),
            "mem_util": np.random.uniform(0.3, 0.85),
            "energy": np.random.uniform(20, 150),
            "latency": np.random.uniform(1, 50) if node_type > 0 else np.random.uniform(10, 100),
            "bandwidth": np.random.uniform(50, 500),
            "queue_len": np.random.randint(0, 15),
            "node_type": node_type
        })
    
    # Create edges (simplified hierarchy)
    edges = []
    for i in range(num_nodes - 1):
        edges.append([i, i + 1])
    
    return nodes, edges


def make_prediction(nodes: List[Dict], edges: List[List[int]], model_type: str) -> Optional[Dict]:
    """Make prediction via API."""
    try:
        payload = {
            "network_state": {
                "nodes": nodes,
                "edges": edges
            },
            "model_type": model_type
        }
        
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


def plot_network_topology(nodes: List[Dict], edges: List[List[int]], selected_node: Optional[int] = None):
    """Visualize network topology."""
    # Create node positions (simple layout)
    num_nodes = len(nodes)
    positions = {}
    
    # Arrange nodes by type
    sensors = [i for i, n in enumerate(nodes) if n['node_type'] == 0]
    fogs = [i for i, n in enumerate(nodes) if n['node_type'] == 1]
    clouds = [i for i, n in enumerate(nodes) if n['node_type'] == 2]
    
    # Position nodes in layers
    y_pos = 0
    for i, sensor in enumerate(sensors):
        positions[sensor] = (i * 2, 0)
    
    for i, fog in enumerate(fogs):
        positions[fog] = (i * 3 + 1, 5)
    
    for i, cloud in enumerate(clouds):
        positions[cloud] = (i * 4 + 2, 10)
    
    # Create edges
    edge_x = []
    edge_y = []
    for edge in edges:
        x0, y0 = positions.get(edge[0], (0, 0))
        x1, y1 = positions.get(edge[1], (0, 0))
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create node traces
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    node_sizes = []
    
    for i, node in enumerate(nodes):
        x, y = positions.get(i, (0, 0))
        node_x.append(x)
        node_y.append(y)
        
        # Color by type
        if node['node_type'] == 0:
            color = 'lightblue'
            type_name = 'Sensor'
        elif node['node_type'] == 1:
            color = 'lightgreen'
            type_name = 'Fog'
        else:
            color = 'orange'
            type_name = 'Cloud'
        
        # Highlight selected node
        if i == selected_node:
            color = 'red'
            node_sizes.append(30)
        else:
            node_sizes.append(20)
        
        node_colors.append(color)
        node_text.append(f"Node {i}<br>{type_name}<br>CPU: {node['cpu_util']:.2f}<br>Queue: {node['queue_len']}")
    
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1, color='gray'),
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        text=[f"N{i}" for i in range(len(nodes))],
        textposition="middle center",
        hovertext=node_text,
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        title="Network Topology",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500
    )
    
    return fig


def plot_node_metrics(nodes: List[Dict]):
    """Plot node resource metrics."""
    df = pd.DataFrame(nodes)
    df['node_id'] = range(len(nodes))
    df['node_type_name'] = df['node_type'].map({0: 'Sensor', 1: 'Fog', 2: 'Cloud'})
    
    # CPU and Memory utilization
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=df['node_id'],
        y=df['cpu_util'],
        name='CPU Utilization',
        marker_color='lightblue'
    ))
    fig1.add_trace(go.Bar(
        x=df['node_id'],
        y=df['mem_util'],
        name='Memory Utilization',
        marker_color='lightgreen'
    ))
    fig1.update_layout(
        title='Resource Utilization by Node',
        xaxis_title='Node ID',
        yaxis_title='Utilization',
        barmode='group',
        height=300
    )
    
    # Latency and Queue Length
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df['node_id'],
        y=df['latency'],
        mode='lines+markers',
        name='Latency (ms)',
        yaxis='y1'
    ))
    fig2.add_trace(go.Scatter(
        x=df['node_id'],
        y=df['queue_len'],
        mode='lines+markers',
        name='Queue Length',
        yaxis='y2'
    ))
    fig2.update_layout(
        title='Latency and Queue Length',
        xaxis_title='Node ID',
        yaxis=dict(title='Latency (ms)'),
        yaxis2=dict(title='Queue Length', overlaying='y', side='right'),
        height=300
    )
    
    return fig1, fig2


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🌐 AI Edge Allocator Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Check API status
    health = check_api_health()
    
    if health:
        st.sidebar.markdown('<div class="success-box">✅ API Server: Connected</div>', unsafe_allow_html=True)
        st.sidebar.metric("Uptime", f"{health.get('uptime_seconds', 0):.0f}s")
    else:
        st.sidebar.markdown('<div class="error-box">❌ API Server: Disconnected</div>', unsafe_allow_html=True)
        st.sidebar.warning("Start API server: `python python_scripts/api/run_api.py`")
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    model_type = st.sidebar.selectbox(
        "Choose Model",
        ["hybrid", "dqn", "ppo", "hybrid_gat", "hybrid_attention"],
        index=0
    )
    
    # Check model status
    if health:
        models_status = get_models_status()
        model_loaded = models_status.get(model_type, False)
        
        if model_loaded:
            st.sidebar.success(f"✅ {model_type.upper()} loaded")
        else:
            st.sidebar.error(f"❌ {model_type.upper()} not loaded")
    
    # Network configuration
    st.sidebar.subheader("🌐 Network Config")
    num_nodes = st.sidebar.slider("Number of Nodes", 5, 20, 10)
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh Interval (s)", 1, 10, 5)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔮 Prediction", "📈 Analytics", "ℹ️ About"])
    
    # ========================================================================
    # TAB 1: OVERVIEW
    # ========================================================================
    with tab1:
        st.subheader("System Overview")
        
        if health:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("API Status", "🟢 Online", "Healthy")
            with col2:
                st.metric("API Version", health.get('version', 'N/A'))
            with col3:
                models_loaded = sum(health.get('models_loaded', {}).values())
                st.metric("Models Loaded", f"{models_loaded}/5")
            with col4:
                st.metric("Uptime", f"{health.get('uptime_seconds', 0)/60:.1f} min")
            
            # Models status
            st.subheader("📦 Models Status")
            models_df = pd.DataFrame([
                {"Model": k.upper(), "Status": "✅ Loaded" if v else "❌ Not Loaded"}
                for k, v in health.get('models_loaded', {}).items()
            ])
            st.dataframe(models_df, use_container_width=True, hide_index=True)
        else:
            st.error("⚠️ API server is not running. Please start it with: `python python_scripts/api/run_api.py`")
    
    # ========================================================================
    # TAB 2: PREDICTION
    # ========================================================================
    with tab2:
        st.subheader("🔮 Task Placement Prediction")
        
        if not health:
            st.warning("API server required for predictions")
        else:
            # Generate network
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if st.button("🎲 Generate Random Network", type="primary"):
                    st.session_state['network_generated'] = True
                    st.session_state['nodes'], st.session_state['edges'] = create_sample_network(num_nodes)
                
                if st.button("🚀 Run Prediction", type="secondary", disabled=not st.session_state.get('network_generated', False)):
                    with st.spinner("Making prediction..."):
                        result = make_prediction(
                            st.session_state['nodes'],
                            st.session_state['edges'],
                            model_type
                        )
                        
                        if result:
                            st.session_state['prediction_result'] = result
                            st.session_state['selected_node'] = result['selected_node']
                            st.success(f"✅ Prediction Complete!")
                        else:
                            st.error("Failed to get prediction")
            
            with col2:
                if st.session_state.get('prediction_result'):
                    result = st.session_state['prediction_result']
                    st.metric("Selected Node", result['selected_node'])
                    st.metric("Confidence", f"{result['confidence']:.3f}")
                    st.metric("Processing Time", f"{result['processing_time_ms']:.2f} ms")
            
            # Visualization
            if st.session_state.get('network_generated'):
                st.subheader("Network Topology")
                fig = plot_network_topology(
                    st.session_state['nodes'],
                    st.session_state['edges'],
                    st.session_state.get('selected_node')
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Node metrics
                st.subheader("Node Metrics")
                fig1, fig2 = plot_node_metrics(st.session_state['nodes'])
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Node details
                if st.session_state.get('selected_node') is not None:
                    selected = st.session_state['selected_node']
                    st.subheader(f"Selected Node: {selected}")
                    node_data = st.session_state['nodes'][selected]
                    
                    cols = st.columns(4)
                    cols[0].metric("CPU", f"{node_data['cpu_util']:.2%}")
                    cols[1].metric("Memory", f"{node_data['mem_util']:.2%}")
                    cols[2].metric("Latency", f"{node_data['latency']:.1f} ms")
                    cols[3].metric("Queue", int(node_data['queue_len']))
    
    # ========================================================================
    # TAB 3: ANALYTICS
    # ========================================================================
    with tab3:
        st.subheader("📈 Performance Analytics")
        
        st.info("🚧 Training history and comparison charts coming soon!")
        
        # Placeholder charts
        st.subheader("Model Performance Comparison")
        
        # Sample data
        comparison_data = pd.DataFrame({
            'Model': ['DQN', 'PPO', 'Hybrid', 'Hybrid GAT', 'Hybrid Attention'],
            'Mean Reward': [244.15, 241.87, 246.02, 248.5, 247.3],
            'Std Dev': [9.20, 11.84, 8.57, 8.2, 8.9]
        })
        
        fig = px.bar(comparison_data, x='Model', y='Mean Reward', error_y='Std Dev',
                    title='Model Performance Comparison')
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # TAB 4: ABOUT
    # ========================================================================
    with tab4:
        st.subheader("ℹ️ About AI Edge Allocator")
        
        st.markdown("""
        ### 🌐 Project Overview
        
        **AI Edge Allocator** is a reinforcement learning framework for optimizing
        resource allocation in IoT edge computing environments.
        
        ### 🎯 Key Features
        
        - **Hybrid RL Architecture**: Combines DQN and PPO with GNN encoding
        - **Advanced Encoders**: GAT, GraphSAGE, and Hybrid GNN options
        - **Real-time Inference**: FastAPI server for production deployment
        - **Interactive Dashboard**: Streamlit UI for monitoring and visualization
        
        ### 📊 Model Performance
        
        | Model | Mean Reward | Std Dev |
        |-------|-------------|---------|
        | DQN   | 244.15     | 9.20    |
        | PPO   | 241.87     | 11.84   |
        | Hybrid| **246.02** | **8.57**|
        
        ### 🔗 Links
        
        - **GitHub**: [Sirius-ashwak/DeepSea-IoT](https://github.com/Sirius-ashwak/DeepSea-IoT)
        - **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
        - **Documentation**: See README.md
        
        ### 👨‍💻 Author
        
        **Mohamed Ashwak** - [@Sirius-ashwak](https://github.com/Sirius-ashwak)
        
        ### 📄 License
        
        MIT License - See LICENSE file for details
        """)
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
