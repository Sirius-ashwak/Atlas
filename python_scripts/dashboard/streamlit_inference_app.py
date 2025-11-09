"""
Enhanced Streamlit Interface for AI Edge Allocator Model Inference
Real-time model inference with data input and visualization
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from datetime import datetime
import time
from typing import Dict, List, Optional, Tuple

# Page configuration
st.set_page_config(
    page_title="AI Edge Allocator - Model Inference",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS for sleek, modern premium UI
st.markdown("""
<style>
    /* Root variables */
    :root {
        --primary: #0f172a;
        --secondary: #1e293b;
        --accent: #3b82f6;
        --accent-light: #60a5fa;
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --border: #334155;
        --success: #10b981;
    }
    
    /* Overall App Background */
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1a1f35 100%);
        color: #f1f5f9;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1a1f35 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid #334155;
    }
    
    [data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    /* Main Header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #60a5fa 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -1px;
        text-transform: uppercase;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #60a5fa;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #3b82f6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* All text */
    p, span, label, div, .stMarkdown {
        color: #f1f5f9 !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
        font-weight: 700;
    }
    
    /* Buttons - Premium Style */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
        color: white !important;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.8rem 1.5rem !important;
        border: 1px solid #60a5fa;
        border-radius: 8px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5);
        transform: translateY(-2px);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Input fields */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
        border-radius: 6px;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus {
        border: 2px solid #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Slider */
    .stSlider [data-testid="stTickBar"] {
        background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%);
    }
    
    .stSlider > div > div > div {
        color: #f1f5f9;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #f1f5f9 !important;
        font-weight: 500;
    }
    
    /* Metrics */
    [data-testid="stMetricLabel"] {
        color: #cbd5e1 !important;
        font-weight: 500;
    }
    
    [data-testid="stMetricValue"] {
        color: #60a5fa !important;
        font-weight: 800;
        font-size: 2rem !important;
    }
    
    /* Info box */
    .stAlert {
        background-color: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 8px;
        color: #f1f5f9 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
        border-bottom: 2px solid #334155;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #cbd5e1;
        border-radius: 6px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        background-color: transparent;
        border-bottom: 3px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        color: #60a5fa !important;
        border-bottom: 3px solid #3b82f6 !important;
        background-color: rgba(59, 130, 246, 0.1);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(59, 130, 246, 0.05);
        color: #f1f5f9;
        border: 1px solid #334155;
        border-radius: 6px;
        font-weight: 600;
    }
    
    /* DataTable */
    .stDataFrame {
        background-color: #1e293b;
    }
    
    .dataframe {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
    }
    
    .dataframe th {
        background-color: #0f172a !important;
        color: #60a5fa !important;
        font-weight: 700;
    }
    
    .dataframe td {
        border-color: #334155 !important;
    }
    
    /* Divider */
    hr {
        border-color: #334155 !important;
        margin: 1.5rem 0;
    }
    
    /* JSON display */
    .stJson {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
    }
    
    /* Success message */
    .stSuccess {
        background-color: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid #10b981 !important;
        color: #86efac !important;
    }
    
    /* Warning message */
    .stWarning {
        background-color: rgba(251, 146, 60, 0.1) !important;
        border: 1px solid #fb923c !important;
        color: #fed7aa !important;
    }
    
    /* Error message */
    .stError {
        background-color: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid #ef4444 !important;
        color: #fca5a5 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'inference_history' not in st.session_state:
    st.session_state.inference_history = []
if 'network_data' not in st.session_state:
    st.session_state.network_data = None
if 'last_inference' not in st.session_state:
    st.session_state.last_inference = None


def check_api_health() -> Tuple[bool, Optional[Dict]]:
    """Check if API server is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except Exception as e:
        return False, {"error": str(e)}


def perform_inference(network_state: Dict) -> Optional[Dict]:
    """Send network state to API and get model inference."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/inference",
            json=network_state,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def generate_sample_network(num_nodes: int = 10, seed: int = 42) -> Dict:
    """Generate a sample IoT network for testing."""
    np.random.seed(seed)
    
    nodes = []
    for i in range(num_nodes):
        # Determine node type
        if i < num_nodes * 0.5:
            node_type = 0  # Sensor
            cpu = np.random.uniform(0.5, 1.5)
            memory = np.random.uniform(0.25, 0.75)
        elif i < num_nodes * 0.8:
            node_type = 1  # Fog
            cpu = np.random.uniform(2, 4)
            memory = np.random.uniform(2, 4)
        else:
            node_type = 2  # Cloud
            cpu = np.random.uniform(8, 16)
            memory = np.random.uniform(16, 32)
        
        node = {
            "id": i,
            "type": node_type,
            "resources": {
                "cpu": float(cpu),
                "memory": float(memory),
                "storage": float(memory * 2)
            },
            "latency_to_cloud": float(np.random.uniform(5, 100)),
            "energy_consumption": float(np.random.uniform(10, 50)),
            "reliability": float(np.random.uniform(0.9, 0.99))
        }
        nodes.append(node)
    
    # Create adjacency matrix (simplified)
    adjacency = np.random.choice([0, 1], size=(num_nodes, num_nodes), p=[0.7, 0.3])
    adjacency = (adjacency + adjacency.T) > 0  # Make symmetric
    np.fill_diagonal(adjacency, 0)  # No self-loops
    
    network_state = {
        "nodes": nodes,
        "adjacency_matrix": adjacency.tolist(),
        "current_time": datetime.now().isoformat(),
        "network_load": float(np.random.uniform(0.3, 0.8))
    }
    
    return network_state


def create_network_visualization(nodes: List[Dict], allocations: Optional[List[float]] = None):
    """Create an interactive network graph visualization."""
    G = nx.Graph()
    
    # Add nodes
    for i, node in enumerate(nodes):
        G.add_node(i, 
                  type=node['type'],
                  cpu=node['resources']['cpu'],
                  memory=node['resources']['memory'])
    
    # Add edges for visualization
    for i in range(len(nodes)):
        for j in range(i+1, min(i+3, len(nodes))):
            G.add_edge(i, j)
    
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    
    for node_id in G.nodes():
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        
        node_info = nodes[node_id]
        node_type = ['Sensor', 'Fog', 'Cloud'][node_info['type']]
        
        # Color based on allocation or type
        if allocations and node_id < len(allocations):
            node_colors.append(allocations[node_id])
        else:
            node_colors.append(node_info['type'] * 50)
        
        # Node text
        text = f"Node {node_id} ({node_type})<br>"
        text += f"CPU: {node_info['resources']['cpu']:.1f}<br>"
        text += f"Memory: {node_info['resources']['memory']:.1f}"
        if allocations and node_id < len(allocations):
            text += f"<br>Score: {allocations[node_id]:.2f}"
        node_text.append(text)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_text,
        text=[str(i) for i in range(len(nodes))],
        textposition="middle center",
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=15,
            color=node_colors,
            colorbar=dict(
                thickness=15,
                title="Score",
                xanchor='left'
            ),
            line=dict(width=2, color='white')
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title="IoT Network Topology",
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0, l=0, r=0, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=500
                   ))
    return fig


# Main Application
def main():
    st.markdown('<h1 class="main-header">AI EDGE ALLOCATOR | MODEL INFERENCE</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### CONFIGURATION")
        
        # API Health Check
        api_healthy, health_data = check_api_health()
        if api_healthy:
            st.success("API Server Connected")
            with st.expander("View API Details"):
                st.json(health_data)
        else:
            st.error("API Server Offline")
            st.info("Start the API with: `python python_scripts/api/run_api.py`")
        
        st.divider()
        
        # Model Information
        st.markdown("### MODEL INFORMATION")
        st.info("""
**Model:** Hybrid DQN-PPO-GNN  
**Performance:** 246.02 ± 8.57  
**Architecture:** GCN (3 layers)  
**Checkpoint:** 5,000 steps  
**Status:** Production Ready
        """)
        
        st.divider()
        
        # Data Input Options
        st.markdown("### DATA INPUT METHOD")
        input_method = st.radio(
            "Choose input method:",
            ["Generate Sample Data", "Upload JSON", "Manual Input"],
            label_visibility="collapsed"
        )
    
    # Main Content Area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="section-header">NETWORK DATA INPUT</div>', unsafe_allow_html=True)
        
        if input_method == "Generate Sample Data":
            st.markdown("#### Generate Test Network")
            
            num_nodes = st.slider("Number of Nodes", 5, 30, 10)
            seed = st.number_input("Random Seed", value=42)
            
            if st.button("Generate Network", key="generate"):
                network_data = generate_sample_network(num_nodes, seed)
                st.session_state.network_data = network_data
                st.success(f"Generated network with {num_nodes} nodes")
        
        elif input_method == "Upload JSON":
            st.markdown("#### Upload Network State")
            uploaded_file = st.file_uploader("Choose a JSON file", type="json")
            
            if uploaded_file is not None:
                try:
                    network_data = json.load(uploaded_file)
                    st.session_state.network_data = network_data
                    st.success("File uploaded successfully")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
        else:  # Manual Input
            st.markdown("#### Manual Network Configuration")
            
            with st.expander("Network Parameters"):
                num_sensors = st.number_input("Sensor Nodes", 1, 20, 5)
                num_fog = st.number_input("Fog Nodes", 1, 10, 3)
                num_cloud = st.number_input("Cloud Nodes", 1, 5, 2)
                network_load = st.slider("Network Load", 0.0, 1.0, 0.5)
            
            if st.button("Create Network", key="manual"):
                total_nodes = num_sensors + num_fog + num_cloud
                network_data = generate_sample_network(total_nodes)
                network_data['network_load'] = network_load
                st.session_state.network_data = network_data
                st.success("Network created manually")
        
        # Display current network data
        if st.session_state.network_data:
            st.divider()
            st.markdown("#### Current Network State")
            
            data = st.session_state.network_data
            st.metric("Total Nodes", len(data['nodes']))
            st.metric("Network Load", f"{data.get('network_load', 0):.1%}")
            
            with st.expander("View Raw Data"):
                st.json(data)
    
    with col2:
        st.markdown('<div class="section-header">MODEL INFERENCE</div>', unsafe_allow_html=True)
        
        if st.session_state.network_data:
            # Run Inference Button
            if st.button("RUN MODEL INFERENCE", type="primary", key="inference"):
                with st.spinner("Running inference..."):
                    start_time = time.time()
                    
                    # Simulate inference result (replace with actual API call)
                    result = perform_inference(st.session_state.network_data)
                    
                    # If API is not available, generate mock result
                    if result and "error" in result:
                        # Generate mock inference result for demo
                        num_nodes = len(st.session_state.network_data['nodes'])
                        result = {
                            "action": np.random.randint(0, num_nodes),
                            "confidence": np.random.uniform(0.7, 0.95),
                            "expected_reward": 246.02 + np.random.randn() * 8.57,
                            "allocations": np.random.uniform(0, 1, num_nodes).tolist(),
                            "metrics": {
                                "latency": np.random.uniform(10, 50),
                                "energy": np.random.uniform(20, 100),
                                "qos": np.random.uniform(0.8, 1.0),
                                "balance": np.random.uniform(0.6, 0.9)
                            }
                        }
                        st.warning("Using simulated inference (API offline)")
                    
                    inference_time = time.time() - start_time
                    
                    st.session_state.last_inference = result
                    st.session_state.inference_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "result": result,
                        "inference_time": inference_time
                    })
                    
                    st.success(f"Inference completed in {inference_time:.3f}s")
            
            # Display Results
            if st.session_state.last_inference:
                st.divider()
                st.markdown("#### Inference Results")
                
                result = st.session_state.last_inference
                
                # Metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Predicted Action", result.get("action", "N/A"))
                with col_b:
                    st.metric("Confidence", f"{result.get('confidence', 0):.1%}")
                with col_c:
                    st.metric("Expected Reward", f"{result.get('expected_reward', 0):.2f}")
                
                # Allocation details
                if "allocations" in result:
                    st.markdown("#### Node Allocations")
                    allocations_df = pd.DataFrame({
                        "Node ID": range(len(result["allocations"])),
                        "Allocation Score": result["allocations"],
                        "Recommended": ["Yes" if a > 0.5 else "No" for a in result["allocations"]]
                    })
                    st.dataframe(allocations_df, use_container_width=True)
                
                # Performance metrics
                if "metrics" in result:
                    st.markdown("#### Performance Metrics")
                    metrics = result["metrics"]
                    col_1, col_2 = st.columns(2)
                    with col_1:
                        st.metric("Latency", f"{metrics.get('latency', 0):.1f} ms")
                        st.metric("Energy", f"{metrics.get('energy', 0):.1f} W")
                    with col_2:
                        st.metric("QoS Score", f"{metrics.get('qos', 0):.2f}")
                        st.metric("Load Balance", f"{metrics.get('balance', 0):.1%}")
        else:
            st.info("← Please input network data first")
    
    # Visualization Section
    if st.session_state.network_data and st.session_state.last_inference:
        st.divider()
        st.markdown('<div class="section-header">VISUALIZATIONS</div>', unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Network Topology", "Allocation Heatmap", "Performance History"])
        
        with tab1:
            allocations = st.session_state.last_inference.get("allocations", None)
            fig = create_network_visualization(
                st.session_state.network_data["nodes"],
                allocations
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if "allocations" in st.session_state.last_inference:
                allocations = st.session_state.last_inference["allocations"]
                # Create heatmap
                fig = px.imshow(
                    [allocations],
                    labels=dict(x="Node ID", y="", color="Allocation Score"),
                    color_continuous_scale="RdYlGn",
                    aspect="auto"
                )
                fig.update_layout(title="Node Allocation Scores")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            if len(st.session_state.inference_history) > 0:
                # Extract history data
                history_df = pd.DataFrame([
                    {
                        "Time": h["timestamp"],
                        "Expected Reward": h["result"].get("expected_reward", 0),
                        "Inference Time (s)": h["inference_time"]
                    }
                    for h in st.session_state.inference_history
                ])
                
                # Plot reward history
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=history_df["Time"],
                    y=history_df["Expected Reward"],
                    mode='lines+markers',
                    name='Expected Reward',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig.update_layout(
                    title="Inference History",
                    xaxis_title="Time",
                    yaxis_title="Expected Reward",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show history table
                st.markdown("#### Inference History Table")
                st.dataframe(history_df, use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px; font-size: 0.9rem;'>
    <p><strong>Hybrid DQN-PPO-GNN Model</strong> | Performance: 246.02 ± 8.57 | Checkpoint: 5,000 steps | Production Ready</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
