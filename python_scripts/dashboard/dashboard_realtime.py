"""
Real-time Streaming Dashboard for AI Edge Allocator with MQTT Integration

This version connects to MQTT broker and displays live IoT device telemetry
and allocation decisions in real-time.

Usage:
    streamlit run dashboard_realtime.py
    streamlit run dashboard_realtime.py --server.port 8502
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import json
from typing import Dict, List, Optional
import sys

try:
    import paho.mqtt.client as mqtt
except ImportError:
    st.error("❌ paho-mqtt not installed. Install with: pip install paho-mqtt")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Edge Allocator - Real-time Dashboard",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .live-indicator {
        background-color: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
    .offline-indicator {
        background-color: #dc3545;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'mqtt_connected' not in st.session_state:
    st.session_state.mqtt_connected = False
if 'device_states' not in st.session_state:
    st.session_state.device_states = {}
if 'message_count' not in st.session_state:
    st.session_state.message_count = 0
if 'allocation_history' not in st.session_state:
    st.session_state.allocation_history = []
if 'mqtt_client' not in st.session_state:
    st.session_state.mqtt_client = None


def mqtt_on_connect(client, userdata, flags, rc):
    """MQTT connection callback."""
    if rc == 0:
        st.session_state.mqtt_connected = True
        # Subscribe to batch topic
        client.subscribe("iot/devices/batch", qos=1)
    else:
        st.session_state.mqtt_connected = False


def mqtt_on_message(client, userdata, msg):
    """MQTT message callback."""
    try:
        payload = json.loads(msg.payload.decode())
        
        # Update device states
        if "/batch" in msg.topic:
            devices = payload.get("devices", [])
            for device in devices:
                device_id = device.get("device_id")
                if device_id is not None:
                    st.session_state.device_states[device_id] = device
        
        st.session_state.message_count += 1
        
    except Exception as e:
        st.error(f"Error processing message: {e}")


def mqtt_on_disconnect(client, userdata, rc):
    """MQTT disconnect callback."""
    st.session_state.mqtt_connected = False


def connect_mqtt(broker: str, port: int):
    """Connect to MQTT broker."""
    if st.session_state.mqtt_client is not None:
        try:
            st.session_state.mqtt_client.disconnect()
        except:
            pass
    
    try:
        client = mqtt.Client(client_id=f"dashboard_{int(time.time())}")
        client.on_connect = mqtt_on_connect
        client.on_message = mqtt_on_message
        client.on_disconnect = mqtt_on_disconnect
        
        client.connect(broker, port, keepalive=60)
        client.loop_start()
        
        st.session_state.mqtt_client = client
        
        # Wait for connection
        time.sleep(1)
        
        return st.session_state.mqtt_connected
    except Exception as e:
        st.error(f"Connection error: {e}")
        return False


def disconnect_mqtt():
    """Disconnect from MQTT broker."""
    if st.session_state.mqtt_client is not None:
        try:
            st.session_state.mqtt_client.loop_stop()
            st.session_state.mqtt_client.disconnect()
        except:
            pass
        st.session_state.mqtt_client = None
        st.session_state.mqtt_connected = False


def plot_device_metrics(devices: List[Dict]):
    """Plot device metrics in real-time."""
    if not devices:
        st.info("⏳ Waiting for device data...")
        return
    
    # Prepare data
    device_ids = []
    cpu_utils = []
    mem_utils = []
    energies = []
    latencies = []
    queue_lens = []
    
    for device in devices:
        device_ids.append(f"D{device['device_id']}")
        metrics = device.get('metrics', {})
        cpu_utils.append(metrics.get('cpu_util', 0))
        mem_utils.append(metrics.get('mem_util', 0))
        energies.append(metrics.get('energy', 0))
        latencies.append(metrics.get('latency', 0))
        queue_lens.append(metrics.get('queue_len', 0))
    
    # Create subplots
    col1, col2 = st.columns(2)
    
    with col1:
        # CPU and Memory utilization
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=device_ids,
            y=cpu_utils,
            name='CPU Util',
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            x=device_ids,
            y=mem_utils,
            name='Memory Util',
            marker_color='lightcoral'
        ))
        fig.update_layout(
            title="Resource Utilization",
            xaxis_title="Device",
            yaxis_title="Utilization (%)",
            barmode='group',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Queue length
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=device_ids,
            y=queue_lens,
            marker_color='lightgreen'
        ))
        fig.update_layout(
            title="Queue Length",
            xaxis_title="Device",
            yaxis_title="Queue Size",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Energy and Latency
    col3, col4 = st.columns(2)
    
    with col3:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=device_ids,
            y=energies,
            mode='lines+markers',
            marker_color='orange',
            line=dict(width=2)
        ))
        fig.update_layout(
            title="Energy Consumption",
            xaxis_title="Device",
            yaxis_title="Energy (W)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=device_ids,
            y=latencies,
            mode='lines+markers',
            marker_color='purple',
            line=dict(width=2)
        ))
        fig.update_layout(
            title="Network Latency",
            xaxis_title="Device",
            yaxis_title="Latency (ms)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)


def plot_network_topology(devices: List[Dict]):
    """Visualize network topology."""
    if not devices:
        return
    
    # Create node positions
    positions = {}
    sensors = [d for d in devices if d.get('device_type') == 0]
    fogs = [d for d in devices if d.get('device_type') == 1]
    clouds = [d for d in devices if d.get('device_type') == 2]
    
    # Position nodes in layers
    for i, device in enumerate(sensors):
        positions[device['device_id']] = (i * 2, 0)
    
    for i, device in enumerate(fogs):
        positions[device['device_id']] = (i * 3 + 1, 5)
    
    for i, device in enumerate(clouds):
        positions[device['device_id']] = (i * 4 + 2, 10)
    
    # Create edges (simple hierarchy)
    edge_x = []
    edge_y = []
    for i in range(len(devices) - 1):
        id1 = devices[i]['device_id']
        id2 = devices[i + 1]['device_id']
        if id1 in positions and id2 in positions:
            x0, y0 = positions[id1]
            x1, y1 = positions[id2]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
    
    # Create node traces
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    
    for device in devices:
        device_id = device['device_id']
        if device_id in positions:
            x, y = positions[device_id]
            node_x.append(x)
            node_y.append(y)
            
            # Color by type
            dtype = device.get('device_type', 0)
            colors = {0: 'lightblue', 1: 'lightgreen', 2: 'orange'}
            types = {0: 'Sensor', 1: 'Fog', 2: 'Cloud'}
            node_colors.append(colors[dtype])
            
            metrics = device.get('metrics', {})
            node_text.append(
                f"Device {device_id}<br>"
                f"{types[dtype]}<br>"
                f"CPU: {metrics.get('cpu_util', 0):.2f}<br>"
                f"Queue: {metrics.get('queue_len', 0)}"
            )
    
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
        marker=dict(size=20, color=node_colors, line=dict(width=2, color='white')),
        text=[f"D{devices[i]['device_id']}" for i in range(len(node_x))],
        textposition="middle center",
        hovertext=node_text,
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        title="Network Topology (Live)",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


# Main app
def main():
    st.markdown('<h1 class="main-header">🌐 AI Edge Allocator - Real-time Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar - MQTT Configuration
    st.sidebar.header("⚙️ MQTT Configuration")
    
    broker = st.sidebar.text_input("Broker Address", value="localhost")
    port = st.sidebar.number_input("Port", min_value=1, max_value=65535, value=1883)
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔌 Connect", use_container_width=True):
            if connect_mqtt(broker, port):
                st.success("Connected!")
            else:
                st.error("Connection failed")
    
    with col2:
        if st.button("🔴 Disconnect", use_container_width=True):
            disconnect_mqtt()
            st.info("Disconnected")
    
    # Connection status
    st.sidebar.markdown("---")
    if st.session_state.mqtt_connected:
        st.sidebar.markdown('<div class="live-indicator">🟢 LIVE</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="offline-indicator">🔴 OFFLINE</div>', unsafe_allow_html=True)
    
    # Statistics
    st.sidebar.markdown("---")
    st.sidebar.metric("Messages Received", st.session_state.message_count)
    st.sidebar.metric("Devices Tracked", len(st.session_state.device_states))
    st.sidebar.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
    
    # Auto-refresh
    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
    
    if auto_refresh:
        time.sleep(5)
        st.rerun()
    
    # Main content
    if not st.session_state.mqtt_connected:
        st.warning("⚠️ Not connected to MQTT broker. Configure and connect in the sidebar.")
        st.info("""
        **Quick Start:**
        1. Start MQTT broker: `docker run -d -p 1883:1883 eclipse-mosquitto`
        2. Start IoT simulator: `python iot_device_simulator.py`
        3. Click 'Connect' in the sidebar
        """)
        return
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Live Metrics", "🌐 Network Topology", "📈 Analytics"])
    
    with tab1:
        st.header("Real-time Device Metrics")
        devices = list(st.session_state.device_states.values())
        
        if devices:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            avg_cpu = np.mean([d.get('metrics', {}).get('cpu_util', 0) for d in devices])
            avg_mem = np.mean([d.get('metrics', {}).get('mem_util', 0) for d in devices])
            total_energy = np.sum([d.get('metrics', {}).get('energy', 0) for d in devices])
            avg_queue = np.mean([d.get('metrics', {}).get('queue_len', 0) for d in devices])
            
            col1.metric("Avg CPU Utilization", f"{avg_cpu:.1%}")
            col2.metric("Avg Memory Utilization", f"{avg_mem:.1%}")
            col3.metric("Total Energy", f"{total_energy:.1f} W")
            col4.metric("Avg Queue Length", f"{avg_queue:.1f}")
            
            st.markdown("---")
            
            # Plots
            plot_device_metrics(devices)
        else:
            st.info("⏳ Waiting for device data from MQTT...")
    
    with tab2:
        st.header("Network Topology")
        devices = list(st.session_state.device_states.values())
        plot_network_topology(devices)
        
        if devices:
            # Device table
            st.subheader("Device Details")
            df_data = []
            for d in devices:
                metrics = d.get('metrics', {})
                df_data.append({
                    'Device ID': d['device_id'],
                    'Type': d.get('type_name', 'unknown'),
                    'CPU': f"{metrics.get('cpu_util', 0):.2%}",
                    'Memory': f"{metrics.get('mem_util', 0):.2%}",
                    'Energy': f"{metrics.get('energy', 0):.1f}W",
                    'Latency': f"{metrics.get('latency', 0):.1f}ms",
                    'Queue': metrics.get('queue_len', 0)
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
    
    with tab3:
        st.header("Analytics & History")
        st.info("Historical analytics coming soon! Currently showing live data only.")
        
        if st.session_state.device_states:
            # Device type distribution
            devices = list(st.session_state.device_states.values())
            type_counts = {0: 0, 1: 0, 2: 0}
            for d in devices:
                dtype = d.get('device_type', 0)
                type_counts[dtype] = type_counts.get(dtype, 0) + 1
            
            fig = go.Figure(data=[go.Pie(
                labels=['Sensors', 'Fog Nodes', 'Cloud Nodes'],
                values=[type_counts[0], type_counts[1], type_counts[2]],
                hole=0.3
            )])
            fig.update_layout(title="Device Type Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
