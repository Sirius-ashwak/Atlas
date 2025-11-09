"""
Network Topology Visualization

Visualize the IoT network graph structure and resource allocation decisions.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.graph_utils import IoTGraphBuilder

class NetworkVisualizer:
    """Visualize IoT network topology."""
    
    def __init__(self, save_dir: str = "reports/figures"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.graph_builder = IoTGraphBuilder()
    
    def visualize_hierarchical_topology(self):
        """Visualize the hierarchical IoT network."""
        print("🌐 Creating network topology visualization...")
        
        # Build network
        graph = self.graph_builder.build_hierarchical_topology(
            num_sensors=8,
            num_fog=5,
            num_cloud=1
        )
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes with attributes
        for node_id, features in enumerate(graph['node_features']):
            node_type = features[4]  # Node type from features
            if node_type == 0:
                layer = 'sensor'
                color = '#3498db'  # Blue
            elif node_type == 1:
                layer = 'fog'
                color = '#e74c3c'  # Red
            else:
                layer = 'cloud'
                color = '#2ecc71'  # Green
            
            G.add_node(node_id, layer=layer, color=color)
        
        # Add edges
        edge_index = graph['edge_index']
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0][i], edge_index[1][i])
        
        # Create layout
        pos = self._hierarchical_layout(G)
        
        # Plot
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=2, ax=ax)
        
        # Draw nodes by layer
        for layer, color, label in [('sensor', '#3498db', 'Sensor Nodes'),
                                     ('fog', '#e74c3c', 'Fog Nodes'),
                                     ('cloud', '#2ecc71', 'Cloud Node')]:
            node_list = [n for n, d in G.nodes(data=True) if d['layer'] == layer]
            nx.draw_networkx_nodes(G, pos, nodelist=node_list, 
                                  node_color=color, node_size=800 if layer=='cloud' else 500,
                                  alpha=0.9, label=label, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        
        ax.set_title('IoT Hierarchical Network Topology\n(Sensors → Fog → Cloud)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=12)
        ax.axis('off')
        
        plt.tight_layout()
        
        save_path = self.save_dir / 'network_topology.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved to {save_path}")
        plt.show()
        
        # Print network stats
        print("\n" + "="*60)
        print("NETWORK STATISTICS")
        print("="*60)
        print(f"Total Nodes: {G.number_of_nodes()}")
        print(f"Total Edges: {G.number_of_edges()}")
        print(f"Sensor Nodes: {sum(1 for n, d in G.nodes(data=True) if d['layer']=='sensor')}")
        print(f"Fog Nodes: {sum(1 for n, d in G.nodes(data=True) if d['layer']=='fog')}")
        print(f"Cloud Nodes: {sum(1 for n, d in G.nodes(data=True) if d['layer']=='cloud')}")
        print(f"Average Degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
        print("="*60 + "\n")
    
    def _hierarchical_layout(self, G):
        """Create hierarchical layout for visualization."""
        pos = {}
        layers = {'sensor': [], 'fog': [], 'cloud': []}
        
        # Organize nodes by layer
        for node, data in G.nodes(data=True):
            layers[data['layer']].append(node)
        
        # Position nodes
        y_positions = {'sensor': 0, 'fog': 1, 'cloud': 2}
        
        for layer, nodes in layers.items():
            y = y_positions[layer]
            num_nodes = len(nodes)
            
            for i, node in enumerate(nodes):
                # Spread nodes horizontally
                x = (i - num_nodes/2) * (3.0 / max(num_nodes, 1))
                pos[node] = (x, y)
        
        return pos
    
    def visualize_resource_utilization(self):
        """Visualize node resource utilization heatmap."""
        print("📊 Creating resource utilization heatmap...")
        
        # Simulate resource data
        num_nodes = 17
        resources = ['CPU', 'Memory', 'Energy', 'Latency', 'Bandwidth', 'Queue']
        
        # Generate sample data (you can replace with actual model predictions)
        data = np.random.uniform(0.2, 0.9, size=(num_nodes, len(resources)))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(resources)))
        ax.set_yticks(np.arange(num_nodes))
        ax.set_xticklabels(resources)
        ax.set_yticklabels([f'Node {i}' for i in range(num_nodes)])
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Utilization (0-1)', rotation=270, labelpad=20, fontweight='bold')
        
        # Add values in cells
        for i in range(num_nodes):
            for j in range(len(resources)):
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                             ha="center", va="center", color="black" if data[i, j] > 0.5 else "white",
                             fontsize=8)
        
        ax.set_title('Node Resource Utilization Heatmap', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Resource Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Node ID', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.save_dir / 'resource_utilization.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
        plt.show()
    
    def visualize_allocation_patterns(self):
        """Visualize task allocation patterns across nodes."""
        print("📊 Creating allocation patterns visualization...")
        
        # Simulate allocation data
        num_nodes = 17
        num_timesteps = 100
        
        # Generate allocation counts (which nodes get tasks)
        allocation_counts = np.random.poisson(5, size=num_nodes)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        colors = ['#2ecc71' if c > np.median(allocation_counts) else '#3498db' 
                 for c in allocation_counts]
        bars = ax1.bar(range(num_nodes), allocation_counts, color=colors, 
                      alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_xlabel('Node ID', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Number of Tasks Allocated', fontsize=12, fontweight='bold')
        ax1.set_title('Task Distribution Across Nodes', fontsize=14, fontweight='bold')
        ax1.axhline(np.mean(allocation_counts), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(allocation_counts):.1f}')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Pie chart - node type distribution
        node_types = ['Sensor (50%)', 'Fog (40%)', 'Cloud (10%)']
        sizes = [50, 40, 10]
        colors_pie = ['#3498db', '#e74c3c', '#2ecc71']
        
        ax2.pie(sizes, labels=node_types, colors=colors_pie, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax2.set_title('Task Allocation by Node Type', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        save_path = self.save_dir / 'allocation_patterns.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved to {save_path}")
        plt.show()
    
    def run_full_visualization(self):
        """Run complete visualization pipeline."""
        print("\n" + "="*80)
        print("🎨 STARTING NETWORK VISUALIZATION")
        print("="*80 + "\n")
        
        self.visualize_hierarchical_topology()
        print()
        
        self.visualize_resource_utilization()
        print()
        
        self.visualize_allocation_patterns()
        
        print("\n" + "="*80)
        print("✅ VISUALIZATION COMPLETE!")
        print("="*80)
        print(f"\n📁 All visualizations saved to: {self.save_dir}\n")


if __name__ == "__main__":
    visualizer = NetworkVisualizer()
    visualizer.run_full_visualization()
