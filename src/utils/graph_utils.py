"""
Graph Construction Utilities for IoT Networks

This module provides tools to:
- Build graph representations of IoT/fog topologies
- Create edge connectivity based on network structure
- Convert to PyTorch Geometric Data format
- Visualize network graphs
"""

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IoTGraphBuilder:
    """
    Constructs graph representations of IoT edge computing networks.
    
    Graph Structure:
    - Nodes: sensors, fog devices, cloud servers
    - Edges: network connections with latency/bandwidth attributes
    """
    
    def __init__(self, topology_config: Optional[Dict] = None):
        """
        Args:
            topology_config: Dict with keys 'num_fog_nodes', 'num_sensors', etc.
        """
        self.topology_config = topology_config or {}
        self.graph = None
        self.node_mapping = {}  # node_id -> integer index
        
    def build_hierarchical_topology(
        self,
        num_sensors: int = 8,
        num_fog: int = 10,
        num_cloud: int = 1
    ) -> nx.Graph:
        """
        Build a hierarchical IoT topology: sensors -> fog -> cloud.
        
        Args:
            num_sensors: Number of IoT sensors
            num_fog: Number of fog nodes
            num_cloud: Number of cloud servers
        
        Returns:
            NetworkX graph with hierarchical structure
        """
        G = nx.Graph()
        
        # Add nodes with hierarchy attribute
        sensor_ids = [f"sensor_{i}" for i in range(num_sensors)]
        fog_ids = [f"fog_{i}" for i in range(num_fog)]
        cloud_ids = [f"cloud_{i}" for i in range(num_cloud)]
        
        # Add sensors (level 2)
        for sid in sensor_ids:
            G.add_node(sid, level=2, type="sensor", cpu=100, memory=128)
        
        # Add fog nodes (level 1)
        for fid in fog_ids:
            G.add_node(fid, level=1, type="fog", cpu=2800, memory=4000)
        
        # Add cloud (level 0)
        for cid in cloud_ids:
            G.add_node(cid, level=0, type="cloud", cpu=40000, memory=40000)
        
        # Connect sensors to fog nodes (round-robin)
        for i, sid in enumerate(sensor_ids):
            fog_id = fog_ids[i % num_fog]
            G.add_edge(sid, fog_id, bandwidth=50, latency=6.0)
        
        # Connect fog nodes to cloud (star topology)
        for fid in fog_ids:
            for cid in cloud_ids:
                G.add_edge(fid, cid, bandwidth=200, latency=10.0)
        
        # Connect fog nodes to each other (mesh)
        for i in range(num_fog):
            for j in range(i + 1, num_fog):
                G.add_edge(fog_ids[i], fog_ids[j], bandwidth=100, latency=5.0)
        
        self.graph = G
        self._create_node_mapping()
        
        logger.info(f"Built topology: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def _create_node_mapping(self):
        """Create bidirectional mapping between node IDs and integer indices."""
        if self.graph is None:
            raise ValueError("Graph not built yet. Call build_hierarchical_topology() first.")
        
        self.node_mapping = {node: i for i, node in enumerate(self.graph.nodes())}
        self.reverse_mapping = {i: node for node, i in self.node_mapping.items()}
    
    def to_pytorch_geometric(
        self,
        node_features: np.ndarray,
        edge_features: Optional[np.ndarray] = None
    ) -> Data:
        """
        Convert graph to PyTorch Geometric Data object.
        
        Args:
            node_features: Node feature matrix (N x F)
            edge_features: Edge feature matrix (E x F_edge), optional
        
        Returns:
            PyTorch Geometric Data object
        """
        if self.graph is None:
            raise ValueError("Graph not built yet.")
        
        # Convert to edge index format (2 x E)
        edge_index = []
        for u, v in self.graph.edges():
            u_idx = self.node_mapping[u]
            v_idx = self.node_mapping[v]
            edge_index.append([u_idx, v_idx])
            edge_index.append([v_idx, u_idx])  # undirected -> bidirectional
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # Convert node features
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index)
        
        # Add edge attributes if provided
        if edge_features is not None:
            # Duplicate for bidirectional edges
            edge_attr = np.vstack([edge_features, edge_features])
            data.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Add edge attributes from graph (bandwidth, latency)
        if not edge_features:
            edge_attrs = []
            for u, v in self.graph.edges():
                attrs = self.graph[u][v]
                edge_attrs.append([attrs.get('bandwidth', 100), attrs.get('latency', 5.0)])
                edge_attrs.append([attrs.get('bandwidth', 100), attrs.get('latency', 5.0)])
            data.edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        return data
    
    def get_adjacency_matrix(self, weighted: bool = False) -> np.ndarray:
        """
        Get adjacency matrix representation.
        
        Args:
            weighted: If True, use edge weights (latency)
        
        Returns:
            Adjacency matrix (N x N)
        """
        if self.graph is None:
            raise ValueError("Graph not built yet.")
        
        if weighted:
            return nx.to_numpy_array(self.graph, weight='latency')
        else:
            return nx.to_numpy_array(self.graph)
    
    def compute_shortest_paths(self) -> Dict[Tuple[str, str], float]:
        """
        Compute all-pairs shortest paths based on latency.
        
        Returns:
            Dict mapping (source, target) -> shortest path latency
        """
        if self.graph is None:
            raise ValueError("Graph not built yet.")
        
        return dict(nx.all_pairs_dijkstra_path_length(self.graph, weight='latency'))
    
    def visualize(self, save_path: Optional[str] = None):
        """
        Visualize the network topology.
        
        Args:
            save_path: If provided, save figure to this path
        """
        if self.graph is None:
            raise ValueError("Graph not built yet.")
        
        import matplotlib.pyplot as plt
        
        # Position nodes by hierarchy level
        pos = nx.spring_layout(self.graph, k=2, iterations=50)
        
        # Color by node type
        node_colors = []
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node]['type']
            if node_type == 'sensor':
                node_colors.append('lightblue')
            elif node_type == 'fog':
                node_colors.append('orange')
            else:  # cloud
                node_colors.append('red')
        
        plt.figure(figsize=(12, 8))
        nx.draw(
            self.graph,
            pos,
            node_color=node_colors,
            with_labels=True,
            node_size=500,
            font_size=8,
            edge_color='gray',
            alpha=0.7
        )
        plt.title("IoT Edge Computing Network Topology")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved topology visualization to {save_path}")
        else:
            plt.show()


def create_batch_from_snapshots(
    graphs: List[Data],
    batch_size: int
) -> List[Data]:
    """
    Create mini-batches from list of graph snapshots.
    
    Args:
        graphs: List of PyTorch Geometric Data objects
        batch_size: Number of graphs per batch
    
    Returns:
        List of batched Data objects
    """
    from torch_geometric.data import Batch
    
    batches = []
    for i in range(0, len(graphs), batch_size):
        batch = Batch.from_data_list(graphs[i:i + batch_size])
        batches.append(batch)
    
    return batches


def extract_node_features(
    df: pd.DataFrame,
    node_ids: List[str],
    feature_cols: List[str]
) -> np.ndarray:
    """
    Extract node feature matrix from dataframe.
    
    Args:
        df: DataFrame with node data
        node_ids: Ordered list of node IDs
        feature_cols: List of feature column names
    
    Returns:
        Feature matrix (N x F) in node_ids order
    """
    features = []
    for node_id in node_ids:
        node_data = df[df['node_id'] == node_id]
        if len(node_data) == 0:
            # Missing node: use zeros
            features.append(np.zeros(len(feature_cols)))
        else:
            # Use most recent data if multiple records
            features.append(node_data[feature_cols].values[-1])
    
    return np.array(features)


def compute_graph_statistics(graph: nx.Graph) -> Dict:
    """
    Compute graph-level statistics.
    
    Returns:
        Dict with metrics like degree, clustering, centrality
    """
    stats = {
        'num_nodes': graph.number_of_nodes(),
        'num_edges': graph.number_of_edges(),
        'density': nx.density(graph),
        'avg_degree': np.mean([d for n, d in graph.degree()]),
        'avg_clustering': nx.average_clustering(graph),
        'diameter': nx.diameter(graph) if nx.is_connected(graph) else -1,
    }
    
    return stats


if __name__ == "__main__":
    # Example usage
    builder = IoTGraphBuilder()
    
    # Build topology
    graph = builder.build_hierarchical_topology(
        num_sensors=8,
        num_fog=10,
        num_cloud=1
    )
    
    # Print statistics
    stats = compute_graph_statistics(graph)
    print("\n=== Graph Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Convert to PyTorch Geometric
    num_nodes = graph.number_of_nodes()
    dummy_features = np.random.randn(num_nodes, 6)  # 6 features per node
    
    pyg_data = builder.to_pytorch_geometric(dummy_features)
    print(f"\nPyTorch Geometric Data:")
    print(f"  Nodes: {pyg_data.x.shape}")
    print(f"  Edges: {pyg_data.edge_index.shape}")
    print(f"  Edge attributes: {pyg_data.edge_attr.shape}")
    
    # Visualize (optional)
    # builder.visualize(save_path="reports/topology.png")
