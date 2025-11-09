"""
Graph Neural Network Encoder for IoT Networks

Implements multiple GNN architectures:
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- GraphSAGE (Inductive representation learning)

Used to encode the network topology and node states into embeddings
for downstream RL agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GNNEncoder(nn.Module):
    """
    Base GNN encoder for graph-structured IoT network observations.
    
    Architecture:
        Input: Node features (N x F) + Edge index (2 x E)
        -> GNN layers (with skip connections)
        -> Global pooling
        -> MLP projection
        Output: Graph embedding (1 x D)
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 3,
        conv_type: str = "GCN",
        dropout: float = 0.1,
        use_edge_attr: bool = True,
        pool_type: str = "mean"
    ):
        """
        Args:
            node_feature_dim: Input feature dimension per node
            hidden_dim: Hidden layer dimension
            output_dim: Final embedding dimension
            num_layers: Number of GNN layers
            conv_type: 'GCN', 'GAT', or 'GraphSAGE'
            dropout: Dropout probability
            use_edge_attr: Whether to use edge attributes
            pool_type: Graph pooling ('mean', 'max', or 'sum')
        """
        super().__init__()
        
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.conv_type = conv_type
        self.dropout = dropout
        self.use_edge_attr = use_edge_attr
        self.pool_type = pool_type
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()  # Use LayerNorm instead of BatchNorm
        
        # First layer
        self.convs.append(self._create_conv_layer(node_feature_dim, hidden_dim))
        self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(self._create_conv_layer(hidden_dim, hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # MLP head for final projection
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        logger.info(f"Initialized {conv_type} encoder: "
                   f"{node_feature_dim} -> {hidden_dim} x {num_layers} -> {output_dim}")
    
    def _create_conv_layer(self, in_dim: int, out_dim: int) -> nn.Module:
        """Create a GNN convolution layer based on conv_type."""
        if self.conv_type == "GCN":
            return GCNConv(in_dim, out_dim)
        elif self.conv_type == "GAT":
            # Multi-head attention with 4 heads
            return GATConv(in_dim, out_dim // 4, heads=4, concat=True)
        elif self.conv_type == "GraphSAGE":
            return SAGEConv(in_dim, out_dim)
        else:
            raise ValueError(f"Unknown conv_type: {self.conv_type}")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GNN encoder.
        
        Args:
            x: Node features (N x F)
            edge_index: Edge connectivity (2 x E)
            edge_attr: Edge attributes (E x F_edge), optional
            batch: Batch assignment vector (N,) for batched graphs
        
        Returns:
            Graph-level embedding (B x D) where B is batch size
        """
        # Apply GNN layers with residual connections
        h = x
        for i, (conv, ln) in enumerate(zip(self.convs, self.layer_norms)):
            h_new = conv(h, edge_index)
            h_new = ln(h_new)  # LayerNorm works with (N, F) directly
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            
            # Skip connection (if dimensions match)
            if h.shape[-1] == h_new.shape[-1]:
                h = h + h_new
            else:
                h = h_new
        
        # Global graph pooling
        if batch is None:
            # Single graph: pool all nodes
            if self.pool_type == "mean":
                h_graph = h.mean(dim=0, keepdim=True)
            elif self.pool_type == "max":
                h_graph = h.max(dim=0, keepdim=True)[0]
            elif self.pool_type == "sum":
                h_graph = h.sum(dim=0, keepdim=True)
            else:
                raise ValueError(f"Unknown pool_type: {self.pool_type}")
        else:
            # Batched graphs: pool per graph
            if self.pool_type == "mean":
                h_graph = global_mean_pool(h, batch)
            elif self.pool_type == "max":
                h_graph = global_max_pool(h, batch)
            else:
                raise ValueError(f"Unknown pool_type for batched: {self.pool_type}")
        
        # Final MLP projection
        embedding = self.mlp(h_graph)
        
        return embedding
    
    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get per-node embeddings (before pooling).
        Useful for node-level tasks or visualization.
        
        Returns:
            Node embeddings (N x hidden_dim)
        """
        h = x
        for conv, ln in zip(self.convs, self.layer_norms):
            h = conv(h, edge_index)
            h = ln(h)  # LayerNorm works with (N, F) directly
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h


class HierarchicalGNNEncoder(nn.Module):
    """
    Hierarchical GNN that processes sensor, fog, and cloud nodes separately
    before combining them.
    
    This can capture the hierarchical structure of IoT networks explicitly.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 2,
        conv_type: str = "GCN"
    ):
        super().__init__()
        
        # Separate encoders for each hierarchy level
        self.sensor_encoder = GNNEncoder(
            node_feature_dim, hidden_dim, hidden_dim, num_layers, conv_type
        )
        self.fog_encoder = GNNEncoder(
            node_feature_dim, hidden_dim, hidden_dim, num_layers, conv_type
        )
        self.cloud_encoder = GNNEncoder(
            node_feature_dim, hidden_dim, hidden_dim, num_layers, conv_type
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_types: torch.Tensor,  # 0=sensor, 1=fog, 2=cloud
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            node_types: Integer tensor (N,) indicating node type
        """
        # Split nodes by type
        sensor_mask = (node_types == 0)
        fog_mask = (node_types == 1)
        cloud_mask = (node_types == 2)
        
        # Encode each hierarchy separately (simplified - in practice need subgraphs)
        # For full implementation, extract subgraphs per type
        sensor_emb = self.sensor_encoder(x, edge_index, batch=batch)
        fog_emb = self.fog_encoder(x, edge_index, batch=batch)
        cloud_emb = self.cloud_encoder(x, edge_index, batch=batch)
        
        # Concatenate and fuse
        combined = torch.cat([sensor_emb, fog_emb, cloud_emb], dim=-1)
        output = self.fusion(combined)
        
        return output


def build_gnn_encoder(config: dict) -> nn.Module:
    """
    Factory function to build GNN encoder from config.
    
    Args:
        config: Dict with keys 'node_feature_dim', 'hidden_dim', etc.
    
    Returns:
        Instantiated GNN encoder
    """
    return GNNEncoder(
        node_feature_dim=config.get('node_feature_dim', 6),
        hidden_dim=config.get('hidden_dim', 64),
        output_dim=config.get('output_dim', 128),
        num_layers=config.get('num_layers', 3),
        conv_type=config.get('conv_type', 'GCN'),
        dropout=config.get('dropout', 0.1),
        pool_type=config.get('pool_type', 'mean')
    )


if __name__ == "__main__":
    # Test GNN encoder
    torch.manual_seed(42)
    
    # Create dummy graph
    num_nodes = 20
    num_edges = 50
    node_features = 6
    
    x = torch.randn(num_nodes, node_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Single graph
    encoder = GNNEncoder(
        node_feature_dim=node_features,
        hidden_dim=64,
        output_dim=128,
        num_layers=3,
        conv_type="GCN"
    )
    
    embedding = encoder(x, edge_index)
    print(f"Single graph embedding shape: {embedding.shape}")  # (1, 128)
    
    # Batched graphs
    batch = torch.cat([torch.zeros(10, dtype=torch.long), torch.ones(10, dtype=torch.long)])
    embedding_batch = encoder(x, edge_index, batch=batch)
    print(f"Batched embedding shape: {embedding_batch.shape}")  # (2, 128)
    
    # Node embeddings
    node_emb = encoder.get_node_embeddings(x, edge_index)
    print(f"Node embeddings shape: {node_emb.shape}")  # (20, 64)
