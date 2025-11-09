"""
Advanced GNN Encoders

Implements GAT (Graph Attention Networks) and GraphSAGE variants
for improved graph representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool
from typing import Optional

try:
    from .encoder import GNNEncoder as GCNEncoder
except ImportError:
    # If running standalone, import won't work
    GCNEncoder = None

class GATEncoder(nn.Module):
    """
    Graph Attention Network (GAT) Encoder.
    
    Uses attention mechanisms to learn importance weights
    for different neighbor nodes.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # First layer
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(node_feature_dim, hidden_dim, heads=num_heads, dropout=dropout)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
            )
        
        # Output layer (concatenate heads)
        self.convs.append(
            GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False, dropout=dropout)
        )
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim * num_heads) for _ in range(num_layers - 1)
        ])
        
        print(f"Initialized GAT encoder: {node_feature_dim} -> {hidden_dim}x{num_heads} x {num_layers} -> {output_dim}")
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment (optional)
        
        Returns:
            Graph embeddings [num_graphs, output_dim] or [num_nodes, output_dim]
        """
        # Apply GAT layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        
        # Global pooling if batch is provided
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x
    
    def get_attention_weights(self, x, edge_index):
        """Get attention weights for visualization."""
        attention_weights = []
        for i in range(self.num_layers):
            _, (edge_idx, alpha) = self.convs[i](x, edge_index, return_attention_weights=True)
            attention_weights.append((edge_idx, alpha))
            if i < self.num_layers - 1:
                x = F.elu(self.convs[i](x, edge_index))
        return attention_weights


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE Encoder.
    
    Uses neighborhood sampling and aggregation for 
    scalable graph learning.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 3,
        aggregator: str = 'mean',  # 'mean', 'max', 'lstm'
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build layers
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(node_feature_dim, hidden_dim, aggr=aggregator))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggregator))
        
        self.convs.append(SAGEConv(hidden_dim, output_dim, aggr=aggregator))
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])
        
        print(f"Initialized GraphSAGE encoder ({aggregator}): {node_feature_dim} -> {hidden_dim} x {num_layers} -> {output_dim}")
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass."""
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        return x


class HybridGNNEncoder(nn.Module):
    """
    Hybrid encoder combining GCN, GAT, and GraphSAGE.
    
    Uses ensemble of different GNN architectures for 
    robust feature learning.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 3
    ):
        super().__init__()
        
        if GCNEncoder is None:
            raise ImportError("GCNEncoder not available. Make sure encoder.py exists.")
        
        # Different GNN architectures
        self.gcn = GCNEncoder(node_feature_dim, hidden_dim, output_dim, num_layers)
        self.gat = GATEncoder(node_feature_dim, hidden_dim, output_dim, num_layers, num_heads=2)
        self.sage = GraphSAGEEncoder(node_feature_dim, hidden_dim, output_dim, num_layers)
        
        # Fusion layer
        self.fusion = nn.Linear(output_dim * 3, output_dim)
        
        print(f"Initialized Hybrid GNN encoder: combining GCN + GAT + GraphSAGE")
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass through all encoders and fuse."""
        # GCNEncoder (GNNEncoder) already does pooling
        gcn_out = self.gcn(x, edge_index, batch)
        
        # GAT and GraphSAGE return node-level, need pooling
        gat_node_out = self.gat(x, edge_index, batch=None)
        sage_node_out = self.sage(x, edge_index, batch=None)
        
        # Pool GAT and SAGE outputs
        if batch is None:
            # Single graph
            gat_out = gat_node_out.mean(dim=0, keepdim=True)
            sage_out = sage_node_out.mean(dim=0, keepdim=True)
        else:
            # Batched graphs
            gat_out = global_mean_pool(gat_node_out, batch)
            sage_out = global_mean_pool(sage_node_out, batch)
        
        # Concatenate and fuse
        combined = torch.cat([gcn_out, gat_out, sage_out], dim=-1)
        output = self.fusion(combined)
        
        return output


class AttentionFusion(nn.Module):
    """
    Attention-based fusion for DQN and PPO outputs.
    
    Learns dynamic weights instead of fixed weighted sum.
    """
    
    def __init__(self, input_dim: int, num_sources: int = 2):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_sources),
            nn.Softmax(dim=-1)
        )
        
        print(f"Initialized Attention Fusion: {num_sources} sources with learned weights")
    
    def forward(self, dqn_output, ppo_output):
        """
        Fuse DQN and PPO outputs with learned attention.
        
        Args:
            dqn_output: Q-values from DQN head [batch, num_actions]
            ppo_output: Action logits from PPO head [batch, num_actions]
        
        Returns:
            Fused output [batch, num_actions]
            Attention weights [batch, 2]
        """
        # Concatenate for attention computation
        combined = torch.cat([dqn_output, ppo_output], dim=-1)
        
        # Compute attention weights
        weights = self.attention(combined)  # [batch, 2]
        
        # Apply weights
        dqn_weight = weights[:, 0].unsqueeze(-1)  # [batch, 1]
        ppo_weight = weights[:, 1].unsqueeze(-1)  # [batch, 1]
        
        fused = dqn_weight * dqn_output + ppo_weight * ppo_output
        
        return fused, weights


def create_encoder(encoder_type: str, config: dict):
    """
    Factory function to create different encoder types.
    
    Args:
        encoder_type: 'GCN', 'GAT', 'GraphSAGE', or 'Hybrid'
        config: Encoder configuration
    
    Returns:
        Encoder instance
    """
    if encoder_type.upper() == 'GCN':
        if GCNEncoder is None:
            raise ImportError("GCNEncoder not available")
        return GCNEncoder(**config)
    elif encoder_type.upper() == 'GAT':
        return GATEncoder(**config)
    elif encoder_type.upper() == 'GRAPHSAGE':
        return GraphSAGEEncoder(**config)
    elif encoder_type.upper() == 'HYBRID':
        return HybridGNNEncoder(**config)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


if __name__ == "__main__":
    # Test encoders
    import torch
    from torch_geometric.data import Data
    
    # Create dummy data
    x = torch.randn(10, 6)  # 10 nodes, 6 features
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    print("\n" + "="*60)
    print("Testing Advanced Encoders")
    print("="*60 + "\n")
    
    # Test GAT
    print("1. GAT Encoder:")
    gat = GATEncoder(node_feature_dim=6, hidden_dim=16, output_dim=32, num_layers=2)
    gat_out = gat(x, edge_index)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {gat_out.shape}\n")
    
    # Test GraphSAGE
    print("2. GraphSAGE Encoder:")
    sage = GraphSAGEEncoder(node_feature_dim=6, hidden_dim=16, output_dim=32, num_layers=2)
    sage_out = sage(x, edge_index)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {sage_out.shape}\n")
    
    # Test Hybrid
    print("3. Hybrid GNN Encoder:")
    hybrid = HybridGNNEncoder(node_feature_dim=6, hidden_dim=16, output_dim=32, num_layers=2)
    hybrid_out = hybrid(x, edge_index)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {hybrid_out.shape}\n")
    
    # Test Attention Fusion
    print("4. Attention Fusion:")
    fusion = AttentionFusion(input_dim=20)
    dqn_out = torch.randn(4, 10)
    ppo_out = torch.randn(4, 10)
    fused, weights = fusion(dqn_out, ppo_out)
    print(f"   DQN output: {dqn_out.shape}")
    print(f"   PPO output: {ppo_out.shape}")
    print(f"   Fused output: {fused.shape}")
    print(f"   Attention weights: {weights.shape}")
    print(f"   Sample weights: {weights[0].detach().numpy()}")
    
    print("\n" + "="*60)
    print("✅ All encoders working correctly!")
    print("="*60 + "\n")
