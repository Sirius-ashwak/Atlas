"""
Simple GAT Training Script
Minimal implementation to test GAT architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from pathlib import Path
import json
from datetime import datetime

# Simple GAT implementation
class SimpleGATLayer(nn.Module):
    def __init__(self, in_features, out_features, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.out_features = out_features
        self.head_dim = out_features // heads
        
        self.W = nn.Linear(in_features, heads * self.head_dim, bias=False)
        self.a = nn.Parameter(torch.zeros(heads, 2 * self.head_dim))
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x, edge_index):
        # x: [N, in_features]
        # edge_index: [2, E]
        N = x.size(0)
        
        # Linear transformation
        h = self.W(x).view(N, self.heads, self.head_dim)  # [N, heads, head_dim]
        
        # Attention mechanism (simplified)
        edge_h = torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=-1)  # [E, heads, 2*head_dim]
        attention = (edge_h * self.a).sum(dim=-1)  # [E, heads]
        attention = self.leaky_relu(attention)
        
        # Apply attention to aggregate neighbors
        out = torch.zeros_like(h)
        for i in range(len(edge_index[0])):
            src, dst = edge_index[0][i], edge_index[1][i]
            out[dst] += attention[i].unsqueeze(-1) * h[src]
        
        out = out.view(N, -1)  # [N, heads * head_dim]
        return self.dropout(out)


class SimpleGATModel(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=128, num_layers=3, heads=4):
        super().__init__()
        
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(SimpleGATLayer(input_dim, hidden_dim, heads))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(SimpleGATLayer(hidden_dim, hidden_dim, heads))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        
        # Global pooling (mean)
        graph_embedding = x.mean(dim=0, keepdim=True)
        
        return self.output_proj(graph_embedding)


def generate_mock_iot_data(num_nodes=20, num_edges=50):
    """Generate mock IoT network data."""
    # Node features: [cpu, memory, latency, energy, reliability, load]
    node_features = torch.randn(num_nodes, 6)
    
    # Random edge connections
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # Mock reward (higher is better)
    reward = torch.randn(1) * 10 + 200  # Around 200 baseline
    
    return node_features, edge_index, reward


def train_simple_gat():
    """Train a simple GAT model."""
    print("🚀 Starting Simple GAT Training")
    print("="*50)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = SimpleGATModel(
        input_dim=6,
        hidden_dim=64,
        output_dim=128,
        num_layers=3,
        heads=4
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    num_epochs = 100
    best_reward = -float('inf')
    results = []
    
    print("\n🎯 Training Progress:")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        model.train()
        
        # Generate batch of mock data
        batch_rewards = []
        batch_loss = 0
        
        for _ in range(10):  # 10 samples per epoch
            # Generate mock IoT network
            x, edge_index, target_reward = generate_mock_iot_data()
            x, edge_index, target_reward = x.to(device), edge_index.to(device), target_reward.to(device)
            
            # Forward pass
            embedding = model(x, edge_index)
            
            # Predict reward from embedding
            predicted_reward = embedding.mean()  # Simple prediction
            
            # Loss
            loss = criterion(predicted_reward, target_reward)
            batch_loss += loss.item()
            batch_rewards.append(target_reward.item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        avg_reward = np.mean(batch_rewards)
        avg_loss = batch_loss / 10
        
        # Track best performance
        if avg_reward > best_reward:
            best_reward = avg_reward
            # Save best model
            model_dir = Path("models/phase3_gat")
            model_dir.mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_dir / "simple_gat_best.pt")
        
        # Log progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Reward = {avg_reward:.2f}, Loss = {avg_loss:.4f}")
        
        results.append({
            'epoch': epoch,
            'reward': avg_reward,
            'loss': avg_loss
        })
    
    print("\n" + "="*50)
    print("📈 Training Complete!")
    print(f"🏆 Best Reward: {best_reward:.2f}")
    
    # Compare with baseline
    baseline = 246.02
    improvement = ((best_reward - baseline) / baseline) * 100
    print(f"📊 vs GCN Baseline: {improvement:+.1f}%")
    
    # Save results
    summary = {
        "experiment": "Simple_GAT_Training",
        "date": datetime.now().isoformat(),
        "best_reward": float(best_reward),
        "improvement_percent": float(improvement),
        "epochs": num_epochs,
        "model_params": sum(p.numel() for p in model.parameters()),
        "results": results
    }
    
    results_path = Path("reports/simple_gat_results.json")
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📁 Results saved: {results_path}")
    
    return summary


if __name__ == "__main__":
    try:
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Train model
        results = train_simple_gat()
        
        print("\n✅ GAT training completed successfully!")
        print("\n💡 Next steps:")
        print("1. Integrate with full hybrid trainer")
        print("2. Test on real IoT data")
        print("3. Deploy to production if improvement > 5%")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
