"""
Efficient GAT Training with Real IoT Data
Optimized for large datasets with sampling and batching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class EfficientGATLayer(nn.Module):
    """Efficient GAT layer for large IoT networks."""
    
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
        N = x.size(0)
        
        # Linear transformation
        h = self.W(x).view(N, self.heads, self.head_dim)
        
        # Simplified attention for efficiency
        if edge_index.size(1) > 0:
            # Sample edges if too many
            if edge_index.size(1) > 1000:
                sample_idx = torch.randperm(edge_index.size(1))[:1000]
                edge_index = edge_index[:, sample_idx]
            
            # Attention mechanism
            edge_h = torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=-1)
            attention = (edge_h * self.a).sum(dim=-1)
            attention = self.leaky_relu(attention)
            
            # Apply attention
            out = h.clone()
            for i in range(min(len(edge_index[0]), 500)):  # Limit for efficiency
                src, dst = edge_index[0][i], edge_index[1][i]
                out[dst] += 0.1 * attention[i].unsqueeze(-1) * h[src]
        else:
            out = h
        
        out = out.view(N, -1)
        return self.dropout(out)


class CompactGATModel(nn.Module):
    """Compact GAT model for efficient training."""
    
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=1, heads=2):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        self.gat1 = EfficientGATLayer(input_dim, hidden_dim, heads)
        self.gat2 = EfficientGATLayer(hidden_dim, hidden_dim, heads)
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, edge_index):
        x = self.input_norm(x)
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        
        # Global pooling
        graph_embedding = x.mean(dim=0, keepdim=True)
        output = self.predictor(graph_embedding)
        
        return output, graph_embedding


def load_sample_data(csv_path, sample_size=5000):
    """Load and sample IoT data efficiently."""
    
    print(f"📊 Loading sample of {sample_size} records from IoT data...")
    
    # Load and sample data
    df = pd.read_csv(csv_path)
    print(f"  - Total records: {len(df):,}")
    
    # Sample data for efficiency
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"  - Sampled to: {len(df):,} records")
    
    # Clean column names
    df.columns = ['id', 'room_id', 'noted_date', 'temp', 'location']
    
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['noted_date'], format='%d-%m-%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['datetime']).reset_index(drop=True)
    
    # Extract simple features
    df['hour'] = df['datetime'].dt.hour
    df['is_outdoor'] = (df['location'] == 'Out').astype(int)
    
    # Normalize temperature
    df['temp_norm'] = (df['temp'] - df['temp'].mean()) / df['temp'].std()
    
    print(f"  - Processed records: {len(df):,}")
    print(f"  - Temperature range: {df['temp'].min():.1f}°C to {df['temp'].max():.1f}°C")
    print(f"  - Unique rooms: {df['room_id'].nunique()}")
    
    return df


def create_simple_graphs(df, max_graphs=50):
    """Create simple network graphs efficiently."""
    
    print(f"🏗️  Creating up to {max_graphs} network graphs...")
    
    # Group by hour for temporal graphs
    df['hour_group'] = df['datetime'].dt.floor('H')
    
    graphs = []
    
    for hour, hour_data in df.groupby('hour_group'):
        if len(hour_data) < 5 or len(graphs) >= max_graphs:
            continue
        
        # Limit nodes per graph for efficiency
        if len(hour_data) > 20:
            hour_data = hour_data.sample(n=20, random_state=42)
        
        # Node features: [temp_norm, is_outdoor, hour, room_count]
        room_counts = hour_data['room_id'].value_counts()
        hour_data['room_sensor_count'] = hour_data['room_id'].map(room_counts)
        
        nodes = []
        for _, row in hour_data.iterrows():
            node_features = [
                row['temp_norm'],
                row['is_outdoor'],
                row['hour'] / 24.0,
                min(row['room_sensor_count'] / 10.0, 1.0)  # Normalized
            ]
            nodes.append(node_features)
        
        node_tensor = torch.tensor(nodes, dtype=torch.float32)
        
        # Create simple edges (connect nearby temperatures)
        edges = []
        num_nodes = len(hour_data)
        
        for i in range(num_nodes):
            for j in range(i+1, min(i+6, num_nodes)):  # Connect to 5 nearest
                temp_diff = abs(hour_data.iloc[i]['temp'] - hour_data.iloc[j]['temp'])
                if temp_diff < 10:  # Similar temperatures
                    edges.extend([[i, j], [j, i]])
        
        # Ensure some connectivity
        if len(edges) == 0:
            for i in range(min(num_nodes-1, 10)):
                edges.extend([[i, i+1], [i+1, i]])
        
        edge_tensor = torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros((2, 0), dtype=torch.long)
        
        # Target: Energy efficiency based on temperature stability
        temp_std = hour_data['temp'].std()
        efficiency = max(0, 100 - temp_std * 2)  # Lower variance = higher efficiency
        target = torch.tensor([efficiency], dtype=torch.float32)
        
        graphs.append({
            'nodes': node_tensor,
            'edges': edge_tensor,
            'target': target,
            'metadata': {
                'hour': hour,
                'num_sensors': num_nodes,
                'avg_temp': hour_data['temp'].mean(),
                'temp_std': temp_std
            }
        })
    
    print(f"  - Created {len(graphs)} graphs")
    print(f"  - Avg nodes per graph: {np.mean([g['nodes'].shape[0] for g in graphs]):.1f}")
    
    return graphs


def train_efficient_gat():
    """Train GAT efficiently on real IoT data."""
    
    print("\n" + "="*60)
    print("🚀 EFFICIENT GAT TRAINING WITH REAL IOT DATA")
    print("="*60)
    
    # Load sample data
    df = load_sample_data("IOT-temp.csv", sample_size=5000)
    
    # Create graphs
    graphs = create_simple_graphs(df, max_graphs=100)
    
    if len(graphs) < 10:
        print("❌ Not enough graphs created")
        return
    
    # Split data
    split_idx = int(0.8 * len(graphs))
    train_graphs = graphs[:split_idx]
    test_graphs = graphs[split_idx:]
    
    print(f"\n📊 Data: {len(train_graphs)} train, {len(test_graphs)} test graphs")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CompactGATModel(
        input_dim=4,
        hidden_dim=32,
        output_dim=1,
        heads=2
    ).to(device)
    
    print(f"🤖 Model: {sum(p.numel() for p in model.parameters()):,} parameters on {device}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Training loop
    num_epochs = 50
    best_reward = -float('inf')
    results = []
    
    print(f"\n🎯 Training for {num_epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_rewards = []
        
        # Training
        for graph in train_graphs:
            nodes = graph['nodes'].to(device)
            edges = graph['edges'].to(device)
            target = graph['target'].to(device)
            
            optimizer.zero_grad()
            
            prediction, _ = model(nodes, edges)
            loss = criterion(prediction, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_rewards.append(prediction.item())
        
        # Validation
        model.eval()
        val_loss = 0
        val_rewards = []
        
        with torch.no_grad():
            for graph in test_graphs:
                nodes = graph['nodes'].to(device)
                edges = graph['edges'].to(device)
                target = graph['target'].to(device)
                
                prediction, _ = model(nodes, edges)
                loss = criterion(prediction, target)
                
                val_loss += loss.item()
                val_rewards.append(prediction.item())
        
        # Metrics
        avg_train_loss = train_loss / len(train_graphs)
        avg_val_loss = val_loss / len(test_graphs)
        avg_val_reward = np.mean(val_rewards)
        
        # Save best model
        if avg_val_reward > best_reward:
            best_reward = avg_val_reward
            
            model_dir = Path("models/phase3_gat")
            model_dir.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_reward': best_reward,
                'config': {
                    'input_dim': 4,
                    'hidden_dim': 32,
                    'heads': 2
                }
            }, model_dir / "efficient_gat_best.pt")
        
        # Log progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch:2d}: Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {avg_val_loss:.4f}, Val Reward = {avg_val_reward:.2f}")
        
        results.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_reward': avg_val_reward
        })
    
    print("\n" + "="*60)
    print("📈 EFFICIENT GAT TRAINING COMPLETE!")
    print("="*60)
    
    print(f"🏆 Best Validation Reward: {best_reward:.2f}")
    
    # Compare with baseline
    baseline_reward = 246.02
    improvement = ((best_reward - baseline_reward) / baseline_reward) * 100
    print(f"📊 vs GCN Baseline (246.02): {improvement:+.1f}%")
    
    # Interpretation
    if improvement > 5:
        print("✅ GAT shows significant improvement! Ready for production.")
    elif improvement > 0:
        print("⚠️  GAT shows modest improvement. Consider more training.")
    else:
        print("❌ GAT underperforms baseline. May need architecture tuning.")
    
    # Save results
    summary = {
        "experiment": "Efficient_GAT_Real_Data",
        "date": datetime.now().isoformat(),
        "data_source": "IOT-temp.csv (sampled)",
        "sample_size": len(df),
        "num_graphs": len(graphs),
        "best_reward": float(best_reward),
        "improvement_percent": float(improvement),
        "model_params": sum(p.numel() for p in model.parameters()),
        "results": results
    }
    
    results_path = Path("reports/efficient_gat_results.json")
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📁 Results: {results_path}")
    print(f"💾 Model: models/phase3_gat/efficient_gat_best.pt")
    
    return summary


if __name__ == "__main__":
    try:
        torch.manual_seed(42)
        np.random.seed(42)
        
        results = train_efficient_gat()
        
        print("\n✅ Efficient GAT training completed!")
        print("\n💡 Next steps:")
        print("1. Compare with GCN on same sampled data")
        print("2. Scale up if results are promising")
        print("3. Deploy to production API")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
