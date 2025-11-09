"""
Train GAT Model with Real IoT Temperature/Humidity Data
Uses actual sensor data from IOT-temp.csv
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
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class RealDataGATLayer(nn.Module):
    """GAT layer optimized for real IoT sensor data."""
    
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
        
        # Attention mechanism
        edge_h = torch.cat([h[edge_index[0]], h[edge_index[1]]], dim=-1)
        attention = (edge_h * self.a).sum(dim=-1)
        attention = self.leaky_relu(attention)
        
        # Softmax attention weights per node
        attention_weights = torch.zeros(N, self.heads, device=x.device)
        for i in range(len(edge_index[0])):
            src, dst = edge_index[0][i], edge_index[1][i]
            attention_weights[dst] += torch.softmax(attention[i], dim=0)
        
        # Apply attention to aggregate neighbors
        out = torch.zeros_like(h)
        for i in range(len(edge_index[0])):
            src, dst = edge_index[0][i], edge_index[1][i]
            out[dst] += attention_weights[dst].unsqueeze(-1) * h[src]
        
        out = out.view(N, -1)
        return self.dropout(out)


class IoTGATModel(nn.Module):
    """GAT model for IoT sensor network optimization."""
    
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1, num_layers=3, heads=4):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        self.layers = nn.ModuleList()
        
        # First layer
        self.layers.append(RealDataGATLayer(input_dim, hidden_dim, heads))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(RealDataGATLayer(hidden_dim, hidden_dim, heads))
        
        # Output layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x, edge_index):
        # Normalize input
        x = self.input_norm(x)
        
        # GAT layers
        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)
        
        # Global pooling
        graph_embedding = x.mean(dim=0, keepdim=True)
        
        # Predict efficiency/reward
        output = self.predictor(graph_embedding)
        
        return output, graph_embedding


def load_and_process_iot_data(csv_path):
    """Load and process real IoT temperature/humidity data."""
    
    print("📊 Loading real IoT data...")
    
    # Load CSV data
    df = pd.read_csv(csv_path)
    print(f"  - Raw data shape: {df.shape}")
    print(f"  - Columns: {list(df.columns)}")
    
    # Clean column names
    df.columns = ['id', 'room_id', 'noted_date', 'temp', 'location']
    
    # Parse datetime
    df['datetime'] = pd.to_datetime(df['noted_date'], format='%d-%m-%Y %H:%M', errors='coerce')
    df = df.dropna(subset=['datetime'])
    
    # Extract features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Encode location (In/Out)
    le_location = LabelEncoder()
    df['location_encoded'] = le_location.fit_transform(df['location'])
    
    # Create room-based features
    room_stats = df.groupby('room_id')['temp'].agg(['mean', 'std', 'count']).reset_index()
    room_stats.columns = ['room_id', 'room_temp_mean', 'room_temp_std', 'room_sensor_count']
    df = df.merge(room_stats, on='room_id', how='left')
    
    print(f"  - Processed data shape: {df.shape}")
    print(f"  - Temperature range: {df['temp'].min():.1f}°C to {df['temp'].max():.1f}°C")
    print(f"  - Unique rooms: {df['room_id'].nunique()}")
    print(f"  - Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    return df


def create_iot_network_from_data(df, time_window_hours=1):
    """Create IoT network graphs from real sensor data."""
    
    print("🏗️  Creating IoT network graphs...")
    
    # Group by time windows
    df['time_window'] = df['datetime'].dt.floor(f'{time_window_hours}H')
    
    graphs = []
    targets = []
    
    for time_window, window_data in df.groupby('time_window'):
        if len(window_data) < 3:  # Skip windows with too few sensors
            continue
        
        # Create nodes (one per sensor reading)
        nodes = []
        for _, row in window_data.iterrows():
            node_features = [
                row['temp'] / 50.0,  # Normalized temperature
                row['location_encoded'],  # Indoor/Outdoor
                row['hour'] / 24.0,  # Time of day
                row['is_weekend'],  # Weekend flag
                row['room_temp_mean'] / 50.0  # Room average temp
            ]
            nodes.append(node_features)
        
        node_tensor = torch.tensor(nodes, dtype=torch.float32)
        
        # Create edges (connect sensors in same room + temporal connections)
        edges = []
        num_nodes = len(window_data)
        
        # Room-based connections
        room_groups = window_data.groupby('room_id').groups
        for room_id, indices in room_groups.items():
            indices = list(indices)
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    idx_i = list(window_data.index).index(indices[i])
                    idx_j = list(window_data.index).index(indices[j])
                    edges.extend([[idx_i, idx_j], [idx_j, idx_i]])
        
        # Temporal connections (connect sensors with similar temperatures)
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                temp_diff = abs(window_data.iloc[i]['temp'] - window_data.iloc[j]['temp'])
                if temp_diff < 5:  # Connect sensors with similar temperatures
                    edges.extend([[i, j], [j, i]])
        
        if len(edges) == 0:  # Ensure connectivity
            for i in range(num_nodes - 1):
                edges.extend([[i, i+1], [i+1, i]])
        
        edge_tensor = torch.tensor(edges, dtype=torch.long).t()
        
        # Target: Energy efficiency (lower temp variance = higher efficiency)
        temp_variance = window_data['temp'].var()
        efficiency = max(0, 100 - temp_variance)  # Higher is better
        target = torch.tensor([efficiency], dtype=torch.float32)
        
        graphs.append({
            'nodes': node_tensor,
            'edges': edge_tensor,
            'target': target,
            'metadata': {
                'time_window': time_window,
                'num_sensors': num_nodes,
                'avg_temp': window_data['temp'].mean(),
                'temp_variance': temp_variance
            }
        })
    
    print(f"  - Created {len(graphs)} network graphs")
    print(f"  - Average nodes per graph: {np.mean([g['nodes'].shape[0] for g in graphs]):.1f}")
    print(f"  - Average edges per graph: {np.mean([g['edges'].shape[1] for g in graphs]):.1f}")
    
    return graphs


def train_gat_on_real_data():
    """Train GAT model using real IoT sensor data."""
    
    print("\n" + "="*80)
    print("🚀 TRAINING GAT WITH REAL IOT DATA")
    print("="*80)
    
    # Load and process data
    csv_path = "IOT-temp.csv"
    df = load_and_process_iot_data(csv_path)
    
    # Create network graphs
    graphs = create_iot_network_from_data(df, time_window_hours=1)
    
    if len(graphs) == 0:
        print("❌ No valid graphs created from data")
        return
    
    # Split data
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    print(f"\n📊 Data split: {len(train_graphs)} train, {len(test_graphs)} test graphs")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = IoTGATModel(
        input_dim=5,  # temp, location, hour, weekend, room_avg
        hidden_dim=64,
        output_dim=1,
        num_layers=3,
        heads=4
    ).to(device)
    
    print(f"🤖 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"🔧 Device: {device}")
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    num_epochs = 100
    best_loss = float('inf')
    best_reward = -float('inf')
    results = []
    
    print(f"\n🎯 Training for {num_epochs} epochs...")
    print("-" * 80)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_predictions = []
        train_targets = []
        
        # Training
        for graph in train_graphs:
            nodes = graph['nodes'].to(device)
            edges = graph['edges'].to(device)
            target = graph['target'].to(device)
            
            optimizer.zero_grad()
            
            prediction, embedding = model(nodes, edges)
            loss = criterion(prediction, target)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_predictions.append(prediction.cpu().detach().numpy())
            train_targets.append(target.cpu().numpy())
        
        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for graph in test_graphs:
                nodes = graph['nodes'].to(device)
                edges = graph['edges'].to(device)
                target = graph['target'].to(device)
                
                prediction, embedding = model(nodes, edges)
                loss = criterion(prediction, target)
                
                val_loss += loss.item()
                val_predictions.append(prediction.cpu().numpy())
                val_targets.append(target.cpu().numpy())
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_graphs)
        avg_val_loss = val_loss / len(test_graphs)
        avg_val_reward = np.mean(val_predictions)
        
        scheduler.step(avg_val_loss)
        
        # Track best model
        if avg_val_reward > best_reward:
            best_reward = avg_val_reward
            best_loss = avg_val_loss
            
            # Save best model
            model_dir = Path("models/phase3_gat")
            model_dir.mkdir(exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_reward': best_reward,
                'best_loss': best_loss
            }, model_dir / "real_data_gat_best.pt")
        
        # Log progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss = {avg_train_loss:.4f}, "
                  f"Val Loss = {avg_val_loss:.4f}, Val Reward = {avg_val_reward:.2f}")
        
        results.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_reward': avg_val_reward
        })
    
    print("\n" + "="*80)
    print("📈 REAL DATA GAT TRAINING COMPLETE!")
    print("="*80)
    
    print(f"🏆 Best Validation Reward: {best_reward:.2f}")
    print(f"📉 Best Validation Loss: {best_loss:.4f}")
    
    # Compare with baseline
    baseline_reward = 246.02
    improvement = ((best_reward - baseline_reward) / baseline_reward) * 100
    print(f"📊 vs GCN Baseline (246.02): {improvement:+.1f}%")
    
    # Save results
    summary = {
        "experiment": "GAT_Real_IoT_Data",
        "date": datetime.now().isoformat(),
        "data_source": "IOT-temp.csv",
        "num_graphs": len(graphs),
        "train_graphs": len(train_graphs),
        "test_graphs": len(test_graphs),
        "best_reward": float(best_reward),
        "best_loss": float(best_loss),
        "improvement_percent": float(improvement),
        "epochs": num_epochs,
        "model_params": sum(p.numel() for p in model.parameters()),
        "data_stats": {
            "total_records": len(df),
            "unique_rooms": int(df['room_id'].nunique()),
            "temp_range": [float(df['temp'].min()), float(df['temp'].max())],
            "date_range": [str(df['datetime'].min()), str(df['datetime'].max())]
        },
        "results": results
    }
    
    results_path = Path("reports/gat_real_data_results.json")
    results_path.parent.mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📁 Results saved: {results_path}")
    print(f"💾 Model saved: models/phase3_gat/real_data_gat_best.pt")
    
    return summary


if __name__ == "__main__":
    try:
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Train model
        results = train_gat_on_real_data()
        
        print("\n✅ Real data GAT training completed successfully!")
        print("\n💡 Next steps:")
        print("1. Compare performance with GCN baseline on same data")
        print("2. Integrate with production API if improvement > 5%")
        print("3. Test on larger time windows for better patterns")
        print("4. Add humidity data if available")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
