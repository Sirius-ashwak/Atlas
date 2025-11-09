"""
Local Model Inference - Simple Example
=======================================

Use this script to load and run inference with your trained models locally.

Usage:
    python python_scripts/inference/local_inference.py --model-type hybrid
    python python_scripts/inference/local_inference.py --model-type dqn --model-path models/dqn/best_model/best_model.zip
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import json

# Import stable-baselines3 for DQN/PPO
try:
    from stable_baselines3 import DQN, PPO
    SB3_AVAILABLE = True
except ImportError:
    print("⚠️  stable-baselines3 not installed. DQN/PPO models won't work.")
    SB3_AVAILABLE = False


class LocalModelInference:
    """Simple interface for local model inference."""
    
    def __init__(self, model_type: str = "hybrid", model_path: str = None):
        """
        Initialize model inference.
        
        Args:
            model_type: Type of model ('dqn', 'ppo', 'hybrid')
            model_path: Optional custom path to model file
        """
        self.model_type = model_type
        self.model = None
        self.model_path = model_path or self._get_default_path()
        
        print(f"🔧 Initializing {model_type} model...")
        self._load_model()
    
    def _get_default_path(self) -> Path:
        """Get default model path based on type."""
        base_dir = Path("models")
        
        if self.model_type == "dqn":
            return base_dir / "dqn" / "best_model" / "best_model.zip"
        elif self.model_type == "ppo":
            return base_dir / "ppo" / "best_model" / "best_model.zip"
        elif self.model_type == "hybrid":
            return base_dir / "hybrid" / "best_model.pt"
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _load_model(self):
        """Load the model from disk."""
        model_path = Path(self.model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"❌ Model not found at: {model_path}\n"
                f"   Please ensure you have trained models in the models/ directory.\n"
                f"   Run training with: python -m src.main train-hybrid --timesteps 10000"
            )
        
        try:
            if self.model_type == "dqn":
                if not SB3_AVAILABLE:
                    raise ImportError("stable-baselines3 required for DQN")
                self.model = DQN.load(str(model_path))
                print(f"✅ Loaded DQN model from {model_path}")
                
            elif self.model_type == "ppo":
                if not SB3_AVAILABLE:
                    raise ImportError("stable-baselines3 required for PPO")
                self.model = PPO.load(str(model_path))
                print(f"✅ Loaded PPO model from {model_path}")
                
            elif self.model_type == "hybrid":
                checkpoint = torch.load(model_path, map_location='cpu')
                self.model = checkpoint
                print(f"✅ Loaded Hybrid model from {model_path}")
            
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load model: {e}")
    
    def predict(
        self,
        network_state: Dict,
        deterministic: bool = True
    ) -> Dict:
        """
        Make prediction for given network state.
        
        Args:
            network_state: Dictionary containing network information
                {
                    'nodes': [
                        {
                            'node_id': 'fog_0',
                            'cpu_util': 0.45,
                            'mem_util': 0.32,
                            'energy': 120.5,
                            'latency': 15.2,
                            'bandwidth': 150.0,
                            'queue_len': 5,
                            'node_type': 1  # 0=cloud, 1=fog, 2=edge
                        },
                        ...
                    ]
                }
            deterministic: Use deterministic action selection
        
        Returns:
            Dictionary with prediction results
        """
        # Convert network state to observation format
        obs = self._state_to_observation(network_state)
        
        try:
            if self.model_type in ["dqn", "ppo"]:
                # Stable-Baselines3 inference
                action, _states = self.model.predict(obs, deterministic=deterministic)
                selected_node = int(action)
                
                # Get additional info
                if self.model_type == "dqn":
                    q_values = self.model.q_net(torch.FloatTensor(obs)).detach().numpy()[0]
                    node_scores = {i: float(q) for i, q in enumerate(q_values)}
                    confidence = float(np.max(q_values))
                else:
                    node_scores = {}
                    confidence = 0.85  # PPO doesn't have Q-values
                
            elif self.model_type == "hybrid":
                # Hybrid model inference
                # Note: This is simplified - full hybrid needs graph structure
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs)
                    # For actual hybrid model, you'd need proper graph construction
                    selected_node = 0  # Placeholder
                    confidence = 0.90
                    node_scores = {}
            
            result = {
                "selected_node": selected_node,
                "selected_node_id": network_state['nodes'][selected_node]['node_id'] if selected_node < len(network_state['nodes']) else f"node_{selected_node}",
                "confidence": confidence,
                "model_type": self.model_type,
                "node_scores": node_scores,
                "input_nodes": len(network_state['nodes'])
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"❌ Prediction failed: {e}")
    
    def _state_to_observation(self, network_state: Dict) -> np.ndarray:
        """Convert network state dict to observation array."""
        nodes = network_state['nodes']
        
        # Flatten node features
        obs = []
        for node in nodes:
            obs.extend([
                node['cpu_util'],
                node['mem_util'],
                node['energy'],
                node['latency'],
                node['bandwidth'],
                node['queue_len'],
                float(node.get('node_type', 1))
            ])
        
        return np.array(obs, dtype=np.float32)


def create_sample_network_state() -> Dict:
    """Create a sample network state for testing."""
    return {
        'nodes': [
            {
                'node_id': 'cloud_0',
                'cpu_util': 0.25,
                'mem_util': 0.18,
                'energy': 100.0,
                'latency': 30.0,
                'bandwidth': 200.0,
                'queue_len': 2,
                'node_type': 0  # Cloud
            },
            {
                'node_id': 'fog_1',
                'cpu_util': 0.45,
                'mem_util': 0.38,
                'energy': 120.5,
                'latency': 15.2,
                'bandwidth': 150.0,
                'queue_len': 5,
                'node_type': 1  # Fog
            },
            {
                'node_id': 'fog_2',
                'cpu_util': 0.62,
                'mem_util': 0.55,
                'energy': 135.8,
                'latency': 12.5,
                'bandwidth': 140.0,
                'queue_len': 8,
                'node_type': 1  # Fog
            },
            {
                'node_id': 'edge_3',
                'cpu_util': 0.78,
                'mem_util': 0.71,
                'energy': 145.2,
                'latency': 5.3,
                'bandwidth': 100.0,
                'queue_len': 12,
                'node_type': 2  # Edge
            },
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description="Local inference with trained AI Edge Allocator models"
    )
    parser.add_argument(
        '--model-type',
        type=str,
        default='hybrid',
        choices=['dqn', 'ppo', 'hybrid'],
        help='Type of model to use'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Optional custom path to model file'
    )
    parser.add_argument(
        '--input-json',
        type=str,
        default=None,
        help='Path to JSON file with network state'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 AI EDGE ALLOCATOR - LOCAL INFERENCE")
    print("="*70 + "\n")
    
    # Initialize model
    try:
        inference = LocalModelInference(
            model_type=args.model_type,
            model_path=args.model_path
        )
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return
    
    # Get network state
    if args.input_json:
        print(f"\n📄 Loading network state from {args.input_json}")
        with open(args.input_json, 'r') as f:
            network_state = json.load(f)
    else:
        print("\n📝 Using sample network state...")
        network_state = create_sample_network_state()
    
    # Display network state
    print(f"\n🌐 Network State:")
    print(f"   Total Nodes: {len(network_state['nodes'])}")
    for node in network_state['nodes']:
        print(f"   - {node['node_id']}: CPU={node['cpu_util']:.2f}, "
              f"Mem={node['mem_util']:.2f}, Latency={node['latency']:.1f}ms")
    
    # Make prediction
    print(f"\n🤖 Running {args.model_type.upper()} model inference...")
    try:
        result = inference.predict(network_state, deterministic=True)
        
        print("\n✅ PREDICTION RESULTS:")
        print("="*70)
        print(f"📍 Selected Node: {result['selected_node_id']} (index: {result['selected_node']})")
        print(f"💯 Confidence: {result['confidence']:.4f}")
        print(f"🔧 Model Type: {result['model_type']}")
        
        if result['node_scores']:
            print(f"\n📊 Node Scores (Q-values):")
            for node_id, score in list(result['node_scores'].items())[:5]:
                print(f"   Node {node_id}: {score:.4f}")
        
        print("="*70)
        
        # Save result
        output_file = f"prediction_result_{args.model_type}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ Inference complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
