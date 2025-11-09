"""
Test client for the API server.

Usage:
    python -m src.api.test_client
"""

import requests
import json
from typing import Dict, List


class EdgeAllocatorClient:
    """Client for interacting with the Edge Allocator API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict:
        """List all available models."""
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self, model_type: str) -> Dict:
        """Get information about a specific model."""
        response = self.session.get(f"{self.base_url}/models/{model_type}")
        response.raise_for_status()
        return response.json()
    
    def predict(
        self,
        nodes: List[Dict],
        edges: List[List[int]],
        model_type: str = "hybrid"
    ) -> Dict:
        """
        Get prediction for task placement.
        
        Args:
            nodes: List of node feature dicts
            edges: List of [source, target] edges
            model_type: Model to use for prediction
        
        Returns:
            Prediction response with selected node and scores
        """
        payload = {
            "network_state": {
                "nodes": nodes,
                "edges": edges
            },
            "model_type": model_type
        }
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def batch_predict(
        self,
        network_states: List[Dict],
        model_type: str = "hybrid"
    ) -> Dict:
        """Get predictions for multiple network states."""
        payload = {
            "network_states": network_states,
            "model_type": model_type
        }
        
        response = self.session.post(
            f"{self.base_url}/batch-predict",
            json=payload
        )
        response.raise_for_status()
        return response.json()


def create_sample_network() -> tuple:
    """Create a sample network for testing."""
    nodes = [
        {
            "cpu_util": 0.3,
            "mem_util": 0.4,
            "energy": 45.0,
            "latency": 15.0,
            "bandwidth": 100.0,
            "queue_len": 2.0,
            "node_type": 0  # Sensor
        },
        {
            "cpu_util": 0.5,
            "mem_util": 0.6,
            "energy": 70.0,
            "latency": 8.0,
            "bandwidth": 200.0,
            "queue_len": 4.0,
            "node_type": 1  # Fog
        },
        {
            "cpu_util": 0.7,
            "mem_util": 0.8,
            "energy": 120.0,
            "latency": 3.0,
            "bandwidth": 500.0,
            "queue_len": 10.0,
            "node_type": 2  # Cloud
        }
    ]
    
    edges = [[0, 1], [1, 2]]  # Sensor -> Fog -> Cloud
    
    return nodes, edges


def main():
    """Run test client demonstrations."""
    print("\n" + "="*80)
    print("🧪 TESTING ATLAS API")
    print("="*80 + "\n")
    
    # Create client
    client = EdgeAllocatorClient()
    
    try:
        # 1. Health Check
        print("1️⃣  Health Check")
        print("-" * 60)
        health = client.health_check()
        print(f"✅ Status: {health['status']}")
        print(f"   Version: {health['version']}")
        print(f"   Uptime: {health['uptime_seconds']:.1f}s")
        print(f"   Models Loaded: {health['models_loaded']}")
        print()
        
        # 2. List Models
        print("2️⃣  Available Models")
        print("-" * 60)
        models = client.list_models()
        for model, loaded in models.items():
            status = "✅ Loaded" if loaded else "❌ Not Loaded"
            print(f"   {model}: {status}")
        print()
        
        # 3. Single Prediction
        print("3️⃣  Single Prediction")
        print("-" * 60)
        nodes, edges = create_sample_network()
        
        result = client.predict(nodes, edges, model_type="hybrid")
        print(f"✅ Selected Node: {result['selected_node']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Processing Time: {result['processing_time_ms']:.2f}ms")
        if result.get('node_scores'):
            print(f"   Node Scores: {result['node_scores']}")
        print()
        
        # 4. Batch Prediction
        print("4️⃣  Batch Prediction")
        print("-" * 60)
        network_states = [
            {"nodes": nodes, "edges": edges}
            for _ in range(3)
        ]
        
        batch_result = client.batch_predict(network_states, model_type="hybrid")
        print(f"✅ Predicted {len(batch_result['predictions'])} network states")
        print(f"   Total Time: {batch_result['total_processing_time_ms']:.2f}ms")
        print(f"   Avg Time per State: {batch_result['total_processing_time_ms']/len(batch_result['predictions']):.2f}ms")
        
        for i, pred in enumerate(batch_result['predictions']):
            print(f"   State {i+1}: Node {pred['selected_node']} (confidence: {pred['confidence']:.3f})")
        print()
        
        print("="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to API server")
        print("   Make sure the server is running:")
        print("   python python_scripts/api/run_api.py")
        print()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
