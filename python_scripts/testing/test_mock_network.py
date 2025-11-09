"""Test mock network generation through API"""
import requests
import json

# Test mock network generation
try:
    # First, check if the endpoint exists
    response = requests.post(
        "http://localhost:8000/generate-mock-network",
        json={
            "num_devices": 10,
            "num_fog": 3,
            "num_cloud": 2
        }
    )
    
    if response.status_code == 200:
        network = response.json()
        print("✅ Mock network generated successfully!")
        print(f"Number of nodes: {len(network.get('nodes', []))}")
        print(f"Number of edges: {len(network.get('edges', []))}")
        
        # Try prediction with the mock network
        print("\nTesting prediction with mock network...")
        pred_response = requests.post(
            "http://localhost:8000/predict",
            json={
                "network_state": network,
                "model_type": "dqn"  # Try with DQN first since it's simpler
            }
        )
        
        if pred_response.status_code == 200:
            result = pred_response.json()
            print("✅ Prediction successful with DQN!")
            print(f"Selected Node: {result['selected_node']}")
            print(f"Confidence: {result.get('confidence', 'N/A')}")
        else:
            print(f"❌ Prediction failed: {pred_response.status_code}")
            print(pred_response.text[:500])
            
    else:
        print(f"❌ Mock network generation failed: {response.status_code}")
        print(response.text[:500])
        
except Exception as e:
    print(f"❌ Error: {e}")
