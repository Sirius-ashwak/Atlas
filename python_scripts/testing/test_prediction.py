"""Test ML model prediction through API"""
import requests
import json

# Test data for prediction
test_data = {
    "network_state": {
        "nodes": [
            {
                "cpu_util": 0.5,
                "mem_util": 0.3,
                "energy": 45.0,
                "latency": 10.0,
                "bandwidth": 100.0,
                "queue_len": 2.0,
                "node_type": 0  # sensor/device
            },
            {
                "cpu_util": 0.7,
                "mem_util": 0.6,
                "energy": 80.0,
                "latency": 5.0,
                "bandwidth": 500.0,
                "queue_len": 4.0,
                "node_type": 1  # fog
            },
            {
                "cpu_util": 0.2,
                "mem_util": 0.3,
                "energy": 200.0,
                "latency": 20.0,
                "bandwidth": 1000.0,
                "queue_len": 1.0,
                "node_type": 2  # cloud
            }
        ],
        "edges": [[0, 1], [1, 2], [0, 2]]
    },
    "model_type": "hybrid"
}

# Test the prediction endpoint
try:
    response = requests.post("http://localhost:8000/predict", json=test_data)
    if response.status_code == 200:
        result = response.json()
        print("✅ Prediction successful!")
        print(f"Selected Node: {result['selected_node']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
        if result.get('node_scores'):
            print("\nNode Scores:")
            for node_id, score in result['node_scores'].items():
                print(f"  Node {node_id}: {score:.4f}")
    else:
        print(f"❌ Prediction failed: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"❌ Error: {e}")

# Test list models endpoint
print("\n" + "="*50)
print("Testing model listing...")
try:
    response = requests.get("http://localhost:8000/models")
    if response.status_code == 200:
        models = response.json()
        print("✅ Available models:")
        for model, loaded in models.items():
            status = "✅ Loaded" if loaded else "❌ Not loaded"
            print(f"  {model}: {status}")
    else:
        print(f"❌ Failed to list models: {response.status_code}")
except Exception as e:
    print(f"❌ Error: {e}")
