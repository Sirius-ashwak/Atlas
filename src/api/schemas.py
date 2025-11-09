"""
Request/Response schemas for the API.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from enum import Enum


class ModelType(str, Enum):
    """Available model types."""
    DQN = "dqn"
    PPO = "ppo"
    HYBRID = "hybrid"
    HYBRID_GAT = "hybrid_gat"
    HYBRID_ATTENTION = "hybrid_attention"


class NodeFeatures(BaseModel):
    """Features for a single node in the network."""
    cpu_util: float = Field(..., ge=0, le=1, description="CPU utilization (0-1)")
    mem_util: float = Field(..., ge=0, le=1, description="Memory utilization (0-1)")
    energy: float = Field(..., ge=0, description="Energy consumption (Joules)")
    latency: float = Field(..., ge=0, description="Network latency (ms)")
    bandwidth: float = Field(..., ge=0, description="Available bandwidth (Mbps)")
    queue_len: float = Field(..., ge=0, description="Task queue length")
    node_type: int = Field(..., ge=0, le=2, description="Node type: 0=sensor, 1=fog, 2=cloud")


class NetworkState(BaseModel):
    """Complete network state observation."""
    nodes: List[NodeFeatures] = Field(..., description="List of node features")
    edges: List[List[int]] = Field(..., description="Edge connectivity [[source, target], ...]")
    
    class Config:
        json_schema_extra = {
            "example": {
                "nodes": [
                    {
                        "cpu_util": 0.45,
                        "mem_util": 0.60,
                        "energy": 50.2,
                        "latency": 12.5,
                        "bandwidth": 100.0,
                        "queue_len": 3.0,
                        "node_type": 1
                    }
                ],
                "edges": [[0, 1], [1, 2]]
            }
        }


class PredictionRequest(BaseModel):
    """Request for single task allocation prediction."""
    network_state: NetworkState
    model_type: ModelType = Field(default=ModelType.HYBRID, description="Model to use for prediction")
    
    class Config:
        protected_namespaces = ()  # Allow model_type field
        json_schema_extra = {
            "example": {
                "network_state": {
                    "nodes": [
                        {"cpu_util": 0.3, "mem_util": 0.5, "energy": 40.0, 
                         "latency": 10.0, "bandwidth": 150.0, "queue_len": 2.0, "node_type": 0},
                        {"cpu_util": 0.6, "mem_util": 0.7, "energy": 80.0,
                         "latency": 5.0, "bandwidth": 200.0, "queue_len": 5.0, "node_type": 1}
                    ],
                    "edges": [[0, 1]]
                },
                "model_type": "hybrid"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    network_states: List[NetworkState]
    model_type: ModelType = Field(default=ModelType.HYBRID)
    
    class Config:
        protected_namespaces = ()


class PredictionResponse(BaseModel):
    """Response with prediction results."""
    selected_node: int = Field(..., description="ID of the selected node for task placement")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    q_values: Optional[List[float]] = Field(None, description="Q-values for all nodes (if DQN/Hybrid)")
    node_scores: Dict[int, float] = Field(..., description="Scores for each node")
    processing_time_ms: float = Field(..., description="Inference time in milliseconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "selected_node": 5,
                "confidence": 0.87,
                "q_values": [0.45, 0.62, 0.87, 0.55],
                "node_scores": {0: 0.45, 1: 0.62, 2: 0.87, 3: 0.55},
                "processing_time_ms": 12.5
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response with batch predictions."""
    predictions: List[PredictionResponse]
    total_processing_time_ms: float


class ModelInfo(BaseModel):
    """Information about loaded model."""
    model_type: str
    model_path: str
    loaded: bool
    architecture: Dict[str, Any]
    training_info: Optional[Dict[str, Any]] = None
    
    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "model_type": "hybrid",
                "model_path": "models/hybrid/best_model.pt",
                "loaded": True,
                "architecture": {
                    "gnn_type": "GCN",
                    "hidden_dim": 64,
                    "num_layers": 3,
                    "fusion_strategy": "weighted_sum"
                },
                "training_info": {
                    "mean_reward": 246.02,
                    "std_reward": 8.57,
                    "training_steps": 5000
                }
            }
        }


class HealthCheck(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: Dict[str, bool] = Field(..., description="Status of loaded models")
    uptime_seconds: float = Field(..., description="Server uptime")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "models_loaded": {
                    "dqn": True,
                    "ppo": True,
                    "hybrid": True
                },
                "uptime_seconds": 3600.5
            }
        }


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ModelNotFound",
                "detail": "Hybrid model not loaded. Please check model path."
            }
        }
