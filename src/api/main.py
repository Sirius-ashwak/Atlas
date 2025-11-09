"""
FastAPI Application for IoT Edge Allocator Model Serving
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from contextlib import asynccontextmanager
from typing import Dict

from pydantic import BaseModel
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthCheck,
    ErrorResponse,
    ModelType
)
from .model_loader import get_model_loader


# Additional schema for mock network generation
class MockNetworkRequest(BaseModel):
    num_nodes: int = 10
    num_edges: int = 15

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Track server start time
SERVER_START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app."""
    # Startup
    logger.info("🚀 Starting Atlas API Server...")
    
    # Load models
    model_loader = get_model_loader()
    logger.info("📦 Loading trained models...")
    model_loader.load_all_models()
    
    loaded = model_loader.get_loaded_models()
    logger.info(f"✅ Loaded models: {loaded}")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down Atlas API Server...")


# Create FastAPI app
app = FastAPI(
    title="Atlas: Map. Decide. Optimize. API",
    description="REST API for IoT Edge Computing Resource Allocation using Hybrid DQN-PPO-GNN",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Atlas API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    model_loader = get_model_loader()
    uptime = time.time() - SERVER_START_TIME
    
    return HealthCheck(
        status="healthy",
        version="1.0.0",
        models_loaded=model_loader.get_loaded_models(),
        uptime_seconds=uptime
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict optimal node placement for a task.
    
    - **network_state**: Current state of the IoT network
    - **model_type**: Model to use for prediction (dqn, ppo, hybrid, etc.)
    
    Returns the selected node ID and confidence scores.
    """
    try:
        model_loader = get_model_loader()
        
        # Check if model is loaded
        if not model_loader.is_loaded(request.model_type.value):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model '{request.model_type.value}' is not loaded"
            )
        
        # Convert request to dict
        network_state = {
            'nodes': [node.model_dump() for node in request.network_state.nodes],
            'edges': request.network_state.edges
        }
        
        # Make prediction
        selected_node, confidence, metadata = model_loader.predict(
            model_type=request.model_type.value,
            network_state=network_state,
            deterministic=True
        )
        
        return PredictionResponse(
            selected_node=selected_node,
            confidence=confidence,
            q_values=metadata.get('q_values'),
            node_scores=metadata.get('node_scores', {}),
            processing_time_ms=metadata.get('processing_time_ms', 0.0)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Predict optimal node placement for multiple tasks.
    
    - **network_states**: List of network states
    - **model_type**: Model to use for all predictions
    
    Returns predictions for all network states.
    """
    try:
        model_loader = get_model_loader()
        
        if not model_loader.is_loaded(request.model_type.value):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model '{request.model_type.value}' is not loaded"
            )
        
        start_time = time.time()
        predictions = []
        
        for net_state in request.network_states:
            network_state = {
                'nodes': [node.model_dump() for node in net_state.nodes],
                'edges': net_state.edges
            }
            
            selected_node, confidence, metadata = model_loader.predict(
                model_type=request.model_type.value,
                network_state=network_state,
                deterministic=True
            )
            
            predictions.append(PredictionResponse(
                selected_node=selected_node,
                confidence=confidence,
                q_values=metadata.get('q_values'),
                node_scores=metadata.get('node_scores', {}),
                processing_time_ms=metadata.get('processing_time_ms', 0.0)
            ))
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_processing_time_ms=total_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/models")
async def list_models():
    """List all available models with detailed information."""
    model_loader = get_model_loader()
    loaded_status = model_loader.get_loaded_models()
    
    # Model metadata
    model_details = {
        "dqn": {
            "name": "dqn",
            "type": "DQN",
            "description": "Deep Q-Network - Value-based reinforcement learning",
            "path": "models/dqn/best_model/best_model.zip",
            "status": "available" if loaded_status.get("dqn", False) else "unavailable",
            "performance": {
                "mean_reward": 244.15,
                "std_reward": 9.20
            }
        },
        "ppo": {
            "name": "ppo",
            "type": "PPO",
            "description": "Proximal Policy Optimization - Policy-based RL",
            "path": "models/ppo/best_model/best_model.zip",
            "status": "available" if loaded_status.get("ppo", False) else "unavailable",
            "performance": {
                "mean_reward": 241.87,
                "std_reward": 11.84
            }
        },
        "hybrid": {
            "name": "hybrid",
            "type": "Hybrid",
            "description": "Hybrid DQN-PPO-GNN - Best performing model with graph neural networks",
            "path": "models/hybrid/best_model.pt",
            "status": "available" if loaded_status.get("hybrid", False) else "unavailable",
            "performance": {
                "mean_reward": 273.16,
                "std_reward": 8.12
            }
        },
        "hybrid_gat": {
            "name": "hybrid_gat",
            "type": "Hybrid-GAT",
            "description": "Hybrid model with Graph Attention Networks",
            "path": "models/hybrid_gat/best_model.pt",
            "status": "available" if loaded_status.get("hybrid_gat", False) else "unavailable",
            "performance": {
                "mean_reward": 270.0,
                "std_reward": 9.0
            }
        },
        "hybrid_attention": {
            "name": "hybrid_attention",
            "type": "Hybrid-Attention",
            "description": "Hybrid model with attention-based fusion",
            "path": "models/hybrid_attention/best_model.pt",
            "status": "available" if loaded_status.get("hybrid_attention", False) else "unavailable",
            "performance": {
                "mean_reward": 265.0,
                "std_reward": 10.0
            }
        }
    }
    
    # Filter to only include available models
    available_models = {k: v for k, v in model_details.items() if v["status"] == "available"}
    
    return {
        "models": list(available_models.values()),
        "total": len(available_models),
        "loaded": loaded_status
    }


@app.get("/models/{model_type}", response_model=ModelInfo)
async def get_model_info(model_type: ModelType):
    """Get detailed information about a specific model."""
    model_loader = get_model_loader()
    
    if not model_loader.is_loaded(model_type.value):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_type.value}' is not loaded"
        )
    
    info = model_loader.get_model_info(model_type.value)
    
    return ModelInfo(
        model_type=info['type'],
        model_path=info['path'],
        loaded=True,
        architecture=info.get('architecture', {}),
        training_info=info.get('training_info')
    )


@app.post("/models/{model_type}/load")
async def load_model(model_type: ModelType):
    """Load a specific model."""
    model_loader = get_model_loader()
    
    success = model_loader.load_model(model_type.value)
    
    if success:
        return {"status": "success", "message": f"Model '{model_type.value}' loaded successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model '{model_type.value}'"
        )


@app.post("/generate-mock-network")
async def generate_mock_network(request: MockNetworkRequest):
    """
    Generate a mock IoT network for testing.
    
    - **num_nodes**: Number of nodes in the network (default: 10)
    - **num_edges**: Number of edges/connections (default: 15)
    
    Returns a sample network state that can be used for predictions.
    """
    import random
    
    num_nodes = request.num_nodes
    num_edges = request.num_edges
    
    try:
        # Generate nodes with random features
        nodes = []
        node_types = ['sensor', 'fog', 'cloud']
        
        for i in range(num_nodes):
            node_type_idx = i % 3  # Distribute types evenly
            nodes.append({
                "cpu_util": round(random.uniform(0.1, 0.9), 2),
                "mem_util": round(random.uniform(0.1, 0.8), 2),
                "energy": round(random.uniform(50, 200), 2),
                "latency": round(random.uniform(5, 50), 2),
                "bandwidth": round(random.uniform(50, 1000), 2),
                "queue_len": round(random.uniform(0, 20), 2),
                "node_type": node_type_idx
            })
        
        # Generate random edges (connections between nodes)
        edges = []
        for _ in range(min(num_edges, num_nodes * (num_nodes - 1) // 2)):
            source = random.randint(0, num_nodes - 1)
            target = random.randint(0, num_nodes - 1)
            if source != target and [source, target] not in edges and [target, source] not in edges:
                edges.append([source, target])
        
        network_state = {
            "nodes": nodes,
            "edges": edges
        }
        
        return {
            "network_state": network_state,
            "num_nodes": num_nodes,
            "num_edges": len(edges),
            "generated_at": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to generate mock network: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate mock network: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "InternalServerError", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
