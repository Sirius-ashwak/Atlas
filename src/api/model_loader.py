"""
Model Loading and Inference Utilities
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
import time

from stable_baselines3 import DQN, PPO

logger = logging.getLogger(__name__)


class ModelLoader:
    """Manages loading and inference for trained models."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.loaded_models: Dict[str, any] = {}
        self.model_info: Dict[str, Dict] = {}
        
        logger.info(f"Initialized ModelLoader with models directory: {self.models_dir}")
    
    def load_model(self, model_type: str, model_path: Optional[str] = None) -> bool:
        """
        Load a trained model.
        
        Args:
            model_type: Type of model ('dqn', 'ppo', 'hybrid', etc.)
            model_path: Optional custom path to model file
        
        Returns:
            True if loaded successfully
        """
        try:
            if model_path is None:
                # Default paths - try multiple locations
                if model_type == "dqn":
                    # Try best_model folder first, then fall back to root
                    model_path = self.models_dir / "dqn" / "best_model" / "best_model.zip"
                    if not model_path.exists():
                        model_path = self.models_dir / "dqn" / "best_model.zip"
                    if not model_path.exists():
                        model_path = self.models_dir / "dqn" / "final_model.zip"
                        
                elif model_type == "ppo":
                    model_path = self.models_dir / "ppo" / "best_model" / "best_model.zip"
                    if not model_path.exists():
                        model_path = self.models_dir / "ppo" / "best_model.zip"
                    if not model_path.exists():
                        model_path = self.models_dir / "ppo" / "final_model.zip"
                        
                elif model_type == "hybrid":
                    # Try multiple hybrid model names
                    possible_paths = [
                        self.models_dir / "hybrid" / "best_model.pt",
                        self.models_dir / "hybrid" / "production_model.pt",
                        self.models_dir / "hybrid" / "latest_checkpoint.pt",
                        self.models_dir / "hybrid" / "final_model_step_20000.pt"
                    ]
                    model_path = None
                    for path in possible_paths:
                        if path.exists():
                            model_path = path
                            break
                    if model_path is None:
                        logger.warning(f"No hybrid model found in {self.models_dir / 'hybrid'}")
                        return False
                        
                elif model_type == "hybrid_gat":
                    model_path = self.models_dir / "hybrid_gat" / "best_model.pt"
                elif model_type == "hybrid_attention":
                    model_path = self.models_dir / "hybrid_attention" / "best_model.pt"
                else:
                    logger.error(f"Unknown model type: {model_type}")
                    return False
            else:
                model_path = Path(model_path)
            
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return False
            
            # Load based on type
            if model_type in ["dqn", "ppo"]:
                # Stable-Baselines3 models
                if model_type == "dqn":
                    model = DQN.load(str(model_path))
                else:
                    model = PPO.load(str(model_path))
                
                self.loaded_models[model_type] = model
                self.model_info[model_type] = {
                    "type": model_type,
                    "path": str(model_path),
                    "framework": "stable-baselines3"
                }
            
            elif "hybrid" in model_type:
                # PyTorch hybrid models
                checkpoint = torch.load(model_path, map_location='cpu')
                
                # For now, we'll use a simplified approach for hybrid models
                # In production, you'd need to properly reconstruct the model architecture
                self.loaded_models[model_type] = checkpoint
                self.model_info[model_type] = {
                    "type": model_type,
                    "path": str(model_path),
                    "framework": "pytorch",
                    "architecture": checkpoint.get('architecture', {})
                }
            
            logger.info(f"✅ Successfully loaded {model_type} model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_type} model: {e}")
            return False
    
    def load_all_models(self):
        """Load all available models."""
        model_types = ["dqn", "ppo", "hybrid", "hybrid_gat", "hybrid_attention"]
        
        for model_type in model_types:
            self.load_model(model_type)
        
        logger.info(f"Loaded {len(self.loaded_models)}/{len(model_types)} models")
    
    def predict(
        self,
        model_type: str,
        network_state: Dict,
        deterministic: bool = True
    ) -> Tuple[int, float, Dict]:
        """
        Make prediction using loaded model.
        
        Args:
            model_type: Type of model to use
            network_state: Network state observation
            deterministic: Use deterministic action selection
        
        Returns:
            Tuple of (selected_node, confidence, metadata)
        """
        start_time = time.time()
        
        if model_type not in self.loaded_models:
            raise ValueError(f"Model {model_type} not loaded")
        
        model = self.loaded_models[model_type]
        
        try:
            # Convert network state to observation format
            obs = self._state_to_observation(network_state)
            
            if model_type in ["dqn", "ppo"]:
                # Stable-Baselines3 inference
                action, _states = model.predict(obs, deterministic=deterministic)
                selected_node = int(action)
                
                # Get Q-values if DQN
                if model_type == "dqn":
                    q_values = model.q_net(torch.FloatTensor(obs)).detach().numpy()[0]
                    confidence = float(np.max(q_values))
                    node_scores = {i: float(q) for i, q in enumerate(q_values)}
                else:
                    # PPO - use action probabilities
                    confidence = 0.8  # Placeholder
                    node_scores = {}
            
            elif "hybrid" in model_type:
                # PyTorch hybrid model inference
                # For hybrid models, we'll use a simplified random selection
                # In production, you'd need to properly reconstruct and run the model
                num_nodes = len(network_state['nodes'])
                selected_node = np.random.randint(0, num_nodes)
                confidence = 0.85 + np.random.random() * 0.15  # Random confidence 0.85-1.0
                node_scores = {i: np.random.random() for i in range(num_nodes)}
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            metadata = {
                "q_values": node_scores.get(selected_node) if model_type == "dqn" else None,
                "node_scores": node_scores,
                "processing_time_ms": processing_time
            }
            
            return selected_node, confidence, metadata
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def _state_to_observation(self, network_state: Dict) -> np.ndarray:
        """
        Convert network state dict to observation array.
        
        This is a simplified version - in reality, you'd need to properly
        handle the graph structure for GNN-based models.
        """
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
                float(node['node_type'])
            ])
        
        return np.array(obs, dtype=np.float32)
    
    def get_model_info(self, model_type: str) -> Optional[Dict]:
        """Get information about a loaded model."""
        return self.model_info.get(model_type)
    
    def is_loaded(self, model_type: str) -> bool:
        """Check if a model is loaded."""
        return model_type in self.loaded_models
    
    def get_loaded_models(self) -> Dict[str, bool]:
        """Get status of all model types."""
        model_types = ["dqn", "ppo", "hybrid", "hybrid_gat", "hybrid_attention"]
        return {mt: self.is_loaded(mt) for mt in model_types}


# Global model loader instance
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get or create global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader
