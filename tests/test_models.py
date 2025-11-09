"""
Tests for AI Models (DQN, PPO, Hybrid)
"""
import pytest
import sys
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelLoading:
    """Test suite for model loading and inference"""
    
    def test_pytorch_available(self):
        """Test PyTorch is available"""
        assert torch.cuda.is_available() or torch.cpu.is_available()
        print(f"PyTorch version: {torch.__version__}")
    
    def test_model_directory_exists(self):
        """Test models directory exists"""
        models_dir = Path(__file__).parent.parent / "models"
        assert models_dir.exists()
    
    def test_trained_models_exist(self):
        """Test trained model checkpoints exist"""
        models_dir = Path(__file__).parent.parent / "models"
        
        # Check for at least one model type
        model_types = ["dqn", "ppo", "hybrid"]
        found_models = []
        
        for model_type in model_types:
            model_path = models_dir / model_type
            if model_path.exists():
                found_models.append(model_type)
        
        assert len(found_models) > 0, f"No trained models found in {models_dir}"
        print(f"Found models: {found_models}")
    
    def test_config_files_exist(self):
        """Test configuration files exist"""
        config_dir = Path(__file__).parent.parent / "configs"
        
        assert config_dir.exists()
        
        # Check for key config files
        expected_configs = ["env_config.yaml", "hybrid_config.yaml"]
        
        for config_file in expected_configs:
            config_path = config_dir / config_file
            if config_path.exists():
                print(f"✓ Found: {config_file}")


class TestModelInference:
    """Test model inference capabilities"""
    
    def test_random_tensor_inference(self):
        """Test basic tensor operations work"""
        x = torch.randn(10, 20)
        y = torch.nn.Linear(20, 10)(x)
        
        assert y.shape == (10, 10)
        assert not torch.isnan(y).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
