"""
Integration Tests for Complete System
"""
import pytest
import sys
from pathlib import Path
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestProjectStructure:
    """Test overall project structure"""
    
    def test_required_directories_exist(self):
        """Test all required directories exist"""
        base_dir = Path(__file__).parent.parent
        
        required_dirs = [
            "src",
            "configs",
            "models",
            "data",
            "docs",
            "tests"
        ]
        
        for dir_name in required_dirs:
            dir_path = base_dir / dir_name
            assert dir_path.exists(), f"Missing directory: {dir_name}"
    
    def test_required_config_files_exist(self):
        """Test required configuration files exist"""
        config_dir = Path(__file__).parent.parent / "configs"
        
        required_configs = [
            "env_config.yaml",
            "hybrid_config.yaml",
        ]
        
        for config_file in required_configs:
            config_path = config_dir / config_file
            assert config_path.exists(), f"Missing config: {config_file}"
    
    def test_documentation_exists(self):
        """Test key documentation files exist"""
        base_dir = Path(__file__).parent.parent
        
        required_docs = [
            "README.md",
            "QUICKSTART.md",
        ]
        
        for doc_file in required_docs:
            doc_path = base_dir / doc_file
            assert doc_path.exists(), f"Missing documentation: {doc_file}"


class TestDependencies:
    """Test all dependencies are installed"""
    
    def test_pytorch_installed(self):
        """Test PyTorch is installed"""
        try:
            import torch
            print(f"PyTorch {torch.__version__} installed ✓")
            assert True
        except ImportError:
            pytest.fail("PyTorch not installed")
    
    def test_pytorch_geometric_installed(self):
        """Test PyTorch Geometric is installed"""
        try:
            import torch_geometric
            print(f"PyTorch Geometric {torch_geometric.__version__} installed ✓")
            assert True
        except ImportError:
            pytest.fail("PyTorch Geometric not installed")
    
    def test_gymnasium_installed(self):
        """Test Gymnasium is installed"""
        try:
            import gymnasium
            print(f"Gymnasium installed ✓")
            assert True
        except ImportError:
            pytest.fail("Gymnasium not installed")
    
    def test_stable_baselines3_installed(self):
        """Test Stable Baselines3 is installed"""
        try:
            import stable_baselines3
            print(f"Stable Baselines3 installed ✓")
            assert True
        except ImportError:
            pytest.fail("Stable Baselines3 not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
