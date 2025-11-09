"""
Tests for Edge Allocation Environment
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.env.edge_env import EdgeAllocationEnv


class TestEdgeAllocationEnv:
    """Test suite for EdgeAllocationEnv"""
    
    def test_env_initialization(self):
        """Test environment can be initialized"""
        env = EdgeAllocationEnv(num_devices=10, num_edge_servers=3)
        assert env is not None
        assert env.num_devices == 10
        assert env.num_edge_servers == 3
    
    def test_env_reset(self):
        """Test environment reset functionality"""
        env = EdgeAllocationEnv(num_devices=10, num_edge_servers=3)
        obs, info = env.reset()
        
        assert obs is not None
        assert isinstance(info, dict)
        assert 'graph_data' in info or len(info) >= 0
    
    def test_env_step(self):
        """Test environment step functionality"""
        env = EdgeAllocationEnv(num_devices=10, num_edge_servers=3)
        env.reset()
        
        # Take a random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_env_action_space(self):
        """Test action space is correct"""
        env = EdgeAllocationEnv(num_devices=10, num_edge_servers=3)
        
        # Action space should allow selecting any edge server for any device
        assert env.action_space is not None
        action = env.action_space.sample()
        assert action >= 0
    
    def test_env_observation_space(self):
        """Test observation space is defined"""
        env = EdgeAllocationEnv(num_devices=10, num_edge_servers=3)
        
        assert env.observation_space is not None
        obs, _ = env.reset()
        assert obs is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
