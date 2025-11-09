"""
Unit tests for IoT Edge Allocation Environment
"""

import pytest
import numpy as np
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.env.iot_env import IoTEdgeAllocationEnv


@pytest.fixture
def env_config():
    """Load environment configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "env_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)['environment']


@pytest.fixture
def env(env_config):
    """Create test environment."""
    return IoTEdgeAllocationEnv(env_config, use_graph_obs=True)


class TestIoTEdgeAllocationEnv:
    """Test suite for IoT environment."""
    
    def test_initialization(self, env):
        """Test environment initializes correctly."""
        assert env is not None
        assert env.num_nodes > 0
    
    def test_reset(self, env):
        """Test reset returns valid observation."""
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert 'node_features' in obs
    
    def test_step(self, env):
        """Test step execution."""
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (int, float))
