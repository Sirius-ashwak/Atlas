"""
DQN Baseline Trainer

Implements Deep Q-Network for discrete action selection in IoT resource allocation.

Features:
- Experience replay buffer
- Target network with soft updates
- Epsilon-greedy exploration
- Support for flattened or graph observations
"""

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import logging
from pathlib import Path
from typing import Dict, Optional

from src.env.iot_env import make_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DQNTrainer:
    """
    Wrapper for training DQN agents on IoT resource allocation.
    
    DQN is well-suited for:
    - Discrete action spaces (node selection)
    - Off-policy learning (efficient sample reuse)
    - Deterministic policies after training
    """
    
    def __init__(
        self,
        env_config: Dict,
        dqn_config: Dict,
        log_dir: str = "logs/dqn",
        model_dir: str = "models/dqn",
        seed: int = 42
    ):
        """
        Args:
            env_config: Environment configuration
            dqn_config: DQN hyperparameters
            log_dir: Directory for TensorBoard logs
            model_dir: Directory for model checkpoints
            seed: Random seed
        """
        self.env_config = env_config
        self.dqn_config = dqn_config
        self.log_dir = Path(log_dir)
        self.model_dir = Path(model_dir)
        self.seed = seed
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environment
        self.env = self._make_env()
        self.eval_env = self._make_env()
        
        # Initialize model
        self.model = None
        
        logger.info(f"DQN Trainer initialized with seed {seed}")
    
    def _make_env(self):
        """Create and wrap environment."""
        env = make_env(self.env_config, use_graph_obs=False)  # DQN uses flattened obs
        env = Monitor(env)
        return env
    
    def train(
        self,
        total_timesteps: int = 100000,
        eval_freq: int = 5000,
        n_eval_episodes: int = 10,
        save_freq: int = 10000
    ):
        """
        Train DQN agent.
        
        Args:
            total_timesteps: Total training steps
            eval_freq: Evaluate every N steps
            n_eval_episodes: Number of episodes for evaluation
            save_freq: Save checkpoint every N steps
        """
        logger.info(f"Starting DQN training for {total_timesteps} timesteps...")
        
        # Initialize DQN model
        self.model = DQN(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.dqn_config.get('learning_rate', 0.0001),
            buffer_size=self.dqn_config.get('buffer_size', 100000),
            learning_starts=self.dqn_config.get('learning_starts', 1000),
            batch_size=self.dqn_config.get('batch_size', 64),
            tau=self.dqn_config.get('tau', 0.005),
            gamma=self.dqn_config.get('gamma', 0.99),
            train_freq=4,
            gradient_steps=1,
            target_update_interval=self.dqn_config.get('target_update_interval', 500),
            exploration_fraction=self.dqn_config.get('exploration_fraction', 0.3),
            exploration_initial_eps=self.dqn_config.get('exploration_initial_eps', 1.0),
            exploration_final_eps=self.dqn_config.get('exploration_final_eps', 0.05),
            tensorboard_log=str(self.log_dir),
            verbose=1,
            seed=self.seed
        )
        
        # Setup callbacks
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.model_dir / "best_model"),
            log_path=str(self.log_dir),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(self.model_dir / "checkpoints"),
            name_prefix="dqn_model"
        )
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            log_interval=100,
            progress_bar=True
        )
        
        # Save final model
        final_path = self.model_dir / "final_model"
        self.model.save(final_path)
        logger.info(f"Training complete! Model saved to {final_path}")
    
    def evaluate(
        self,
        n_episodes: int = 100,
        model_path: Optional[str] = None
    ) -> Dict:
        """
        Evaluate trained model.
        
        Args:
            n_episodes: Number of episodes to evaluate
            model_path: Path to saved model (if None, use current model)
        
        Returns:
            Dict with evaluation metrics
        """
        if model_path:
            self.model = DQN.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        
        if self.model is None:
            raise ValueError("No model to evaluate. Train first or provide model_path.")
        
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(n_episodes):
            obs, info = self.eval_env.reset()
            done = False
            ep_reward = 0
            ep_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_length += 1
            
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
        
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
        
        logger.info(f"Evaluation over {n_episodes} episodes:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.2f}")
        
        return metrics
    
    def close(self):
        """Clean up environments."""
        self.env.close()
        self.eval_env.close()


def run_dqn_baseline(
    env_config: Dict,
    dqn_config: Dict,
    total_timesteps: int = 100000,
    seed: int = 42
):
    """
    Convenience function to run full DQN training pipeline.
    
    Args:
        env_config: Environment configuration
        dqn_config: DQN hyperparameters
        total_timesteps: Training timesteps
        seed: Random seed
    """
    trainer = DQNTrainer(
        env_config=env_config,
        dqn_config=dqn_config,
        seed=seed
    )
    
    # Train
    trainer.train(total_timesteps=total_timesteps)
    
    # Evaluate
    metrics = trainer.evaluate(n_episodes=50)
    
    # Cleanup
    trainer.close()
    
    return metrics


if __name__ == "__main__":
    import yaml
    
    # Load configs
    with open("configs/env_config.yaml", 'r') as f:
        env_cfg = yaml.safe_load(f)['environment']
    
    with open("configs/hybrid_config.yaml", 'r') as f:
        dqn_cfg = yaml.safe_load(f)['hybrid']['dqn']
    
    # Run training
    metrics = run_dqn_baseline(
        env_config=env_cfg,
        dqn_config=dqn_cfg,
        total_timesteps=50000,
        seed=42
    )
    
    print("\n=== Final Evaluation Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")
