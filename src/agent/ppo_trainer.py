"""
PPO Baseline Trainer

Implements Proximal Policy Optimization for IoT resource allocation.

Features:
- On-policy learning with advantage estimation
- Clipped objective for stable updates
- Suitable for continuous and discrete action spaces
- GAE (Generalized Advantage Estimation)
"""

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import logging
from pathlib import Path
from typing import Dict, Optional

from src.env.iot_env import make_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    Wrapper for training PPO agents on IoT resource allocation.
    
    PPO is well-suited for:
    - On-policy learning (better sample efficiency than A2C)
    - Stable training with clipped objective
    - Both discrete and continuous actions
    """
    
    def __init__(
        self,
        env_config: Dict,
        ppo_config: Dict,
        log_dir: str = "logs/ppo",
        model_dir: str = "models/ppo",
        n_envs: int = 4,
        seed: int = 42
    ):
        """
        Args:
            env_config: Environment configuration
            ppo_config: PPO hyperparameters
            log_dir: Directory for TensorBoard logs
            model_dir: Directory for model checkpoints
            n_envs: Number of parallel environments
            seed: Random seed
        """
        self.env_config = env_config
        self.ppo_config = ppo_config
        self.log_dir = Path(log_dir)
        self.model_dir = Path(model_dir)
        self.n_envs = n_envs
        self.seed = seed
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize environments
        self.env = self._make_vec_env(n_envs)
        self.eval_env = self._make_env()
        
        # Initialize model
        self.model = None
        
        logger.info(f"PPO Trainer initialized with {n_envs} parallel envs, seed {seed}")
    
    def _make_env(self):
        """Create single environment."""
        env = make_env(self.env_config, use_graph_obs=False)  # PPO uses flattened obs
        env = Monitor(env)
        return env
    
    def _make_vec_env(self, n_envs: int):
        """Create vectorized environments for parallel sampling."""
        if n_envs == 1:
            return DummyVecEnv([lambda: self._make_env()])
        else:
            # Use subprocess vectorization for true parallelism
            return DummyVecEnv([lambda: self._make_env() for _ in range(n_envs)])
    
    def train(
        self,
        total_timesteps: int = 100000,
        eval_freq: int = 5000,
        n_eval_episodes: int = 10,
        save_freq: int = 10000
    ):
        """
        Train PPO agent.
        
        Args:
            total_timesteps: Total training steps
            eval_freq: Evaluate every N steps
            n_eval_episodes: Number of episodes for evaluation
            save_freq: Save checkpoint every N steps
        """
        logger.info(f"Starting PPO training for {total_timesteps} timesteps...")
        
        # Initialize PPO model
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.ppo_config.get('learning_rate', 0.0003),
            n_steps=self.ppo_config.get('n_steps', 2048),
            batch_size=self.ppo_config.get('batch_size', 64),
            n_epochs=self.ppo_config.get('n_epochs', 10),
            gamma=self.ppo_config.get('gamma', 0.99),
            gae_lambda=self.ppo_config.get('gae_lambda', 0.95),
            clip_range=self.ppo_config.get('clip_range', 0.2),
            ent_coef=self.ppo_config.get('ent_coef', 0.01),
            vf_coef=self.ppo_config.get('vf_coef', 0.5),
            max_grad_norm=self.ppo_config.get('max_grad_norm', 0.5),
            tensorboard_log=str(self.log_dir),
            verbose=1,
            seed=self.seed
        )
        
        # Setup callbacks
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=str(self.model_dir / "best_model"),
            log_path=str(self.log_dir),
            eval_freq=eval_freq // self.n_envs,  # Adjust for vectorized env
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq // self.n_envs,
            save_path=str(self.model_dir / "checkpoints"),
            name_prefix="ppo_model"
        )
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            log_interval=10,
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
            self.model = PPO.load(model_path)
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


def run_ppo_baseline(
    env_config: Dict,
    ppo_config: Dict,
    total_timesteps: int = 100000,
    seed: int = 42
):
    """
    Convenience function to run full PPO training pipeline.
    
    Args:
        env_config: Environment configuration
        ppo_config: PPO hyperparameters
        total_timesteps: Training timesteps
        seed: Random seed
    """
    trainer = PPOTrainer(
        env_config=env_config,
        ppo_config=ppo_config,
        n_envs=4,
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
        ppo_cfg = yaml.safe_load(f)['hybrid']['ppo']
    
    # Run training
    metrics = run_ppo_baseline(
        env_config=env_cfg,
        ppo_config=ppo_cfg,
        total_timesteps=50000,
        seed=42
    )
    
    print("\n=== Final Evaluation Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")
