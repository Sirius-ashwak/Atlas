"""
Hybrid DQN-PPO-GNN Trainer

Combines three components:
1. GNN Encoder: Graph representation learning for network topology
2. DQN Agent: Q-value estimation for discrete action selection
3. PPO Agent: Policy gradient refinement for exploration

Fusion strategies:
- Weighted ensemble: Combine Q-values and policy logits
- Hierarchical: DQN for coarse selection, PPO for refinement
- Attention-based: Learn to weight components dynamically
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from src.env.iot_env import make_env
from src.gnn.encoder import GNNEncoder
from src.utils.graph_utils import IoTGraphBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridPolicy(nn.Module):
    """
    Custom policy network that fuses GNN, DQN, and PPO components.
    
    Architecture:
        GNN Encoder -> Graph embedding
        DQN Head -> Q-values
        PPO Head -> Policy logits + Value
        Fusion Layer -> Combined action distribution
    """
    
    def __init__(
        self,
        gnn_config: Dict,
        num_actions: int,
        fusion_strategy: str = "weighted_sum",
        dqn_weight: float = 0.6,
        ppo_weight: float = 0.4
    ):
        super().__init__()
        
        self.num_actions = num_actions
        self.fusion_strategy = fusion_strategy
        self.dqn_weight = dqn_weight
        self.ppo_weight = ppo_weight
        
        # GNN encoder
        self.gnn = GNNEncoder(
            node_feature_dim=gnn_config['node_feature_dim'],
            hidden_dim=gnn_config['hidden_dim'],
            output_dim=gnn_config['output_dim'],
            num_layers=gnn_config['num_layers'],
            conv_type=gnn_config['conv_type']
        )
        
        embedding_dim = gnn_config['output_dim']
        
        # DQN head (Q-values)
        self.dqn_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        # PPO heads (policy and value)
        self.policy_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Attention-based fusion (if using attention strategy)
        if fusion_strategy == "attention":
            self.attention_weights = nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Softmax(dim=-1)
            )
        
        logger.info(f"Hybrid policy initialized: {fusion_strategy} fusion, "
                   f"DQN weight={dqn_weight}, PPO weight={ppo_weight}")
    
    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through hybrid policy.
        
        Returns:
            action_logits: Combined action distribution (B x A)
            value: State value estimate (B x 1)
            q_values: Q-value estimates (B x A)
        """
        # Encode graph
        graph_embedding = self.gnn(node_features, edge_index, edge_attr, batch)
        
        # DQN branch: Q-values
        q_values = self.dqn_head(graph_embedding)
        
        # PPO branch: Policy logits
        policy_logits = self.policy_head(graph_embedding)
        
        # Value estimate
        value = self.value_head(graph_embedding)
        
        # Fuse Q-values and policy logits
        if self.fusion_strategy == "weighted_sum":
            # Simple weighted average
            action_logits = self.dqn_weight * q_values + self.ppo_weight * policy_logits
        
        elif self.fusion_strategy == "attention":
            # Learn to weight dynamically
            weights = self.attention_weights(graph_embedding)  # (B, 2)
            dqn_w = weights[:, 0:1]  # (B, 1)
            ppo_w = weights[:, 1:2]
            action_logits = dqn_w * q_values + ppo_w * policy_logits
        
        elif self.fusion_strategy == "gating":
            # Element-wise gating
            gate = torch.sigmoid(graph_embedding @ self.gnn.mlp[0].weight.T)
            action_logits = gate * q_values + (1 - gate) * policy_logits
        
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        return action_logits, value, q_values


class HybridTrainer:
    """
    Trainer for the hybrid DQN-PPO-GNN architecture.
    
    Training procedure:
    1. Collect experience using current hybrid policy
    2. Update DQN component with replay buffer
    3. Update PPO component with on-policy data
    4. Update GNN encoder with combined loss
    """
    
    def __init__(
        self,
        env_config: Dict,
        hybrid_config: Dict,
        log_dir: str = "logs/hybrid",
        model_dir: str = "models/hybrid",
        seed: int = 42
    ):
        """
        Args:
            env_config: Environment configuration
            hybrid_config: Hybrid training configuration
            log_dir: TensorBoard log directory
            model_dir: Model checkpoint directory
            seed: Random seed
        """
        self.env_config = env_config
        self.hybrid_config = hybrid_config
        self.log_dir = Path(log_dir)
        self.model_dir = Path(model_dir)
        self.seed = seed
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Build graph topology (static)
        self.graph_builder = IoTGraphBuilder()
        self.graph = self.graph_builder.build_hierarchical_topology(
            num_sensors=env_config['observation']['num_nodes'] // 2,
            num_fog=env_config['observation']['num_nodes'] // 3,
            num_cloud=1
        )
        
        # Environment
        self.env = make_env(env_config, 
                           graph_builder=self.graph_builder,
                           use_graph_obs=True)
        self.env = Monitor(self.env)
        
        # Initialize hybrid policy
        gnn_config = {
            'node_feature_dim': env_config['observation']['node_features'],
            'hidden_dim': hybrid_config['architecture']['gnn_hidden_dim'],
            'output_dim': 128,
            'num_layers': hybrid_config['architecture']['gnn_num_layers'],
            'conv_type': hybrid_config['architecture']['gnn_conv_type']
        }
        
        self.policy = HybridPolicy(
            gnn_config=gnn_config,
            num_actions=self.env.action_space.n,
            fusion_strategy=hybrid_config['fusion']['strategy'],
            dqn_weight=hybrid_config['fusion']['dqn_weight'],
            ppo_weight=hybrid_config['fusion']['ppo_weight']
        ).to(self.device)
        
        # Optimizers
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=hybrid_config['dqn']['learning_rate']
        )
        
        # Experience replay buffer (for DQN component)
        self.replay_buffer = []
        self.buffer_size = hybrid_config['dqn']['buffer_size']
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        self.global_step = 0
        self.best_eval_reward = float('-inf')  # Track best performance
        
        logger.info("Hybrid trainer initialized successfully")

    def preload_from_dataframe(
        self,
        df,
        feature_names,
        max_samples: int = 1000
    ):
        """Warm start the replay buffer using historical data."""

        if df is None or len(df) == 0:
            logger.warning("No data provided for replay buffer warm start")
            return

        timestamps = sorted(df['timestamp'].unique())
        prev_obs = None
        stored = 0

        logger.info(
            "Preloading replay buffer with %d samples from dataframe", max_samples
        )

        expected_nodes = self.graph.number_of_nodes() if self.graph is not None else None
        pad_warning_emitted = False
        truncate_warning_emitted = False

        for ts in timestamps:
            snapshot = df[df['timestamp'] == ts].sort_values('node_id')
            if snapshot.empty:
                continue

            features = snapshot[feature_names].to_numpy(dtype=np.float32)

            if expected_nodes is not None and features.shape[0] != expected_nodes:
                num_features = features.shape[1] if features.ndim == 2 else len(feature_names)

                if features.shape[0] < expected_nodes:
                    pad_rows = expected_nodes - features.shape[0]
                    reference = features[-1:] if features.size else np.zeros((1, num_features), dtype=np.float32)
                    pad = np.repeat(reference, pad_rows, axis=0)
                    original_nodes = features.shape[0]
                    features = np.vstack([features, pad])

                    if not pad_warning_emitted:
                        logger.warning(
                            "Replay warm start: padded feature matrix from %d to %d nodes",
                            original_nodes,
                            expected_nodes
                        )
                        pad_warning_emitted = True
                else:
                    original_nodes = features.shape[0]
                    features = features[:expected_nodes]

                    if not truncate_warning_emitted:
                        logger.warning(
                            "Replay warm start: truncated feature matrix from %d to %d nodes",
                            original_nodes,
                            expected_nodes
                        )
                        truncate_warning_emitted = True

            obs = {
                'node_features': features,
                'timestamp': float(ts)
            }

            if prev_obs is not None:
                latency_cost = snapshot['latency'].mean()
                energy_cost = snapshot['energy'].mean()
                reward = float(-(latency_cost + 0.05 * energy_cost))

                self._store_transition(prev_obs, 0, reward, obs, False)
                stored += 1

                if stored >= max_samples:
                    break

            prev_obs = obs

        logger.info("Replay buffer warm start complete: %d samples added", stored)

    def train(
        self,
        total_timesteps: int = 100000,
        batch_size: int = 64,
        eval_freq: int = 5000,
        n_eval_episodes: int = 10
    ):
        """
        Main training loop.
        
        Args:
            total_timesteps: Total training steps
            batch_size: Mini-batch size
            eval_freq: Evaluate every N steps
            n_eval_episodes: Number of evaluation episodes
        """
        logger.info(f"Starting hybrid training for {total_timesteps} timesteps...")
        
        obs, info = self.env.reset(seed=self.seed)
        episode_reward = 0
        episode_length = 0
        
        pbar = tqdm(total=total_timesteps, desc="Training")
        
        while self.global_step < total_timesteps:
            # Select action using hybrid policy
            action, action_logits, value = self._select_action(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition in replay buffer
            self._store_transition(obs, action, reward, next_obs, done)
            
            episode_reward += reward
            episode_length += 1
            
            # Update policy
            if len(self.replay_buffer) >= batch_size:
                loss_dict = self._update_policy(batch_size)
                self._log_training(loss_dict)
            
            # Reset if episode ended
            if done:
                self.writer.add_scalar("train/episode_reward", episode_reward, self.global_step)
                self.writer.add_scalar("train/episode_length", episode_length, self.global_step)
                
                episode_reward = 0
                episode_length = 0
                
                # Reset before evaluation to avoid step on terminated env
                obs, info = self.env.reset()
            else:
                obs = next_obs
            
            # Evaluation
            if self.global_step % eval_freq == 0 and self.global_step > 0:
                eval_metrics = self.evaluate(n_eval_episodes)
                self._log_evaluation(eval_metrics)
                
                # Check if this is the best model so far
                is_best = eval_metrics['mean_reward'] > self.best_eval_reward
                if is_best:
                    self.best_eval_reward = eval_metrics['mean_reward']
                
                # Save checkpoint (with best model tracking)
                self.save_checkpoint(is_best=is_best)
                
                # Reset environment after evaluation to continue training
                obs, info = self.env.reset()
                episode_reward = 0
                episode_length = 0
            
            self.global_step += 1
            pbar.update(1)
        
        pbar.close()
        
        # Save final model
        self.save_checkpoint(name="final_model")
        logger.info("Training complete!")
    
    def _select_action(
        self,
        obs: Dict,
        deterministic: bool = False
    ) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action using hybrid policy.
        
        Returns:
            action: Selected action index
            action_logits: Action distribution logits
            value: State value estimate
        """
        # Convert observation to PyTorch
        # node_features should be [num_nodes, num_features], not [1, num_nodes, num_features]
        node_features = torch.FloatTensor(obs['node_features']).to(self.device)
        edge_index = torch.LongTensor(
            self.graph_builder.to_pytorch_geometric(obs['node_features']).edge_index
        ).to(self.device)
        
        with torch.no_grad():
            action_logits, value, q_values = self.policy(node_features, edge_index)
        
        # Sample action
        if deterministic:
            action = action_logits.argmax(dim=-1).item()
        else:
            # Epsilon-greedy with decaying epsilon
            epsilon = max(0.05, 1.0 - self.global_step / 50000)
            if np.random.rand() < epsilon:
                action = np.random.randint(self.env.action_space.n)
            else:
                probs = F.softmax(action_logits, dim=-1)
                # Ensure probs is 2D for multinomial: [batch_size, num_actions]
                if probs.dim() > 2:
                    probs = probs.squeeze()
                if probs.dim() == 1:
                    probs = probs.unsqueeze(0)
                action = torch.multinomial(probs, 1).squeeze().item()
        
        return action, action_logits, value
    
    def _store_transition(
        self,
        obs: Dict,
        action: int,
        reward: float,
        next_obs: Dict,
        done: bool
    ):
        """Store transition in replay buffer."""
        self.replay_buffer.append((obs, action, reward, next_obs, done))
        
        # Keep buffer at max size
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def _update_policy(self, batch_size: int) -> Dict:
        """
        Update hybrid policy using sampled batch.
        
        Returns:
            Dict with loss components
        """
        # Sample batch from replay buffer
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        # Unpack batch
        obs_batch = [t[0] for t in batch]
        actions = torch.LongTensor([t[1] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_obs_batch = [t[3] for t in batch]
        dones = torch.FloatTensor([t[4] for t in batch]).to(self.device)
        
        # Forward pass (simplified - in practice need to batch graph data properly)
        # For demonstration, we'll process first obs in batch
        obs = obs_batch[0]
        # node_features should be [num_nodes, num_features], not [1, num_nodes, num_features]
        node_features = torch.FloatTensor(obs['node_features']).to(self.device)
        edge_index = torch.LongTensor(
            self.graph_builder.to_pytorch_geometric(obs['node_features']).edge_index
        ).to(self.device)
        
        action_logits, value, q_values = self.policy(node_features, edge_index)
        
        # DQN loss (TD error)
        # Ensure action_idx has correct shape [batch_size, 1]
        if actions[0].dim() == 0:
            action_idx = actions[0].reshape(1, 1)
        else:
            action_idx = actions[0].reshape(-1, 1)
        
        q_pred = q_values.gather(1, action_idx)  # [1, 1]
        q_target = (rewards[0].view(1, 1) + 
                   0.99 * (1 - dones[0].view(1, 1)) * 
                   q_values.max(dim=1, keepdim=True)[0].detach())  # [1, 1]
        dqn_loss = F.mse_loss(q_pred, q_target)
        
        # PPO loss (policy gradient + value loss)
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_log_prob = log_probs.gather(1, action_idx)
        
        # Ensure both value and reward have same shape for loss calculation
        value_pred = value.squeeze()  # Should be scalar or [1]
        reward_target = rewards[0]
        
        # Make both scalars if needed
        if value_pred.dim() > 0:
            value_pred = value_pred.squeeze()
        if reward_target.dim() > 0:
            reward_target = reward_target.squeeze()
            
        advantage = (reward_target - value_pred).detach()
        policy_loss = -(action_log_prob * advantage).mean()
        value_loss = F.mse_loss(value_pred, reward_target)
        
        # Combined loss
        total_loss = dqn_loss + policy_loss + 0.5 * value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'dqn_loss': dqn_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item()
        }
    
    def evaluate(self, n_episodes: int = 10) -> Dict:
        """Evaluate current policy."""
        episode_rewards = []
        
        for _ in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            ep_reward = 0
            
            while not done:
                action, _, _ = self._select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                ep_reward += reward
            
            episode_rewards.append(ep_reward)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
    
    def _log_training(self, loss_dict: Dict):
        """Log training metrics."""
        for key, value in loss_dict.items():
            self.writer.add_scalar(f"train/{key}", value, self.global_step)
    
    def _log_evaluation(self, metrics: Dict):
        """Log evaluation metrics."""
        for key, value in metrics.items():
            self.writer.add_scalar(f"eval/{key}", value, self.global_step)
        
        logger.info(f"Step {self.global_step} - Eval reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    
    def save_checkpoint(self, name: str = "checkpoint", is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            name: Checkpoint name (will add step number)
            is_best: If True, also save as best_model.pt
        """
        # Save periodic checkpoint with step number
        checkpoint_path = self.model_dir / f"{name}_step_{self.global_step}.pt"
        checkpoint_data = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step
        }
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Also save as latest checkpoint (for easy resuming)
        latest_path = self.model_dir / "latest_checkpoint.pt"
        torch.save(checkpoint_data, latest_path)
        
        # Save as best model if requested
        if is_best:
            best_path = self.model_dir / "best_model.pt"
            torch.save(checkpoint_data, best_path)
            logger.info(f"✅ New best model saved to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def close(self):
        """Clean up resources."""
        self.env.close()
        self.writer.close()


if __name__ == "__main__":
    import yaml
    
    # Load configs
    with open("configs/env_config.yaml", 'r') as f:
        env_cfg = yaml.safe_load(f)['environment']
    
    with open("configs/hybrid_config.yaml", 'r') as f:
        hybrid_cfg = yaml.safe_load(f)['hybrid']
    
    # Create trainer
    trainer = HybridTrainer(
        env_config=env_cfg,
        hybrid_config=hybrid_cfg,
        seed=42
    )
    
    # Train
    trainer.train(total_timesteps=50000)
    
    # Final evaluation
    metrics = trainer.evaluate(n_episodes=50)
    print("\n=== Final Evaluation ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.3f}")
    
    trainer.close()
