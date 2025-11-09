"""
Ablation Study

Systematically test impact of different components:
- GNN encoder type (GCN vs GAT vs GraphSAGE)
- Fusion strategy (weighted_sum vs attention)
- DQN-only vs PPO-only vs Hybrid
- Number of GNN layers
"""

import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).parent.parent))

from agent.hybrid_trainer import HybridTrainer
from agent.dqn_trainer import DQNTrainer
from agent.ppo_trainer import PPOTrainer


class AblationStudy:
    """Run ablation experiments to understand component importance."""
    
    def __init__(
        self,
        env_config: dict,
        base_hybrid_config: dict,
        n_timesteps: int = 10000,
        n_seeds: int = 3
    ):
        self.env_config = env_config
        self.base_config = base_hybrid_config
        self.n_timesteps = n_timesteps
        self.n_seeds = n_seeds
        
        self.results = []
        
        print("🔬 Ablation Study initialized")
        print(f"   Training steps: {n_timesteps}")
        print(f"   Random seeds: {n_seeds}")
    
    def test_gnn_architectures(self):
        """Test different GNN encoder types."""
        print("\n" + "="*80)
        print("🧪 Experiment 1: GNN Architecture Comparison")
        print("="*80 + "\n")
        
        architectures = ['GCN', 'GAT', 'GraphSAGE']
        
        for arch in architectures:
            print(f"\n📊 Testing {arch}...")
            
            for seed in range(self.n_seeds):
                config = self.base_config.copy()
                config['architecture']['gnn_conv_type'] = arch
                
                try:
                    trainer = HybridTrainer(
                        env_config=self.env_config,
                        hybrid_config=config,
                        log_dir=f"logs/ablation/gnn_{arch.lower()}_seed{seed}",
                        model_dir=f"models/ablation/gnn_{arch.lower()}_seed{seed}",
                        seed=42 + seed
                    )
                    
                    trainer.train(total_timesteps=self.n_timesteps, eval_freq=5000)
                    metrics = trainer.evaluate(n_episodes=20)
                    trainer.close()
                    
                    self.results.append({
                        'experiment': 'GNN Architecture',
                        'variant': arch,
                        'seed': seed,
                        'mean_reward': metrics['mean_reward'],
                        'std_reward': metrics['std_reward']
                    })
                    
                    print(f"  Seed {seed}: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
                    
                except Exception as e:
                    print(f"  ❌ Seed {seed} failed: {e}")
        
        print(f"\n✅ GNN architecture comparison complete!")
    
    def test_fusion_strategies(self):
        """Test different fusion methods."""
        print("\n" + "="*80)
        print("🧪 Experiment 2: Fusion Strategy Comparison")
        print("="*80 + "\n")
        
        strategies = ['weighted_sum', 'attention']
        
        for strategy in strategies:
            print(f"\n📊 Testing {strategy}...")
            
            for seed in range(self.n_seeds):
                config = self.base_config.copy()
                config['fusion']['strategy'] = strategy
                
                try:
                    trainer = HybridTrainer(
                        env_config=self.env_config,
                        hybrid_config=config,
                        log_dir=f"logs/ablation/fusion_{strategy}_seed{seed}",
                        model_dir=f"models/ablation/fusion_{strategy}_seed{seed}",
                        seed=42 + seed
                    )
                    
                    trainer.train(total_timesteps=self.n_timesteps, eval_freq=5000)
                    metrics = trainer.evaluate(n_episodes=20)
                    trainer.close()
                    
                    self.results.append({
                        'experiment': 'Fusion Strategy',
                        'variant': strategy,
                        'seed': seed,
                        'mean_reward': metrics['mean_reward'],
                        'std_reward': metrics['std_reward']
                    })
                    
                    print(f"  Seed {seed}: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
                    
                except Exception as e:
                    print(f"  ❌ Seed {seed} failed: {e}")
        
        print(f"\n✅ Fusion strategy comparison complete!")
    
    def test_model_components(self):
        """Test DQN-only, PPO-only, and Hybrid."""
        print("\n" + "="*80)
        print("🧪 Experiment 3: Model Component Ablation")
        print("="*80 + "\n")
        
        # DQN only
        print("\n📊 Testing DQN-only...")
        for seed in range(self.n_seeds):
            try:
                from agent.dqn_trainer import DQNTrainer
                
                dqn_config = {
                    'learning_rate': 1e-4,
                    'buffer_size': 50000,
                    'batch_size': 64,
                    'gamma': 0.99,
                    'exploration_fraction': 0.1,
                    'exploration_final_eps': 0.05,
                }
                
                trainer = DQNTrainer(
                    env_config=self.env_config,
                    dqn_config=dqn_config,
                    log_dir=f"logs/ablation/dqn_only_seed{seed}",
                    model_dir=f"models/ablation/dqn_only_seed{seed}",
                    seed=42 + seed
                )
                
                trainer.train(total_timesteps=self.n_timesteps, eval_freq=5000)
                metrics = trainer.evaluate(n_episodes=20)
                
                self.results.append({
                    'experiment': 'Model Components',
                    'variant': 'DQN-only',
                    'seed': seed,
                    'mean_reward': metrics['mean_reward'],
                    'std_reward': metrics['std_reward']
                })
                
                print(f"  Seed {seed}: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
                
            except Exception as e:
                print(f"  ❌ Seed {seed} failed: {e}")
        
        # PPO only
        print("\n📊 Testing PPO-only...")
        for seed in range(self.n_seeds):
            try:
                from agent.ppo_trainer import PPOTrainer
                
                ppo_config = {
                    'learning_rate': 3e-4,
                    'n_steps': 256,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'gamma': 0.99,
                    'clip_range': 0.2,
                }
                
                trainer = PPOTrainer(
                    env_config=self.env_config,
                    ppo_config=ppo_config,
                    log_dir=f"logs/ablation/ppo_only_seed{seed}",
                    model_dir=f"models/ablation/ppo_only_seed{seed}",
                    seed=42 + seed
                )
                
                trainer.train(total_timesteps=self.n_timesteps, eval_freq=5000)
                metrics = trainer.evaluate(n_episodes=20)
                
                self.results.append({
                    'experiment': 'Model Components',
                    'variant': 'PPO-only',
                    'seed': seed,
                    'mean_reward': metrics['mean_reward'],
                    'std_reward': metrics['std_reward']
                })
                
                print(f"  Seed {seed}: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
                
            except Exception as e:
                print(f"  ❌ Seed {seed} failed: {e}")
        
        print(f"\n✅ Model component comparison complete!")
    
    def run_all_experiments(self):
        """Run all ablation experiments."""
        print("\n" + "="*80)
        print("🚀 STARTING COMPREHENSIVE ABLATION STUDY")
        print("="*80)
        
        self.test_gnn_architectures()
        self.test_fusion_strategies()
        self.test_model_components()
        
        # Save results
        self.save_results()
        self.visualize_results()
        
        print("\n" + "="*80)
        print("✅ ABLATION STUDY COMPLETE!")
        print("="*80 + "\n")
    
    def save_results(self):
        """Save results to CSV."""
        results_dir = Path("reports/ablation")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(self.results)
        csv_path = results_dir / 'ablation_results.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"\n📁 Results saved to {csv_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        summary = df.groupby(['experiment', 'variant']).agg({
            'mean_reward': ['mean', 'std'],
            'std_reward': 'mean'
        }).round(2)
        print(summary)
        print("="*60 + "\n")
    
    def visualize_results(self):
        """Create visualization of ablation results."""
        if not self.results:
            print("⚠️  No results to visualize")
            return
        
        df = pd.DataFrame(self.results)
        
        # Group by experiment
        experiments = df['experiment'].unique()
        
        fig, axes = plt.subplots(1, len(experiments), figsize=(6*len(experiments), 6))
        if len(experiments) == 1:
            axes = [axes]
        
        for i, exp in enumerate(experiments):
            exp_data = df[df['experiment'] == exp]
            
            # Calculate mean and std across seeds
            summary = exp_data.groupby('variant').agg({
                'mean_reward': ['mean', 'std']
            })
            
            variants = summary.index
            means = summary[('mean_reward', 'mean')]
            stds = summary[('mean_reward', 'std')]
            
            # Plot
            axes[i].bar(variants, means, yerr=stds, capsize=5, alpha=0.7,
                       color='steelblue', edgecolor='black')
            axes[i].set_title(exp, fontsize=14, fontweight='bold')
            axes[i].set_ylabel('Mean Reward', fontsize=12)
            axes[i].grid(axis='y', alpha=0.3)
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        save_path = Path("reports/ablation/ablation_comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to {save_path}")
        plt.show()


if __name__ == "__main__":
    # Load configs
    with open('configs/env_config.yaml', 'r') as f:
        env_cfg = yaml.safe_load(f)['environment']
    
    with open('configs/hybrid_config.yaml', 'r') as f:
        hybrid_cfg = yaml.safe_load(f)['hybrid']
    
    # Run ablation study
    study = AblationStudy(
        env_config=env_cfg,
        base_hybrid_config=hybrid_cfg,
        n_timesteps=10000,
        n_seeds=3
    )
    
    study.run_all_experiments()
