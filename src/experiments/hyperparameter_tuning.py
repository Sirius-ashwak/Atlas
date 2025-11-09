"""
Hyperparameter Optimization

Automated tuning using Optuna for finding optimal hyperparameters.
"""

import optuna
import numpy as np
import yaml
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from agent.hybrid_trainer import HybridTrainer
from env.iot_env import make_env
from utils.graph_utils import IoTGraphBuilder


class HyperparameterTuner:
    """Optimize hyperparameters using Optuna."""
    
    def __init__(
        self,
        env_config: dict,
        n_trials: int = 50,
        n_timesteps: int = 10000,
        study_name: str = "hybrid_optimization"
    ):
        self.env_config = env_config
        self.n_trials = n_trials
        self.n_timesteps = n_timesteps
        self.study_name = study_name
        
        print(f"🔍 Hyperparameter Tuner initialized")
        print(f"   Trials: {n_trials}")
        print(f"   Training steps per trial: {n_timesteps}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna.
        
        Suggests hyperparameters and trains model.
        Returns mean reward for optimization.
        """
        # Suggest hyperparameters
        hybrid_config = {
            'architecture': {
                'gnn_hidden_dim': trial.suggest_categorical('gnn_hidden_dim', [32, 64, 128]),
                'gnn_num_layers': trial.suggest_int('gnn_num_layers', 2, 4),
                'gnn_conv_type': trial.suggest_categorical('gnn_conv_type', ['GCN', 'GAT', 'GraphSAGE']),
            },
            'fusion': {
                'strategy': trial.suggest_categorical('fusion_strategy', ['weighted_sum', 'attention']),
                'dqn_weight': trial.suggest_float('dqn_weight', 0.3, 0.7) if trial.params['fusion_strategy'] == 'weighted_sum' else 0.5,
                'ppo_weight': trial.suggest_float('ppo_weight', 0.3, 0.7) if trial.params['fusion_strategy'] == 'weighted_sum' else 0.5,
            },
            'dqn': {
                'learning_rate': trial.suggest_float('dqn_lr', 1e-5, 1e-3, log=True),
                'buffer_size': trial.suggest_categorical('buffer_size', [10000, 50000, 100000]),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                'gamma': trial.suggest_float('gamma', 0.95, 0.995),
                'tau': trial.suggest_float('tau', 0.001, 0.01),
            },
            'ppo': {
                'learning_rate': trial.suggest_float('ppo_lr', 1e-5, 1e-3, log=True),
                'n_steps': trial.suggest_categorical('n_steps', [128, 256, 512]),
                'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
            },
            'training': {
                'total_timesteps': self.n_timesteps,
                'eval_freq': 2000,
            }
        }
        
        try:
            # Create trainer
            trainer = HybridTrainer(
                env_config=self.env_config,
                hybrid_config=hybrid_config,
                log_dir=f"logs/optuna/trial_{trial.number}",
                model_dir=f"models/optuna/trial_{trial.number}",
                seed=42 + trial.number
            )
            
            # Train
            trainer.train(
                total_timesteps=self.n_timesteps,
                batch_size=hybrid_config['dqn']['batch_size'],
                eval_freq=2000,
                n_eval_episodes=10
            )
            
            # Evaluate
            eval_metrics = trainer.evaluate(n_episodes=20)
            mean_reward = eval_metrics['mean_reward']
            
            trainer.close()
            
            return mean_reward
            
        except Exception as e:
            print(f"❌ Trial {trial.number} failed: {e}")
            return -float('inf')
    
    def optimize(self, direction: str = 'maximize'):
        """
        Run optimization study.
        
        Args:
            direction: 'maximize' or 'minimize'
        """
        print("\n" + "="*80)
        print("🚀 STARTING HYPERPARAMETER OPTIMIZATION")
        print("="*80 + "\n")
        
        # Create study
        study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Print results
        print("\n" + "="*80)
        print("✅ OPTIMIZATION COMPLETE!")
        print("="*80 + "\n")
        
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value:.2f}")
        print(f"\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        # Save results
        results_dir = Path("reports/optimization")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save best params
        with open(results_dir / 'best_params.yaml', 'w') as f:
            yaml.dump(study.best_params, f)
        
        print(f"\n📁 Best parameters saved to {results_dir / 'best_params.yaml'}")
        
        # Plot optimization history (if optuna.visualization available)
        try:
            from optuna.visualization import plot_optimization_history, plot_param_importances
            import matplotlib.pyplot as plt
            
            fig1 = plot_optimization_history(study)
            fig1.write_image(results_dir / 'optimization_history.png')
            
            fig2 = plot_param_importances(study)
            fig2.write_image(results_dir / 'param_importances.png')
            
            print(f"📊 Visualizations saved to {results_dir}/")
        except ImportError:
            print("ℹ️  Install plotly and kaleido for visualization: pip install plotly kaleido")
        
        return study


def quick_tune():
    """Quick tuning with fewer trials for testing."""
    print("🏃 Running quick hyperparameter tuning (10 trials)...")
    
    # Load config
    with open('configs/env_config.yaml', 'r') as f:
        env_cfg = yaml.safe_load(f)['environment']
    
    tuner = HyperparameterTuner(
        env_config=env_cfg,
        n_trials=10,
        n_timesteps=5000,
        study_name="quick_tune"
    )
    
    study = tuner.optimize()
    return study


def full_tune():
    """Full tuning with many trials."""
    print("🎯 Running full hyperparameter tuning (50 trials)...")
    
    # Load config
    with open('configs/env_config.yaml', 'r') as f:
        env_cfg = yaml.safe_load(f)['environment']
    
    tuner = HyperparameterTuner(
        env_config=env_cfg,
        n_trials=50,
        n_timesteps=10000,
        study_name="full_tune"
    )
    
    study = tuner.optimize()
    return study


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization")
    parser.add_argument('--mode', type=str, default='quick', choices=['quick', 'full'],
                       help='Tuning mode: quick (10 trials) or full (50 trials)')
    args = parser.parse_args()
    
    if args.mode == 'quick':
        study = quick_tune()
    else:
        study = full_tune()
