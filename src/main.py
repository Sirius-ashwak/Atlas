"""
Main Entry Point for Atlas: Map. Decide. Optimize.

Provides CLI interface for:
- Training baseline models (DQN, PPO)
- Training hybrid DQN-PPO-GNN model
- Evaluating trained models
- Running experiments with different configurations
"""

import argparse
import yaml
from pathlib import Path
import logging
import sys

from src.agent.dqn_trainer import DQNTrainer, run_dqn_baseline
from src.agent.ppo_trainer import PPOTrainer, run_ppo_baseline
from src.agent.hybrid_trainer import HybridTrainer
from src.utils.data_loader import IoTDataLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_dqn(args):
    """Train DQN baseline."""
    logger.info("=" * 60)
    logger.info("Training DQN Baseline")
    logger.info("=" * 60)
    
    env_config = load_config(args.env_config)['environment']
    hybrid_config = load_config(args.hybrid_config)['hybrid']
    dqn_config = hybrid_config['dqn']
    
    metrics = run_dqn_baseline(
        env_config=env_config,
        dqn_config=dqn_config,
        total_timesteps=args.timesteps,
        seed=args.seed
    )
    
    logger.info("\n=== DQN Training Complete ===")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.3f}")
    
    return metrics


def train_ppo(args):
    """Train PPO baseline."""
    logger.info("=" * 60)
    logger.info("Training PPO Baseline")
    logger.info("=" * 60)
    
    env_config = load_config(args.env_config)['environment']
    hybrid_config = load_config(args.hybrid_config)['hybrid']
    ppo_config = hybrid_config['ppo']
    
    metrics = run_ppo_baseline(
        env_config=env_config,
        ppo_config=ppo_config,
        total_timesteps=args.timesteps,
        seed=args.seed
    )
    
    logger.info("\n=== PPO Training Complete ===")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.3f}")
    
    return metrics


def train_hybrid(args):
    """Train hybrid DQN-PPO-GNN model."""
    logger.info("=" * 60)
    logger.info("Training Hybrid DQN-PPO-GNN Model")
    logger.info("=" * 60)
    
    env_config = load_config(args.env_config)['environment']
    hybrid_config = load_config(args.hybrid_config)['hybrid']
    
    trainer = HybridTrainer(
        env_config=env_config,
        hybrid_config=hybrid_config,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        seed=args.seed
    )
    
    trainer.train(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval
    )
    
    # Final evaluation
    metrics = trainer.evaluate(n_episodes=50)
    
    logger.info("\n=== Hybrid Training Complete ===")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.3f}")
    
    trainer.close()
    
    return metrics


def evaluate_model(args):
    """Evaluate a trained model."""
    logger.info("=" * 60)
    logger.info(f"Evaluating {args.model_type.upper()} Model")
    logger.info("=" * 60)
    
    env_config = load_config(args.env_config)['environment']
    hybrid_config = load_config(args.hybrid_config)['hybrid']
    
    if args.model_type == "dqn":
        trainer = DQNTrainer(
            env_config=env_config,
            dqn_config=hybrid_config['dqn'],
            seed=args.seed
        )
        metrics = trainer.evaluate(
            n_episodes=args.n_eval,
            model_path=args.model_path
        )
        trainer.close()
    
    elif args.model_type == "ppo":
        trainer = PPOTrainer(
            env_config=env_config,
            ppo_config=hybrid_config['ppo'],
            seed=args.seed
        )
        metrics = trainer.evaluate(
            n_episodes=args.n_eval,
            model_path=args.model_path
        )
        trainer.close()
    
    elif args.model_type == "hybrid":
        trainer = HybridTrainer(
            env_config=env_config,
            hybrid_config=hybrid_config,
            seed=args.seed
        )
        if args.model_path:
            trainer.load_checkpoint(args.model_path)
        metrics = trainer.evaluate(n_episodes=args.n_eval)
        trainer.close()
    
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    logger.info("\n=== Evaluation Results ===")
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.3f}")
    
    return metrics


def run_experiment(args):
    """Run full experiment comparing all methods."""
    logger.info("=" * 60)
    logger.info("Running Comparative Experiment")
    logger.info("=" * 60)
    
    results = {}
    
    # Train DQN
    if 'dqn' in args.methods:
        logger.info("\n>>> Training DQN...")
        args_dqn = argparse.Namespace(**vars(args))
        results['dqn'] = train_dqn(args_dqn)
    
    # Train PPO
    if 'ppo' in args.methods:
        logger.info("\n>>> Training PPO...")
        args_ppo = argparse.Namespace(**vars(args))
        results['ppo'] = train_ppo(args_ppo)
    
    # Train Hybrid
    if 'hybrid' in args.methods:
        logger.info("\n>>> Training Hybrid...")
        args_hybrid = argparse.Namespace(**vars(args))
        results['hybrid'] = train_hybrid(args_hybrid)
    
    # Summary comparison
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)
    
    for method, metrics in results.items():
        logger.info(f"\n{method.upper()}:")
        logger.info(f"  Mean Reward: {metrics['mean_reward']:.3f} ± {metrics['std_reward']:.3f}")
        logger.info(f"  Min/Max: {metrics['min_reward']:.3f} / {metrics['max_reward']:.3f}")
    
    # Save results
    import json
    results_path = Path(args.log_dir) / "experiment_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")
    
    return results


def prepare_data(args):
    """Prepare and preprocess simulation data."""
    logger.info("=" * 60)
    logger.info("Preparing Data")
    logger.info("=" * 60)
    
    loader = IoTDataLoader(
        data_path=args.data_path,
        scaler_type=args.scaler,
        test_split=args.test_split,
        val_split=args.val_split,
        seed=args.seed
    )
    
    train_df, val_df, test_df = loader.load()
    
    logger.info(f"\nData splits:")
    logger.info(f"  Train: {len(train_df)} records")
    logger.info(f"  Val: {len(val_df)} records")
    logger.info(f"  Test: {len(test_df)} records")
    
    # Save processed data
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
    logger.info(f"\nProcessed data saved to {output_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Atlas: Map. Decide. Optimize. - Hybrid RL for IoT Resource Allocation"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train DQN
    parser_dqn = subparsers.add_parser('train-dqn', help='Train DQN baseline')
    parser_dqn.add_argument('--env-config', default='configs/env_config.yaml')
    parser_dqn.add_argument('--hybrid-config', default='configs/hybrid_config.yaml')
    parser_dqn.add_argument('--timesteps', type=int, default=100000)
    parser_dqn.add_argument('--seed', type=int, default=42)
    
    # Train PPO
    parser_ppo = subparsers.add_parser('train-ppo', help='Train PPO baseline')
    parser_ppo.add_argument('--env-config', default='configs/env_config.yaml')
    parser_ppo.add_argument('--hybrid-config', default='configs/hybrid_config.yaml')
    parser_ppo.add_argument('--timesteps', type=int, default=100000)
    parser_ppo.add_argument('--seed', type=int, default=42)
    
    # Train Hybrid
    parser_hybrid = subparsers.add_parser('train-hybrid', help='Train hybrid model')
    parser_hybrid.add_argument('--env-config', default='configs/env_config.yaml')
    parser_hybrid.add_argument('--hybrid-config', default='configs/hybrid_config.yaml')
    parser_hybrid.add_argument('--timesteps', type=int, default=100000)
    parser_hybrid.add_argument('--eval-freq', type=int, default=5000)
    parser_hybrid.add_argument('--n-eval', type=int, default=10)
    parser_hybrid.add_argument('--log-dir', default='logs/hybrid')
    parser_hybrid.add_argument('--model-dir', default='models/hybrid')
    parser_hybrid.add_argument('--seed', type=int, default=42)
    
    # Evaluate
    parser_eval = subparsers.add_parser('evaluate', help='Evaluate trained model')
    parser_eval.add_argument('--model-type', required=True, choices=['dqn', 'ppo', 'hybrid'])
    parser_eval.add_argument('--model-path', required=True)
    parser_eval.add_argument('--env-config', default='configs/env_config.yaml')
    parser_eval.add_argument('--hybrid-config', default='configs/hybrid_config.yaml')
    parser_eval.add_argument('--n-eval', type=int, default=100)
    parser_eval.add_argument('--seed', type=int, default=42)
    
    # Experiment
    parser_exp = subparsers.add_parser('experiment', help='Run full experiment')
    parser_exp.add_argument('--methods', nargs='+', default=['dqn', 'ppo', 'hybrid'],
                           choices=['dqn', 'ppo', 'hybrid'])
    parser_exp.add_argument('--env-config', default='configs/env_config.yaml')
    parser_exp.add_argument('--hybrid-config', default='configs/hybrid_config.yaml')
    parser_exp.add_argument('--timesteps', type=int, default=100000)
    parser_exp.add_argument('--eval-freq', type=int, default=5000)
    parser_exp.add_argument('--n-eval', type=int, default=10)
    parser_exp.add_argument('--log-dir', default='logs/experiment')
    parser_exp.add_argument('--model-dir', default='models/experiment')
    parser_exp.add_argument('--seed', type=int, default=42)
    
    # Prepare data
    parser_data = subparsers.add_parser('prepare-data', help='Prepare simulation data')
    parser_data.add_argument('--data-path', default='data/raw/sim_results.csv')
    parser_data.add_argument('--scaler', default='standard', choices=['standard', 'minmax'])
    parser_data.add_argument('--test-split', type=float, default=0.2)
    parser_data.add_argument('--val-split', type=float, default=0.1)
    parser_data.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate function
    if args.command == 'train-dqn':
        train_dqn(args)
    elif args.command == 'train-ppo':
        train_ppo(args)
    elif args.command == 'train-hybrid':
        train_hybrid(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'experiment':
        run_experiment(args)
    elif args.command == 'prepare-data':
        prepare_data(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
