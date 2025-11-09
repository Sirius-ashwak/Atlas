"""
Phase 3: Research & Experimentation Runner

Provides easy access to all Phase 3 experiments:
1. Test advanced GNN encoders
2. Hyperparameter optimization
3. Ablation studies
4. Attention-based fusion
"""

import argparse
import sys


def test_encoders():
    """Test advanced GNN encoders."""
    print("\n" + "="*80)
    print("🧪 TESTING ADVANCED GNN ENCODERS")
    print("="*80 + "\n")
    
    import torch
    from src.gnn.advanced_encoders import (
        GATEncoder, GraphSAGEEncoder, HybridGNNEncoder, AttentionFusion
    )
    
    # Create dummy data
    x = torch.randn(10, 6)  # 10 nodes, 6 features
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    print("1️⃣  GAT Encoder Test")
    print("-" * 60)
    gat = GATEncoder(node_feature_dim=6, hidden_dim=64, output_dim=128, num_layers=3)
    gat_out = gat(x, edge_index)
    print(f"✅ Output shape: {gat_out.shape}\n")
    
    print("2️⃣  GraphSAGE Encoder Test")
    print("-" * 60)
    sage = GraphSAGEEncoder(node_feature_dim=6, hidden_dim=64, output_dim=128, num_layers=3)
    sage_out = sage(x, edge_index)
    print(f"✅ Output shape: {sage_out.shape}\n")
    
    print("3️⃣  Hybrid GNN Encoder Test")
    print("-" * 60)
    hybrid = HybridGNNEncoder(node_feature_dim=6, hidden_dim=64, output_dim=128, num_layers=3)
    hybrid_out = hybrid(x, edge_index)
    print(f"✅ Output shape: {hybrid_out.shape}\n")
    
    print("4️⃣  Attention Fusion Test")
    print("-" * 60)
    fusion = AttentionFusion(input_dim=20)
    dqn_out = torch.randn(4, 10)
    ppo_out = torch.randn(4, 10)
    fused, weights = fusion(dqn_out, ppo_out)
    print(f"✅ Fused output: {fused.shape}")
    print(f"✅ Sample attention weights: DQN={weights[0, 0]:.3f}, PPO={weights[0, 1]:.3f}\n")
    
    print("="*80)
    print("✅ ALL ENCODERS WORKING!")
    print("="*80 + "\n")


def run_hyperparameter_tuning(mode='quick'):
    """Run hyperparameter optimization."""
    print("\n" + "="*80)
    print(f"🔍 HYPERPARAMETER OPTIMIZATION ({'QUICK' if mode == 'quick' else 'FULL'})")
    print("="*80 + "\n")
    
    from src.experiments.hyperparameter_tuning import quick_tune, full_tune
    
    if mode == 'quick':
        study = quick_tune()
    else:
        study = full_tune()
    
    print(f"\n✅ Optimization complete! Best reward: {study.best_value:.2f}")
    print(f"📁 Results saved to reports/optimization/\n")


def run_ablation_study():
    """Run ablation experiments."""
    print("\n" + "="*80)
    print("🔬 ABLATION STUDY")
    print("="*80 + "\n")
    
    import yaml
    from src.experiments.ablation_study import AblationStudy
    
    # Load configs
    with open('configs/env_config.yaml', 'r') as f:
        env_cfg = yaml.safe_load(f)['environment']
    
    with open('configs/hybrid_config.yaml', 'r') as f:
        hybrid_cfg = yaml.safe_load(f)['hybrid']
    
    # Run study
    study = AblationStudy(
        env_config=env_cfg,
        base_hybrid_config=hybrid_cfg,
        n_timesteps=10000,
        n_seeds=3
    )
    
    study.run_all_experiments()
    
    print(f"\n✅ Ablation study complete!")
    print(f"📁 Results saved to reports/ablation/\n")


def train_with_gat():
    """Train hybrid model with GAT encoder."""
    print("\n" + "="*80)
    print("🚀 TRAINING HYBRID MODEL WITH GAT")
    print("="*80 + "\n")
    
    import yaml
    from src.agent.hybrid_trainer import HybridTrainer
    
    # Load configs
    with open('configs/env_config.yaml', 'r') as f:
        env_cfg = yaml.safe_load(f)['environment']
    
    with open('configs/hybrid_config.yaml', 'r') as f:
        hybrid_cfg = yaml.safe_load(f)['hybrid']
    
    # Modify to use GAT
    hybrid_cfg['architecture']['gnn_conv_type'] = 'GAT'
    
    trainer = HybridTrainer(
        env_config=env_cfg,
        hybrid_config=hybrid_cfg,
        log_dir="logs/hybrid_gat",
        model_dir="models/hybrid_gat",
        seed=42
    )
    
    trainer.train(total_timesteps=20000, eval_freq=5000)
    metrics = trainer.evaluate(n_episodes=50)
    
    print(f"\n✅ Training complete!")
    print(f"   Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"📁 Model saved to models/hybrid_gat/\n")
    
    trainer.close()


def train_with_attention_fusion():
    """Train hybrid model with attention-based fusion."""
    print("\n" + "="*80)
    print("🚀 TRAINING HYBRID MODEL WITH ATTENTION FUSION")
    print("="*80 + "\n")
    
    import yaml
    from src.agent.hybrid_trainer import HybridTrainer
    
    # Load configs
    with open('configs/env_config.yaml', 'r') as f:
        env_cfg = yaml.safe_load(f)['environment']
    
    with open('configs/hybrid_config.yaml', 'r') as f:
        hybrid_cfg = yaml.safe_load(f)['hybrid']
    
    # Modify to use attention fusion
    hybrid_cfg['fusion']['strategy'] = 'attention'
    
    trainer = HybridTrainer(
        env_config=env_cfg,
        hybrid_config=hybrid_cfg,
        log_dir="logs/hybrid_attention",
        model_dir="models/hybrid_attention",
        seed=42
    )
    
    trainer.train(total_timesteps=20000, eval_freq=5000)
    metrics = trainer.evaluate(n_episodes=50)
    
    print(f"\n✅ Training complete!")
    print(f"   Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"📁 Model saved to models/hybrid_attention/\n")
    
    trainer.close()


def show_menu():
    """Display interactive menu."""
    print("\n" + "="*80)
    print("🔬 PHASE 3: RESEARCH & EXPERIMENTATION")
    print("="*80)
    print("\nAvailable Experiments:")
    print("  1. Test Advanced GNN Encoders (GAT, GraphSAGE, Hybrid)")
    print("  2. Hyperparameter Optimization (Quick - 10 trials)")
    print("  3. Hyperparameter Optimization (Full - 50 trials)")
    print("  4. Ablation Study (Architecture, Fusion, Components)")
    print("  5. Train with GAT Encoder")
    print("  6. Train with Attention Fusion")
    print("  7. Run All Experiments (Full Phase 3)")
    print("  0. Exit")
    print("="*80)
    
    choice = input("\n👉 Select experiment (0-7): ").strip()
    return choice


def run_all():
    """Run all Phase 3 experiments."""
    print("\n" + "="*80)
    print("🚀 RUNNING ALL PHASE 3 EXPERIMENTS")
    print("="*80 + "\n")
    
    print("This will:")
    print("  ✅ Test all advanced encoders")
    print("  ✅ Run hyperparameter optimization")
    print("  ✅ Conduct ablation study")
    print("  ✅ Train models with GAT and Attention")
    print(f"\n⏱️  Estimated time: 2-4 hours\n")
    
    confirm = input("Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    # Run all experiments
    test_encoders()
    run_hyperparameter_tuning(mode='quick')
    train_with_gat()
    train_with_attention_fusion()
    
    print("\n" + "="*80)
    print("🎉 ALL PHASE 3 EXPERIMENTS COMPLETE!")
    print("="*80)
    print("\n📊 Results Summary:")
    print("   - Advanced encoders tested ✅")
    print("   - Hyperparameters optimized ✅")
    print("   - GAT model trained ✅")
    print("   - Attention fusion trained ✅")
    print("\n📁 Check reports/ and models/ folders for results!\n")


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Research & Experimentation")
    parser.add_argument('--experiment', type=str, choices=[
        'test-encoders', 'tune-quick', 'tune-full', 'ablation',
        'train-gat', 'train-attention', 'all'
    ], help='Experiment to run')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive or not args.experiment:
        # Interactive mode
        while True:
            choice = show_menu()
            
            if choice == '0':
                print("\n👋 Goodbye!\n")
                break
            elif choice == '1':
                test_encoders()
            elif choice == '2':
                run_hyperparameter_tuning(mode='quick')
            elif choice == '3':
                run_hyperparameter_tuning(mode='full')
            elif choice == '4':
                run_ablation_study()
            elif choice == '5':
                train_with_gat()
            elif choice == '6':
                train_with_attention_fusion()
            elif choice == '7':
                run_all()
            else:
                print("❌ Invalid choice. Try again.")
            
            input("\nPress Enter to continue...")
    
    else:
        # Command-line mode
        if args.experiment == 'test-encoders':
            test_encoders()
        elif args.experiment == 'tune-quick':
            run_hyperparameter_tuning(mode='quick')
        elif args.experiment == 'tune-full':
            run_hyperparameter_tuning(mode='full')
        elif args.experiment == 'ablation':
            run_ablation_study()
        elif args.experiment == 'train-gat':
            train_with_gat()
        elif args.experiment == 'train-attention':
            train_with_attention_fusion()
        elif args.experiment == 'all':
            run_all()


if __name__ == "__main__":
    main()
