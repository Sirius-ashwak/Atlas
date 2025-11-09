"""
Quick script to enable GAT training in existing infrastructure
"""

import yaml
from pathlib import Path


def create_gat_config():
    """Create a working GAT configuration based on existing hybrid config."""
    
    # Load existing hybrid config
    hybrid_config_path = Path("configs/hybrid_config.yaml")
    
    if hybrid_config_path.exists():
        with open(hybrid_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
    else:
        # Create minimal config if none exists
        base_config = {
            'hybrid': {
                'architecture': {
                    'use_gnn': True,
                    'gnn_hidden_dim': 64,
                    'gnn_num_layers': 3,
                    'gnn_conv_type': 'GCN'
                },
                'dqn': {
                    'enabled': True,
                    'learning_rate': 0.0001,
                    'buffer_size': 100000,
                    'batch_size': 64
                },
                'ppo': {
                    'enabled': True,
                    'learning_rate': 0.0003,
                    'n_steps': 2048,
                    'batch_size': 64
                },
                'fusion': {
                    'strategy': 'weighted_sum',
                    'dqn_weight': 0.6,
                    'ppo_weight': 0.4
                },
                'training': {
                    'total_timesteps': 10000,
                    'eval_freq': 1000
                }
            }
        }
    
    # Modify for GAT
    gat_config = base_config.copy()
    
    # Update architecture for GAT
    gat_config['hybrid']['architecture']['gnn_conv_type'] = 'GAT'
    gat_config['hybrid']['architecture']['gat_heads'] = 4
    gat_config['hybrid']['architecture']['gat_dropout'] = 0.1
    gat_config['hybrid']['architecture']['gat_concat'] = True
    
    # Update fusion to attention-based
    gat_config['hybrid']['fusion']['strategy'] = 'attention'
    gat_config['hybrid']['fusion']['attention_hidden_dim'] = 64
    gat_config['hybrid']['fusion']['temperature'] = 0.5
    
    # Update training for early stopping
    gat_config['hybrid']['training']['total_timesteps'] = 5000
    gat_config['hybrid']['training']['eval_freq'] = 500
    gat_config['hybrid']['training']['early_stopping'] = {
        'enabled': True,
        'patience': 3,
        'min_delta': 0.5,
        'monitor': 'mean_reward',
        'mode': 'max'
    }
    
    # Save GAT config
    gat_config_path = Path("configs/gat_config.yaml")
    gat_config_path.parent.mkdir(exist_ok=True)
    
    with open(gat_config_path, 'w') as f:
        yaml.dump(gat_config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created GAT config: {gat_config_path}")
    return gat_config


def update_phase3_config():
    """Update the existing phase3_gat_config.yaml to be more practical."""
    
    config_path = Path("configs/phase3_gat_config.yaml")
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Make it more practical for actual training
        config['hybrid']['training']['total_timesteps'] = 5000  # Reasonable for testing
        config['hybrid']['training']['eval_freq'] = 500
        config['hybrid']['training']['n_eval_episodes'] = 10  # Faster evaluation
        
        # Ensure checkpoint directory exists
        config['hybrid']['checkpoint']['save_path'] = "models/phase3_gat"
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Updated Phase 3 config: {config_path}")
    else:
        print(f"⚠️  Phase 3 config not found: {config_path}")


def main():
    """Enable GAT training in the project."""
    
    print("🔧 Enabling GAT training...")
    
    # Create configs directory if it doesn't exist
    Path("configs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("models/phase3_gat").mkdir(exist_ok=True)
    
    # Create/update configurations
    gat_config = create_gat_config()
    update_phase3_config()
    
    print("\n✅ GAT training enabled!")
    print("\n📋 Next steps:")
    print("1. Run: python train_gat_model.py")
    print("2. Or use existing trainer with GAT config:")
    print("   python -m src.main train-hybrid --config configs/gat_config.yaml")
    print("\n📊 GAT features enabled:")
    print("- Multi-head attention (4 heads)")
    print("- Attention-based fusion")
    print("- Early stopping")
    print("- Automatic checkpointing")


if __name__ == "__main__":
    main()
