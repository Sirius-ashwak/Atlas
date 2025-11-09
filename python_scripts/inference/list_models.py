"""
List Available Models (Like Hugging Face CLI)
==============================================

This script lists all trained models stored locally.
Similar to: transformers-cli list
"""

from pathlib import Path
import json

def list_models():
    """List all available models in the models/ directory."""
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("❌ Models directory not found!")
        return
    
    print("="*70)
    print("📦 AVAILABLE MODELS (Like Hugging Face Local Cache)")
    print("="*70)
    print()
    
    model_types = ["dqn", "ppo", "hybrid"]
    total_size = 0
    
    for model_type in model_types:
        model_path = models_dir / model_type
        if not model_path.exists():
            continue
        
        # Count model files
        zip_files = list(model_path.glob("**/*.zip"))
        pt_files = list(model_path.glob("**/*.pt"))
        all_files = zip_files + pt_files
        
        if not all_files:
            continue
        
        # Calculate total size
        size_mb = sum(f.stat().st_size for f in all_files) / (1024 * 1024)
        total_size += size_mb
        
        print(f"🤖 {model_type.upper()}")
        print(f"   Location: {model_path}")
        print(f"   Files: {len(all_files)}")
        print(f"   Size: {size_mb:.2f} MB")
        
        # List individual files
        for file in all_files:
            file_size = file.stat().st_size / (1024 * 1024)
            print(f"   └─ {file.name} ({file_size:.2f} MB)")
        
        print()
    
    print("="*70)
    print(f"📊 Total Models: {len([d for d in models_dir.iterdir() if d.is_dir()])}")
    print(f"💾 Total Size: {total_size:.2f} MB")
    print("="*70)
    print()
    print("💡 Usage (like Hugging Face):")
    print("   python python_scripts/inference/local_inference.py --model-type hybrid")
    print("   python python_scripts/inference/local_inference.py --model-type dqn")
    print("   python python_scripts/inference/local_inference.py --model-type ppo")
    print()

if __name__ == "__main__":
    list_models()
