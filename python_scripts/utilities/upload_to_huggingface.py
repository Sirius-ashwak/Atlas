"""
Upload AI Edge Allocator models to Hugging Face Hub.

Usage:
    python upload_to_huggingface.py --username your-username --token your-token
    
Or use interactive mode:
    python upload_to_huggingface.py --interactive
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, login
import shutil


def check_requirements():
    """Check if required packages are installed."""
    try:
        import huggingface_hub
        print("✅ huggingface_hub installed")
        return True
    except ImportError:
        print("❌ huggingface_hub not installed")
        print("   Install with: pip install huggingface-hub")
        return False


def prepare_upload_directory():
    """Prepare a temporary directory with files to upload."""
    print("\n📦 Preparing files for upload...")
    
    upload_dir = Path("./hf_upload")
    upload_dir.mkdir(exist_ok=True)
    
    # Copy README (Model Card)
    print("   Copying MODEL_CARD.md → README.md")
    shutil.copy("MODEL_CARD.md", upload_dir / "README.md")
    
    # Copy models directory
    if os.path.exists("models"):
        print("   Copying models/")
        shutil.copytree("models", upload_dir / "models", dirs_exist_ok=True)
    else:
        print("   ⚠️  models/ directory not found")
    
    # Copy configs
    if os.path.exists("configs"):
        print("   Copying configs/")
        shutil.copytree("configs", upload_dir / "configs", dirs_exist_ok=True)
    
    # Copy other essential files
    for file in ["requirements.txt", "LICENSE"]:
        if os.path.exists(file):
            print(f"   Copying {file}")
            shutil.copy(file, upload_dir / file)
    
    # Create .gitattributes for Git LFS
    print("   Creating .gitattributes")
    with open(upload_dir / ".gitattributes", "w") as f:
        f.write("*.pt filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.zip filter=lfs diff=lfs merge=lfs -text\n")
        f.write("*.pth filter=lfs diff=lfs merge=lfs -text\n")
    
    print(f"\n✅ Files prepared in {upload_dir}/")
    return upload_dir


def upload_to_hf(username: str, repo_name: str = "DeepSea-IoT", token: str = None):
    """Upload files to Hugging Face Hub."""
    
    # Login
    if token:
        login(token=token)
        print("✅ Logged in with token")
    else:
        print("Using cached HF credentials...")
    
    # Create API instance
    api = HfApi()
    
    # Create repository
    repo_id = f"{username}/{repo_name}"
    print(f"\n🏗️  Creating repository: {repo_id}")
    
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            exist_ok=True
        )
        print(f"✅ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"⚠️  Repository may already exist: {e}")
    
    # Prepare files
    upload_dir = prepare_upload_directory()
    
    # Upload entire directory
    print(f"\n📤 Uploading files to {repo_id}...")
    print("   This may take a few minutes for large models...")
    
    try:
        api.upload_folder(
            folder_path=str(upload_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload AI Edge Allocator models and configs"
        )
        print(f"\n✅ Upload complete!")
        print(f"🔗 View your model at: https://huggingface.co/{repo_id}")
        
        # Cleanup
        print("\n🧹 Cleaning up temporary files...")
        shutil.rmtree(upload_dir)
        print("✅ Cleanup complete")
        
        return True
    
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check your token has write permissions")
        print("2. Ensure you're logged in: huggingface-cli login")
        print("3. Try manual upload (see HUGGINGFACE_GUIDE.md)")
        return False


def interactive_mode():
    """Interactive mode for uploading."""
    print("\n" + "="*80)
    print("🤗 HUGGING FACE UPLOAD - INTERACTIVE MODE")
    print("="*80 + "\n")
    
    # Get username
    username = input("Enter your Hugging Face username: ").strip()
    if not username:
        print("❌ Username cannot be empty")
        return
    
    # Get repo name
    repo_name = input("Repository name [DeepSea-IoT]: ").strip() or "DeepSea-IoT"
    
    # Ask about token
    print("\nDo you want to:")
    print("  1. Use cached credentials (huggingface-cli login)")
    print("  2. Provide token now")
    
    choice = input("Choice (1/2): ").strip()
    
    token = None
    if choice == "2":
        token = input("Enter your Hugging Face token: ").strip()
        if not token:
            print("❌ Token cannot be empty")
            return
    
    # Confirm
    print(f"\n📋 Summary:")
    print(f"   Repository: {username}/{repo_name}")
    print(f"   Token: {'Provided' if token else 'Using cached'}")
    
    confirm = input("\nProceed with upload? (y/n): ").strip().lower()
    
    if confirm == 'y':
        upload_to_hf(username, repo_name, token)
    else:
        print("❌ Upload cancelled")


def main():
    parser = argparse.ArgumentParser(
        description="Upload AI Edge Allocator models to Hugging Face Hub"
    )
    parser.add_argument(
        '--username',
        type=str,
        help='Your Hugging Face username'
    )
    parser.add_argument(
        '--repo-name',
        type=str,
        default='DeepSea-IoT',
        help='Repository name (default: DeepSea-IoT)'
    )
    parser.add_argument(
        '--token',
        type=str,
        help='Your Hugging Face token (or use huggingface-cli login)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    # Check requirements
    if not check_requirements():
        print("\nInstall required packages:")
        print("  pip install huggingface-hub")
        return
    
    print("\n" + "="*80)
    print("🤗 HUGGING FACE MODEL UPLOAD")
    print("="*80)
    
    # Run interactive or command-line mode
    if args.interactive or not args.username:
        interactive_mode()
    else:
        upload_to_hf(args.username, args.repo_name, args.token)
    
    print("\n" + "="*80)
    print("📚 For more information, see HUGGINGFACE_GUIDE.md")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
