"""
Organize documentation files into docs/ directory.

This script moves markdown files to a cleaner structure.

Usage:
    python organize_docs.py
"""

import shutil
from pathlib import Path


def organize_documentation():
    """Move documentation files to docs/ directory."""
    
    print("\n" + "="*80)
    print("📚 ORGANIZING DOCUMENTATION")
    print("="*80 + "\n")
    
    # Create docs directory if it doesn't exist
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    print(f"✅ Created/verified docs/ directory")
    
    # Files to move to docs/
    files_to_move = {
        # Phase guides
        "PHASE3_GUIDE.md": "docs/PHASE3_GUIDE.md",
        "PHASE4_SUMMARY.md": "docs/PHASE4_SUMMARY.md",
        
        # Deployment docs
        "API_GUIDE.md": "docs/API_GUIDE.md",
        "DASHBOARD_GUIDE.md": "docs/DASHBOARD_GUIDE.md",
        "DOCKER_GUIDE.md": "docs/DOCKER_GUIDE.md",
        
        # Model sharing
        "HUGGINGFACE_GUIDE.md": "docs/HUGGINGFACE_GUIDE.md",
        "MODEL_CARD.md": "docs/MODEL_CARD.md",
        
        # Original docs
        "GETTING_STARTED.md": "docs/GETTING_STARTED.md",
        "PROJECT_SUMMARY.md": "docs/PROJECT_SUMMARY.md",
    }
    
    # Files to keep in root
    root_files = [
        "README.md",
        "LICENSE",
        "requirements.txt",
        "requirements_api.txt",
        "requirements_dashboard.txt",
    ]
    
    print("\n📦 Moving documentation files:\n")
    
    moved_count = 0
    for source, dest in files_to_move.items():
        source_path = Path(source)
        dest_path = Path(dest)
        
        if source_path.exists():
            # Move file
            shutil.move(str(source_path), str(dest_path))
            print(f"   ✅ {source} → {dest}")
            moved_count += 1
        else:
            print(f"   ⏭️  {source} (not found, skipping)")
    
    print(f"\n✅ Moved {moved_count} files to docs/")
    
    # Create a DOCS_INDEX.md in root pointing to docs/
    print("\n📝 Creating DOCS_INDEX.md in root...")
    
    index_content = """# 📚 Documentation

All project documentation has been organized in the `docs/` directory.

## 📖 Quick Links

- **[Complete Documentation Index](docs/README.md)** - All documentation
- **[Quick Start](docs/QUICKSTART.md)** - Get started in 5 minutes
- **[API Guide](docs/API_GUIDE.md)** - REST API documentation
- **[Dashboard Guide](docs/DASHBOARD_GUIDE.md)** - Streamlit dashboard
- **[Docker Guide](docs/DOCKER_GUIDE.md)** - Container deployment
- **[Hugging Face Guide](docs/HUGGINGFACE_GUIDE.md)** - Share your models

## 🗂️ Documentation Structure

```
docs/
├── README.md                 # Main documentation index
├── QUICKSTART.md            # Quick start guide
├── GETTING_STARTED.md       # Detailed setup
│
├── Phase Guides/
│   ├── PHASE3_GUIDE.md      # Advanced experiments
│   └── PHASE4_SUMMARY.md    # Deployment guide
│
└── Deployment/
    ├── API_GUIDE.md         # FastAPI docs
    ├── DASHBOARD_GUIDE.md   # Streamlit docs
    ├── DOCKER_GUIDE.md      # Docker docs
    ├── HUGGINGFACE_GUIDE.md # HF upload
    └── MODEL_CARD.md        # Model card
```

## 🚀 Getting Started

1. See [README.md](README.md) for project overview
2. Follow [Quick Start Guide](docs/QUICKSTART.md) to set up
3. Explore [Phase 4 Summary](docs/PHASE4_SUMMARY.md) for deployment

---

**Browse all documentation**: [docs/README.md](docs/README.md)
"""
    
    with open("DOCS_INDEX.md", "w") as f:
        f.write(index_content)
    
    print("✅ Created DOCS_INDEX.md")
    
    print("\n" + "="*80)
    print("✅ DOCUMENTATION ORGANIZED!")
    print("="*80)
    
    print("\n📋 Summary:")
    print(f"   Files moved to docs/: {moved_count}")
    print(f"   Root documentation: {len(root_files)} files")
    print(f"\n📚 Browse documentation:")
    print(f"   Main index: docs/README.md")
    print(f"   Quick links: DOCS_INDEX.md")
    print("\n")


if __name__ == "__main__":
    try:
        organize_documentation()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run this script from the ai_edge_allocator/ directory")
