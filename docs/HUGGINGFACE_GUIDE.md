# 🤗 Hugging Face Upload Guide

Complete guide for sharing your AI Edge Allocator models on Hugging Face Hub.

## 📋 Prerequisites

### 1. Create Hugging Face Account

1. Go to [https://huggingface.co/join](https://huggingface.co/join)
2. Sign up for a free account
3. Verify your email

### 2. Get Access Token

1. Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Choose "Write" access
4. Copy the token (keep it secret!)

### 3. Install Hugging Face Hub

```bash
pip install huggingface-hub
```

---

## 🚀 Quick Upload (Recommended)

### Step 1: Login

```bash
huggingface-cli login
# Paste your token when prompted
```

### Step 2: Create Repository

```bash
# Create model repository
huggingface-cli repo create DeepSea-IoT --type model

# Or use web interface at https://huggingface.co/new
```

### Step 3: Upload Files

```bash
# Clone the repository
git clone https://huggingface.co/<your-username>/DeepSea-IoT
cd DeepSea-IoT

# Copy your files
cp MODEL_CARD.md README.md
cp -r ../ai_edge_allocator/models ./models
cp -r ../ai_edge_allocator/configs ./configs
cp ../ai_edge_allocator/requirements.txt ./
cp ../ai_edge_allocator/LICENSE ./

# Commit and push
git add .
git commit -m "Upload AI Edge Allocator models"
git push
```

---

## 📦 What to Upload

### Essential Files

```
DeepSea-IoT/
├── README.md              # MODEL_CARD.md renamed
├── models/
│   ├── dqn/
│   │   └── best_model/
│   │       └── best_model.zip
│   ├── ppo/
│   │   └── best_model/
│   │       └── best_model.zip
│   └── hybrid/
│       ├── best_model.pt
│       └── final_model.pt
├── configs/
│   ├── env_config.yaml
│   ├── hybrid_config.yaml
│   └── sim_config.yaml
├── requirements.txt
├── LICENSE
└── .gitattributes         # For large files (see below)
```

### Optional Files

```
├── src/                   # Source code (optional)
├── examples/              # Usage examples
│   └── inference_example.py
└── figures/               # Visualizations
    └── architecture.png
```

---

## 📝 Step-by-Step Upload Process

### Method 1: Using Hugging Face Hub Python Library

Create `upload_to_hf.py`:

```python
from huggingface_hub import HfApi, create_repo
import os

# Configuration
USERNAME = "your-username"  # Replace with your HF username
REPO_NAME = "DeepSea-IoT"
TOKEN = "hf_xxx"  # Replace with your token or use login

# Create API instance
api = HfApi()

# Create repository (if it doesn't exist)
try:
    create_repo(
        repo_id=f"{USERNAME}/{REPO_NAME}",
        token=TOKEN,
        repo_type="model",
        private=False  # Set to True for private repo
    )
    print(f"✅ Created repository: {USERNAME}/{REPO_NAME}")
except Exception as e:
    print(f"Repository may already exist: {e}")

# Upload files
print("📤 Uploading files...")

# Upload README (Model Card)
api.upload_file(
    path_or_fileobj="MODEL_CARD.md",
    path_in_repo="README.md",
    repo_id=f"{USERNAME}/{REPO_NAME}",
    token=TOKEN
)

# Upload models directory
api.upload_folder(
    folder_path="models",
    repo_id=f"{USERNAME}/{REPO_NAME}",
    path_in_repo="models",
    token=TOKEN
)

# Upload configs
api.upload_folder(
    folder_path="configs",
    repo_id=f"{USERNAME}/{REPO_NAME}",
    path_in_repo="configs",
    token=TOKEN
)

# Upload other files
for file in ["requirements.txt", "LICENSE"]:
    if os.path.exists(file):
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=file,
            repo_id=f"{USERNAME}/{REPO_NAME}",
            token=TOKEN
        )

print("✅ Upload complete!")
print(f"🔗 View at: https://huggingface.co/{USERNAME}/{REPO_NAME}")
```

Then run:
```bash
python upload_to_hf.py
```

### Method 2: Using Git LFS (For Large Files)

If your models are large (>100MB):

```bash
# Install Git LFS
git lfs install

# Clone HF repo
git clone https://huggingface.co/<username>/DeepSea-IoT
cd DeepSea-IoT

# Track large files
git lfs track "*.zip"
git lfs track "*.pt"
git add .gitattributes

# Copy and commit files
cp MODEL_CARD.md README.md
cp -r ../ai_edge_allocator/models ./
cp -r ../ai_edge_allocator/configs ./
cp ../ai_edge_allocator/requirements.txt ./

git add .
git commit -m "Upload AI Edge Allocator models"
git push
```

---

## 🎨 Customize Your Model Card

Edit `MODEL_CARD.md` before uploading:

1. **Add your username** in links
2. **Update performance metrics** if you have new results
3. **Add screenshots** of your dashboard
4. **Include example outputs**
5. **Add your contact info**

Example additions:

```markdown
## Demo

Try it online: [Space Demo](https://huggingface.co/spaces/your-username/deepsea-iot-demo)

## Example Predictions

![Network Visualization](figures/network_viz.png)

### Sample Input
\```json
{
  "nodes": [...],
  "edges": [...]
}
\```

### Sample Output
\```json
{
  "selected_node": 5,
  "confidence": 0.87
}
\```
```

---

## 🌐 Create a Hugging Face Space (Optional)

Make your model interactive with a Gradio or Streamlit Space:

### Option 1: Streamlit Space

1. Create a new Space: https://huggingface.co/new-space
2. Choose "Streamlit" as the SDK
3. Upload your `python_scripts/dashboard/dashboard_app.py`
4. Add `requirements_dashboard.txt` as `requirements.txt`
5. Your dashboard will be live!

### Option 2: Gradio Interface

Create `app.py` for a simple interface:

```python
import gradio as gr
import torch
from src.agent.hybrid_trainer import HybridTrainer

# Load model
checkpoint = torch.load('models/hybrid/best_model.pt')
policy = checkpoint['policy']

def predict(cpu_util, mem_util, energy, latency, bandwidth, queue_len, node_type):
    # Create observation
    obs = [cpu_util, mem_util, energy, latency, bandwidth, queue_len, node_type]
    
    # Predict
    action, _ = policy.predict(obs, deterministic=True)
    
    return f"Selected Node: {action}"

# Create Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(0, 1, label="CPU Utilization"),
        gr.Slider(0, 1, label="Memory Utilization"),
        gr.Number(label="Energy (J)"),
        gr.Number(label="Latency (ms)"),
        gr.Number(label="Bandwidth (Mbps)"),
        gr.Number(label="Queue Length"),
        gr.Dropdown([0, 1, 2], label="Node Type")
    ],
    outputs="text",
    title="AI Edge Allocator",
    description="Predict optimal node placement for IoT tasks"
)

interface.launch()
```

---

## 📊 Add Metrics and Tags

In your `README.md` (MODEL_CARD.md), use YAML frontmatter:

```yaml
---
license: mit
tags:
- reinforcement-learning
- iot
- edge-computing
- resource-allocation
- graph-neural-network
- pytorch
- stable-baselines3
datasets:
- Sirius-ashwak/iot-network-traces  # If you upload data
metrics:
- reward
model-index:
- name: DeepSea-IoT-Hybrid
  results:
  - task:
      type: reinforcement-learning
      name: IoT Resource Allocation
    dataset:
      type: iot-network-traces
      name: IoT Network Traces
    metrics:
    - type: mean_reward
      value: 246.02
      name: Mean Reward
    - type: std_reward
      value: 8.57
      name: Std Reward
---
```

---

## 🔍 Verify Upload

After uploading, check:

1. **Model Card**: Should display nicely at `https://huggingface.co/<username>/DeepSea-IoT`
2. **Files Tab**: All files visible
3. **Download**: Test downloading your model
4. **License**: Properly set to MIT

---

## 🎯 Best Practices

### DO:
- ✅ Include a comprehensive README (Model Card)
- ✅ Add license file
- ✅ Include requirements.txt
- ✅ Add tags for discoverability
- ✅ Include usage examples
- ✅ Add performance metrics
- ✅ Include citations

### DON'T:
- ❌ Upload sensitive data or credentials
- ❌ Upload large files without Git LFS
- ❌ Forget to test model loading
- ❌ Skip the model card
- ❌ Use unclear naming

---

## 📥 Loading Your Model from Hub

After upload, users can load your model:

```python
from huggingface_hub import hf_hub_download
import torch

# Download model
model_path = hf_hub_download(
    repo_id="your-username/DeepSea-IoT",
    filename="models/hybrid/best_model.pt"
)

# Load model
checkpoint = torch.load(model_path)
policy = checkpoint['policy']

print("✅ Model loaded from Hugging Face!")
```

---

## 🌟 Promote Your Model

### Update README.md
Add Hugging Face badge:

```markdown
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/your-username/DeepSea-IoT)
```

### Share on Social Media
- Tweet about your model with #HuggingFace
- Share on LinkedIn
- Post on Reddit (r/MachineLearning)

### Update GitHub README
Link to your Hugging Face model:

```markdown
## 🤗 Pre-trained Models

Download our trained models from Hugging Face:
[https://huggingface.co/your-username/DeepSea-IoT](https://huggingface.co/your-username/DeepSea-IoT)
```

---

## 🐛 Troubleshooting

### Issue: Upload fails with "File too large"

**Solution**: Use Git LFS
```bash
git lfs track "*.pt"
git lfs track "*.zip"
```

### Issue: Authentication error

**Solution**: Login again
```bash
huggingface-cli login
# Or set token: export HF_TOKEN=your_token
```

### Issue: Model card not rendering

**Solution**: Check YAML frontmatter syntax
```yaml
---
license: mit
tags:
- pytorch
---
```

---

## 📚 Additional Resources

- **Hugging Face Hub Docs**: https://huggingface.co/docs/hub
- **Model Cards Guide**: https://huggingface.co/docs/hub/model-cards
- **Git LFS**: https://git-lfs.github.com
- **Creating Spaces**: https://huggingface.co/docs/hub/spaces

---

## 🎓 Example Repositories

Check out these for inspiration:
- https://huggingface.co/facebook/dqn-SpaceInvadersNoFrameskip-v4
- https://huggingface.co/sb3/ppo-CartPole-v1

---

**Ready to share your work with the world!** 🚀

Questions? Check [Hugging Face Docs](https://huggingface.co/docs) or ask on their [forums](https://discuss.huggingface.co).
