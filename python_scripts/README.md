# Python Script Organization

The `python_scripts/` directory groups all command-line utilities for the AI
Edge Allocator project by their purpose. The goal is to keep the repository
root tidy while making it easier to discover related tooling.

## Directory Layout

- `api/` – FastAPI entrypoints and helpers.
- `analysis/` – Offline analytics and visualization pipelines.
- `inference/` – Local inference, benchmarking, and model listing tools.
- `training/` – Training utilities for GAT and hybrid agents.
- `testing/` – Smoke tests and quick validation scripts.
- `dashboard/` – Streamlit dashboards and other UI scripts.
- `simulation/` – IoT simulator utilities.
- `documentation/` – Scripts that manage project documentation.
- `utilities/` – Miscellaneous helpers such as deployment and sharing tools.

Each script can still be executed directly, for example:

```bash
python python_scripts/api/run_api.py --port 8000
python python_scripts/inference/local_inference.py --model-type hybrid
python python_scripts/training/run_phase3.py --experiment train-gat
```

The documentation has been updated to reference these new paths so existing
instructions remain accurate.
