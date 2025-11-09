"""
Train the Phase 3 GAT configuration with optional warm-start data.

This script stitches together the existing environment + hybrid trainer
infrastructure, applies the Phase 3 overrides (GAT encoder, attention
fusion, cosine LR schedule, etc.), and orchestrates a full training run.
"""

import json
import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import yaml

from src.agent.hybrid_trainer import HybridTrainer
from src.utils.data_loader import compute_statistics
from src.utils.paths import get_project_root


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _deep_update(target: Dict, overrides: Dict) -> Dict:
    """Recursively merge ``overrides`` into ``target`` without clobbering sub-dicts."""

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _load_configs() -> Tuple[Dict, Dict]:
    """Load environment + hybrid configs and apply Phase 3 overrides."""

    repo_root = get_project_root()

    with open(repo_root / "configs" / "env_config.yaml", "r", encoding="utf-8") as f:
        env_cfg = yaml.safe_load(f)["environment"]

    with open(repo_root / "configs" / "hybrid_config.yaml", "r", encoding="utf-8") as f:
        base_hybrid_cfg = yaml.safe_load(f)["hybrid"]

    with open(repo_root / "configs" / "phase3_gat_config.yaml", "r", encoding="utf-8") as f:
        phase3_cfg = yaml.safe_load(f)

    hybrid_cfg = _deep_update(deepcopy(base_hybrid_cfg), phase3_cfg.get("hybrid", {}))

    return env_cfg, hybrid_cfg


def _resolve_dataset_path(path_str: str) -> Path:
    project_root = get_project_root()
    path = Path(path_str)
    if not path.is_absolute():
        path = (project_root / path).resolve()
    return path


def _load_warmup_dataset(data_cfg: Dict) -> Optional[Dict]:
    """Load the optional warm-start dataframe + descriptive statistics."""

    dataset_path = data_cfg.get("dataset_path")
    if not dataset_path:
        return None

    resolved_path = _resolve_dataset_path(dataset_path)
    if not resolved_path.exists():
        logger.warning("Warm-start dataset not found: %s", resolved_path)
        return None

    df = pd.read_csv(resolved_path)
    feature_names = [c for c in df.columns if c not in ("timestamp", "node_id")]

    if not feature_names:
        logger.warning("Warm-start dataset is missing feature columns")
        return None

    stats = compute_statistics(df, feature_names)

    print("\n📚 Warm-start dataset summary:")
    print(f"  - Path: {resolved_path.relative_to(Path.cwd()) if resolved_path.is_relative_to(Path.cwd()) else resolved_path}")
    print(f"  - Records: {len(df)}")
    print(f"  - Unique timesteps: {df['timestamp'].nunique()}")
    unique_nodes = df.groupby('timestamp')['node_id'].nunique()
    median_nodes = int(unique_nodes.median()) if not unique_nodes.empty else 0
    print(f"  - Median nodes per timestep: {median_nodes}")
    print("  - Feature statistics:")
    print(stats.to_string(index=False))

    return {
        "dataframe": df.sort_values(["timestamp", "node_id"]).reset_index(drop=True),
        "feature_names": feature_names,
        "stats": stats,
        "path": resolved_path,
    }


def train_gat_model() -> Dict:
    """Train the Phase 3 GAT hybrid agent and persist a result summary."""

    print("\n" + "=" * 80)
    print("🚀 TRAINING GAT MODEL (PHASE 3 CONFIG)")
    print("=" * 80)

    env_cfg, hybrid_cfg = _load_configs()

    print("\n📊 Configuration overview:")
    print(f"  - Encoder: {hybrid_cfg['architecture']['gnn_conv_type']}")
    print(f"  - Heads: {hybrid_cfg['architecture'].get('gat_heads', 'n/a')}")
    print(f"  - Fusion: {hybrid_cfg['fusion']['strategy']}")
    print(f"  - Total timesteps: {hybrid_cfg['training']['total_timesteps']}")

    warmup_bundle = _load_warmup_dataset(hybrid_cfg.get("data", {}))

    trainer: Optional[HybridTrainer] = None
    summary: Dict

    try:
        trainer = HybridTrainer(
            env_config=env_cfg,
            hybrid_config=hybrid_cfg,
            log_dir=hybrid_cfg.get("tensorboard", {}).get("log_dir", "logs/hybrid"),
            model_dir=hybrid_cfg.get("checkpoint", {}).get("save_path", "models/hybrid"),
            seed=hybrid_cfg.get("seed", 42),
        )

        if warmup_bundle:
            warmup_samples = int(hybrid_cfg.get("data", {}).get("warmup_samples", 1000))
            print(f"\n🔥 Preloading replay buffer with {warmup_samples} samples...")
            trainer.preload_from_dataframe(
                warmup_bundle["dataframe"],
                feature_names=warmup_bundle["feature_names"],
                max_samples=warmup_samples,
            )

        training_cfg = hybrid_cfg["training"]

        print("\n🎯 Starting training loop...")
        print(f"  - Eval frequency: {training_cfg['eval_freq']}")
        print(f"  - Eval episodes: {training_cfg['n_eval_episodes']}")
        print(f"  - Device: {trainer.device}")

        trainer.train(
            total_timesteps=training_cfg["total_timesteps"],
            batch_size=hybrid_cfg["dqn"]["batch_size"],
            eval_freq=training_cfg["eval_freq"],
            n_eval_episodes=training_cfg["n_eval_episodes"],
        )

        final_metrics = trainer.evaluate(n_episodes=training_cfg["n_eval_episodes"])
        best_reward = (
            trainer.best_eval_reward
            if trainer.best_eval_reward != float("-inf")
            else final_metrics["mean_reward"]
        )

        baseline_reward = 246.02
        improvement = ((best_reward - baseline_reward) / baseline_reward) * 100.0

        print("\n" + "=" * 80)
        print("📈 TRAINING RESULTS")
        print("=" * 80)
        print(f"🏆 Best evaluation reward: {best_reward:.2f}")
        print(
            f"📊 Final evaluation: {final_metrics['mean_reward']:.2f} ± {final_metrics['std_reward']:.2f}"
        )
        print(f"📍 Timesteps processed: {trainer.global_step}")
        print(f"🔁 Improvement vs. baseline (246.02): {improvement:+.2f}%")

        summary = {
            "experiment": "Phase3_GAT",
            "date": datetime.now().isoformat(),
            "best_reward": float(best_reward),
            "final_metrics": {k: float(v) for k, v in final_metrics.items()},
            "improvement_percent": float(improvement),
            "timesteps": int(trainer.global_step),
            "config": {
                "environment": env_cfg,
                "hybrid": hybrid_cfg,
            },
            "warm_start": {
                "dataset": str(warmup_bundle["path"])
                if warmup_bundle
                else None,
                "feature_names": warmup_bundle["feature_names"] if warmup_bundle else None,
            },
        }

        summary_path = Path("reports/gat_training_results.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\n📁 Results saved to: {summary_path}")

        return summary

    finally:
        if trainer is not None:
            trainer.close()


def main():
    """Entry-point wrapper that sets seeds and runs training."""

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    print(f"🔧 Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("🎲 Random seed: 42")

    try:
        train_gat_model()
        print("\n✅ Training completed successfully!")
        print("💡 Next steps:")
        print("  1. Compare the summary JSON against production benchmarks")
        print("  2. Promote the checkpoint in models/phase3_gat if results are better")
        print("  3. Update inference pipelines to consume the new encoder")
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        logger.exception("Training failed: %s", exc)
        print(f"\n❌ Training failed: {exc}")
        raise


if __name__ == "__main__":
    main()
