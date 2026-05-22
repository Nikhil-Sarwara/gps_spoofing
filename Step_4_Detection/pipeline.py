#!/usr/bin/env python3
"""
GPS Spoofing Detection Pipeline Runner

Runs the complete ML pipeline from raw data to trained model.

Usage:
    python -m pipeline run --config config/pipeline.yaml
    python -m pipeline process-single --raw Step_5_Data/raw/flight_001.csv
    python -m pipeline process-batch --raw-dir Step_5_Data/raw
    python -m pipeline train
    python -m pipeline full  # Run everything
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PipelineConfig:
    def __init__(self, config_path: Optional[str] = None):
        self.default_config = {
            "data": {
                "raw_dir": "Step_5_Data/raw",
                "processed_dir": "Step_5_Data/processed",
                "artifacts_dir": "Step_5_Data/artifacts",
                "models_dir": "Step_5_Data/models",
            },
            "cleaning": {
                "min_fix_type": 3,
                "max_dt_gap_s": 5.0,
                "max_stale_repeats": 5,
                "drop_duplicates": True,
            },
            "windows": {
                "length": 30,
                "stride": 15,
                "min_window_label_ratio": 0.5,
            },
            "split": {
                "train_ratio": 0.7,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
            },
            "training": {
                "model_type": "rf",
                "n_estimators": 100,
                "random_state": 42,
            },
        }
        self.config = self.default_config
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
                self.config = self._deep_merge(self.config, user_config)

    def _deep_merge(self, base: dict, update: dict) -> dict:
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def get(self, *keys, default=None):
        val = self.config
        for key in keys:
            if isinstance(val, dict):
                val = val.get(key)
            else:
                return default
        return val if val is not None else default


def _script_path(name: str) -> Path:
    return Path(__file__).parent / "scripts" / f"{name}.py"

def _run_script(name: str, *args, **kwargs):
    import importlib.util
    path = _script_path(name)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main(*args, **kwargs)


def run_clean(raw_path: Path, cfg: PipelineConfig) -> Path:
    """Run data cleaning step."""
    logger.info(f"Cleaning: {raw_path}")
    output_dir = Path(cfg.get("data", "processed_dir"))
    output_path = _run_script("01_clean_log", raw_path, output_dir, cfg.config["cleaning"])
    return output_path


def run_auto_label(cleaned_path: Path, cfg: PipelineConfig) -> Path:
    """Run automated labeling with a unique output file per flight."""
    logger.info(f"Auto-labeling: {cleaned_path}")
    # Create a unique labels file name based on the cleaned file name
    labels_name = cleaned_path.name.replace("_cleaned.csv", "_labels.csv")
    output_path = cleaned_path.parent / labels_name
    _run_script("02_auto_label", cleaned_path, output_path, cfg.config.get("auto_label", {}))
    return output_path


def run_windows(cleaned_path: Path, labels_path: Path, cfg: PipelineConfig) -> Path:
    """Create sliding windows."""
    logger.info(f"Creating windows from: {cleaned_path}")
    output_dir = Path(cfg.get("data", "artifacts_dir"))
    return _run_script("03_make_windows", cleaned_path, labels_path, output_dir, cfg.config.get("windows"), cfg.config.get("split"))


def run_training(cfg: PipelineConfig):
    """Train the model."""
    logger.info("Training model...")
    artifacts_dir = Path(cfg.get("data", "artifacts_dir"))
    models_dir = Path(cfg.get("data", "models_dir"))
    _run_script("04_train_baseline", artifacts_dir, models_dir, cfg.config.get("training"))


def process_single(raw_path: str, cfg: PipelineConfig):
    """Process a single raw log file."""
    raw_path = Path(raw_path)
    if not raw_path.exists():
        logger.error(f"File not found: {raw_path}")
        return

    cleaned_path = run_clean(raw_path, cfg)
    labels_path = run_auto_label(cleaned_path, cfg)
    artifacts = run_windows(cleaned_path, labels_path, cfg)

    logger.info(f"Pipeline complete! Artifacts: {artifacts}")


def process_batch(raw_dir: str, cfg: PipelineConfig):
    """Process all raw log files in directory (Clean and Label only)."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        logger.error(f"Directory not found: {raw_dir}")
        return

    raw_files = sorted(raw_dir.glob("**/*.csv"))
    # Filter out files that are already in the 'processed' directory
    raw_files = [f for f in raw_files if "processed" not in str(f)]
    
    if not raw_files:
        logger.warning(f"No CSV files found in {raw_dir}")
        return

    logger.info(f"Found {len(raw_files)} files to clean and label")

    for raw_path in raw_files:
        logger.info(f"Cleaning & Labeling: {raw_path.name}")
        try:
            cleaned_path = run_clean(raw_path, cfg)
            run_auto_label(cleaned_path, cfg)
        except Exception as e:
            logger.error(f"Failed to process {raw_path}: {e}")


def run_full(cfg: PipelineConfig):
    """Run the complete pipeline: Clean -> Label -> Aggregated Windows -> Train."""
    raw_dir = Path(cfg.get("data", "raw_dir"))
    if raw_dir.exists():
        process_batch(raw_dir, cfg)
    
    # Aggregated window creation
    logger.info("Creating aggregated windows from all processed and synthetic data...")
    _run_script("03_make_windows", 
                Path(cfg.get("data", "raw_dir")).parent, # top-level 'Step_5_Data'
                None, # labels_path not used in dir mode
                Path(cfg.get("data", "artifacts_dir")), 
                cfg.config.get("windows"), 
                cfg.config.get("split"))
    
    run_training(cfg)
    logger.info("\n✅ Full pipeline complete!")


def main():
    parser = argparse.ArgumentParser(description="GPS Spoofing ML Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    run_parser = subparsers.add_parser("run", help="Run pipeline with config")
    run_parser.add_argument("-c", "--config", default="4_Detection_Engine/config/pipeline.yaml", help="Config file")

    single_parser = subparsers.add_parser("process-single", help="Process single file")
    single_parser.add_argument("raw", help="Path to raw CSV")
    single_parser.add_argument("-c", "--config", default="4_Detection_Engine/config/pipeline.yaml")

    batch_parser = subparsers.add_parser("process-batch", help="Process all files in directory")
    batch_parser.add_argument("--raw-dir", default="Step_5_Data/raw")
    batch_parser.add_argument("-c", "--config", default="4_Detection_Engine/config/pipeline.yaml")

    train_parser = subparsers.add_parser("train", help="Train model only")
    train_parser.add_argument("-c", "--config", default="4_Detection_Engine/config/pipeline.yaml")

    full_parser = subparsers.add_parser("full", help="Run complete pipeline")
    full_parser.add_argument("-c", "--config", default="4_Detection_Engine/config/pipeline.yaml")

    subparsers.add_parser("full-terrain", help="Process data, label terrain, train global + terrain models")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    # Look for config in root if not specified
    if args.config == "4_Detection_Engine/config/pipeline.yaml" and not Path(args.config).exists():
        # Fallback for if we are running from 4_Detection_Engine or other places
        potential_config = Path(__file__).parent / "config" / "pipeline.yaml"
        if potential_config.exists():
            args.config = str(potential_config)

    cfg = PipelineConfig(args.config)

    if args.command == "run":
        run_full(cfg)
    elif args.command == "process-single":
        process_single(args.raw, cfg)
    elif args.command == "process-batch":
        process_batch(args.raw_dir, cfg)
    elif args.command == "train":
        run_training(cfg)
    elif args.command == "full":
        run_full(cfg)
    elif args.command == "full-terrain":
        import subprocess, sys, logging
        _logger = logging.getLogger(__name__)
        _PROJECT_ROOT = Path(__file__).parent.parent
        _logger.info("=== FULL TERRAIN PIPELINE ===")

        _logger.info("[1/3] Processing raw GPS logs...")
        process_batch(str(_PROJECT_ROOT / "Step_5_Data" / "raw"), cfg)

        _logger.info("[2/3] Training global baseline model...")
        run_training(cfg)

        _logger.info("[3/3] Training per-terrain models...")
        _terrain_cmd = [
            sys.executable,
            str(_PROJECT_ROOT / "4_Detection_Engine" / "scripts" / "05_train_terrain_models.py"),
            "--processed-dirs",
            str(_PROJECT_ROOT / "Step_5_Data" / "processed"),
            "--models-dir",   str(_PROJECT_ROOT / "Step_5_Data" / "models" / "terrain"),
            "--artifacts-dir", str(_PROJECT_ROOT / "Step_5_Data" / "artifacts"),
        ]
        _result = subprocess.run(_terrain_cmd, cwd=str(_PROJECT_ROOT))
        if _result.returncode != 0:
            _logger.error("Terrain model training failed.")
            sys.exit(1)
        _logger.info("=== FULL TERRAIN PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
