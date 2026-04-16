#!/usr/bin/env python3
"""
GPS Spoofing Detection Pipeline Runner

Runs the complete ML pipeline from raw data to trained model.

Usage:
    python -m ml.pipeline run --config config/pipeline.yaml
    python -m ml.pipeline process-single --raw gps_logs/raw/flight_001.csv
    python -m ml.pipeline process-batch --raw-dir gps_logs/raw
    python -m ml.pipeline train
    python -m ml.pipeline full  # Run everything
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
                "raw_dir": "gps_logs/raw",
                "processed_dir": "gps_logs/processed",
                "artifacts_dir": "ml/artifacts",
                "models_dir": "ml/models",
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


def run_clean(raw_path: Path, cfg: PipelineConfig) -> Path:
    """Run data cleaning step."""
    logger.info(f"Cleaning: {raw_path}")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ml.scripts import clean_log

    output_dir = Path(cfg.get("data", "processed_dir"))
    output_path = clean_log.main(raw_path, output_dir, cfg.config["cleaning"])
    return output_path


def run_auto_label(cleaned_path: Path, cfg: PipelineConfig) -> Path:
    """Run automated labeling."""
    logger.info(f"Auto-labeling: {cleaned_path}")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ml.scripts import auto_label

    output_path = cleaned_path.parent / "row_labels_auto.csv"
    auto_label.main(cleaned_path, output_path, cfg.config.get("auto_label", {}))
    return output_path


def run_windows(labels_path: Path, cfg: PipelineConfig) -> Path:
    """Create sliding windows."""
    logger.info(f"Creating windows from: {labels_path}")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ml.scripts import make_windows

    output_dir = Path(cfg.get("data", "artifacts_dir"))
    artifacts = make_windows.main(labels_path, output_dir, cfg.config["windows"], cfg.config["split"])
    return artifacts


def run_training(cfg: PipelineConfig):
    """Train the model."""
    logger.info("Training model...")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ml.scripts import train_baseline

    artifacts_dir = Path(cfg.get("data", "artifacts_dir"))
    models_dir = Path(cfg.get("data", "models_dir"))
    train_baseline.main(artifacts_dir, models_dir, cfg.config["training"])


def process_single(raw_path: str, cfg: PipelineConfig):
    """Process a single raw log file."""
    raw_path = Path(raw_path)
    if not raw_path.exists():
        logger.error(f"File not found: {raw_path}")
        return

    cleaned_path = run_clean(raw_path, cfg)
    labels_path = run_auto_label(cleaned_path, cfg)
    artifacts = run_windows(labels_path, cfg)

    logger.info(f"Pipeline complete! Artifacts: {artifacts}")


def process_batch(raw_dir: str, cfg: PipelineConfig):
    """Process all raw log files in directory."""
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        logger.error(f"Directory not found: {raw_dir}")
        return

    raw_files = sorted(raw_dir.glob("*.csv"))
    if not raw_files:
        logger.warning(f"No CSV files found in {raw_dir}")
        return

    logger.info(f"Found {len(raw_files)} files to process")

    for raw_path in raw_files:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Processing: {raw_path.name}")
        try:
            process_single(raw_path, cfg)
        except Exception as e:
            logger.error(f"Failed to process {raw_path}: {e}")


def run_full(cfg: PipelineConfig):
    """Run the complete pipeline."""
    raw_dir = Path(cfg.get("data", "raw_dir"))
    if raw_dir.exists():
        process_batch(raw_dir, cfg)
    run_training(cfg)
    logger.info("\n✅ Full pipeline complete!")


def main():
    parser = argparse.ArgumentParser(description="GPS Spoofing ML Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    run_parser = subparsers.add_parser("run", help="Run pipeline with config")
    run_parser.add_argument("-c", "--config", default="config/pipeline.yaml", help="Config file")

    single_parser = subparsers.add_parser("process-single", help="Process single file")
    single_parser.add_argument("raw", help="Path to raw CSV")
    single_parser.add_argument("-c", "--config", default="config/pipeline.yaml")

    batch_parser = subparsers.add_parser("process-batch", help="Process all files in directory")
    batch_parser.add_argument("--raw-dir", default="gps_logs/raw")
    batch_parser.add_argument("-c", "--config", default="config/pipeline.yaml")

    train_parser = subparsers.add_parser("train", help="Train model only")
    train_parser.add_argument("-c", "--config", default="config/pipeline.yaml")

    full_parser = subparsers.add_parser("full", help="Run complete pipeline")
    full_parser.add_argument("-c", "--config", default="config/pipeline.yaml")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

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


if __name__ == "__main__":
    main()
