#!/usr/bin/env python3
"""Validate DETR-s with original RT-DETRv2 PyTorch evaluator on KITTI corruptions."""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
from pathlib import Path
import os

import yaml

# Portable path defaults; run from the repository root or set these environment variables.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data/kitti_3cls_yolo")).resolve()
WEIGHTS_ROOT = Path(os.environ.get("WEIGHTS_ROOT", "weights")).resolve()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "outputs")).resolve()


RTDETR_DIR = Path(os.environ.get("RTDETR_ROOT", "third_party/RT-DETR/rtdetrv2_pytorch")).resolve()
# The external evaluator, compatible config, and checkpoint must be supplied by the user.
BASE_CFG = Path(os.environ.get("RTDETR_CONFIG", str(RTDETR_DIR / "configs/rtdetrv2/rtdetrv2_r18vd_kitti3.yml"))).resolve()
WEIGHT = WEIGHTS_ROOT / "detr_s/best.pth"

CORRUPT_ROOT = Path(str(DATA_ROOT / 'kitti_corrupt'))
ANN_FILE = Path(str(DATA_ROOT / 'coco_rtdetrv2/annotations/instances_val.json'))

OUTPUT_DIR = OUTPUT_ROOT / "robustness_eval"
RAW_CSV = OUTPUT_DIR / "robustness_raw.csv"
LOG_DIR = OUTPUT_DIR / "detrs_original_logs"

MODEL_NAME = "detr_s"

CONDITIONS = [
    ("clean", 0, "clean/images"),

    ("fog", 1, "fog/sev1/images"),
    ("fog", 3, "fog/sev3/images"),
    ("fog", 5, "fog/sev5/images"),

    ("lowlight", 1, "lowlight/sev1/images"),
    ("lowlight", 3, "lowlight/sev3/images"),
    ("lowlight", 5, "lowlight/sev5/images"),

    ("motion_blur", 1, "motion_blur/sev1/images"),
    ("motion_blur", 3, "motion_blur/sev3/images"),
    ("motion_blur", 5, "motion_blur/sev5/images"),

    ("gaussian_noise", 1, "gaussian_noise/sev1/images"),
    ("gaussian_noise", 3, "gaussian_noise/sev3/images"),
    ("gaussian_noise", 5, "gaussian_noise/sev5/images"),

    ("contrast", 1, "contrast/sev1/images"),
    ("contrast", 3, "contrast/sev3/images"),
    ("contrast", 5, "contrast/sev5/images"),
]

CSV_COLUMNS = [
    "model",
    "weight_path",
    "corruption",
    "severity",
    "yaml_path",
    "mAP50",
    "mAP50_95",
    "P",
    "R",
    "Car_mAP50",
    "Pedestrian_mAP50",
    "Cyclist_mAP50",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite-detrs", action="store_true")
    parser.add_argument("--skip-errors", action="store_true")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conditions", nargs="+", default=None)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def completed_keys(path: Path) -> set[tuple[str, str, int]]:
    keys = set()
    for row in read_csv_rows(path):
        if row.get("model") and row.get("corruption") and row.get("severity"):
            keys.add((row["model"], row["corruption"], int(row["severity"])))
    return keys


def remove_existing_detrs_rows(path: Path) -> None:
    rows = [r for r in read_csv_rows(path) if r.get("model") != MODEL_NAME]
    write_csv_rows(path, rows)


def make_condition_config(corruption: str, severity: int, img_dir: Path, batch: int, workers: int) -> Path:
    with BASE_CFG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["val_dataloader"]["dataset"]["img_folder"] = str(img_dir)
    cfg["val_dataloader"]["dataset"]["ann_file"] = str(ANN_FILE)
    cfg["val_dataloader"]["num_workers"] = workers
    cfg["val_dataloader"]["total_batch_size"] = batch
    cfg["output_dir"] = str(OUTPUT_DIR / "detrs_original_val_runs" / f"{corruption}_sev{severity}")

    generated_cfg = BASE_CFG.parent / f"rtdetrv2_r18vd_kitti3__robust_{MODEL_NAME}_{corruption}_sev{severity}.yml"
    with generated_cfg.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    return generated_cfg


def parse_metrics(text: str) -> tuple[float, float, float]:
    """Parse COCO evaluator AP50, AP50-95 and AR100 from RT-DETR stdout."""
    map50_95 = math.nan
    map50 = math.nan
    ar100 = math.nan

    for line in text.splitlines():
        line = line.strip()

        # Only parse official COCO evaluator summary lines.
        if not (line.startswith("Average Precision") or line.startswith("Average Recall")):
            continue

        value_match = re.search(r"=\s*([0-9]+(?:\.[0-9]+)?)\s*$", line)
        if not value_match:
            continue

        value = float(value_match.group(1))

        if (
            line.startswith("Average Precision")
            and "IoU=0.50:0.95" in line
            and "area=   all" in line
            and "maxDets=100" in line
        ):
            map50_95 = value

        elif (
            line.startswith("Average Precision")
            and "IoU=0.50" in line
            and "IoU=0.50:0.95" not in line
            and "area=   all" in line
            and "maxDets=100" in line
        ):
            map50 = value

        elif (
            line.startswith("Average Recall")
            and "IoU=0.50:0.95" in line
            and "area=   all" in line
            and "maxDets=100" in line
        ):
            ar100 = value

    return map50, map50_95, ar100


def validate_inputs() -> None:
    for path in [RTDETR_DIR, BASE_CFG, WEIGHT, ANN_FILE, CORRUPT_ROOT]:
        if not path.exists():
            raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    validate_inputs()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.overwrite_detrs and RAW_CSV.is_file():
        remove_existing_detrs_rows(RAW_CSV)

    selected_names = set(args.conditions) if args.conditions else None
    conditions = [c for c in CONDITIONS if selected_names is None or c[0] in selected_names]

    done = completed_keys(RAW_CSV)

    for corruption, severity, img_rel in conditions:
        key = (MODEL_NAME, corruption, severity)
        if key in done:
            print(f"Skipping completed: {key}")
            continue

        img_dir = CORRUPT_ROOT / img_rel
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing image dir: {img_dir}")

        generated_cfg = make_condition_config(corruption, severity, img_dir, args.batch, args.workers)
        log_path = LOG_DIR / f"{MODEL_NAME}_{corruption}_sev{severity}.log"

        cmd = [
            sys.executable,
            "tools/train.py",
            "-c",
            str(generated_cfg),
            "-r",
            str(WEIGHT),
            "--test-only",
        ]

        print("=" * 100)
        print(f"Running DETR-s: corruption={corruption}, severity={severity}")
        print("Image dir:", img_dir)
        print("Config:", generated_cfg)
        print("Log:", log_path)

        proc = subprocess.run(
            cmd,
            cwd=str(RTDETR_DIR),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        log_path.write_text(proc.stdout, encoding="utf-8")
        print(proc.stdout)

        if proc.returncode != 0:
            print(f"[ERROR] DETR-s failed on {corruption} sev{severity}. Return code={proc.returncode}")
            if not args.skip_errors:
                raise RuntimeError(f"Failed on {corruption} sev{severity}. See log: {log_path}")
            continue

        map50, map50_95, ar100 = parse_metrics(proc.stdout)

        if math.isnan(map50) or math.isnan(map50_95):
            print(f"[ERROR] Could not parse AP metrics from log: {log_path}")
            if not args.skip_errors:
                raise RuntimeError(f"Metric parse failed: {log_path}")
            continue

        row = {
            "model": MODEL_NAME,
            "weight_path": str(WEIGHT),
            "corruption": corruption,
            "severity": severity,
            "yaml_path": str(generated_cfg),
            "mAP50": map50,
            "mAP50_95": map50_95,
            "P": math.nan,
            "R": math.nan,
            "Car_mAP50": math.nan,
            "Pedestrian_mAP50": math.nan,
            "Cyclist_mAP50": math.nan,
        }

        append_row(RAW_CSV, row)
        print(f"Saved row: mAP50={map50:.6f}, mAP50-95={map50_95:.6f}, AR100={ar100:.6f}")

    print("=" * 100)
    print("Done.")
    print("Raw CSV:", RAW_CSV)
    print("Logs:", LOG_DIR)


if __name__ == "__main__":
    main()
