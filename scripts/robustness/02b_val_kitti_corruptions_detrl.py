#!/usr/bin/env python3
"""Validate DETR-l / RT-DETR model on KITTI corruptions and append to robustness_raw.csv."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import traceback
from pathlib import Path
import os
from typing import Any

# Portable path defaults; run from the repository root or set these environment variables.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data/kitti_3cls_yolo")).resolve()
WEIGHTS_ROOT = Path(os.environ.get("WEIGHTS_ROOT", "weights")).resolve()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "outputs")).resolve()


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CORRUPT_ROOT = Path(str(DATA_ROOT / 'kitti_corrupt'))
OUTPUT_DIR = OUTPUT_ROOT / "robustness_eval"
RAW_CSV = OUTPUT_DIR / "robustness_raw.csv"
ERROR_LOG = OUTPUT_DIR / "robustness_detrl_errors.log"
METRIC_NOTES = OUTPUT_DIR / "robustness_detrl_metric_notes.txt"

MODEL_NAME = "detr_l"
# Placeholder for a user-supplied checkpoint.
WEIGHT_PATH = str(WEIGHTS_ROOT / 'detr_l/best.pt')

CONDITIONS = [
    ("clean", 0, "clean/kitti_clean.yaml"),

    ("fog", 1, "fog/sev1/kitti_fog_sev1.yaml"),
    ("fog", 3, "fog/sev3/kitti_fog_sev3.yaml"),
    ("fog", 5, "fog/sev5/kitti_fog_sev5.yaml"),

    ("lowlight", 1, "lowlight/sev1/kitti_lowlight_sev1.yaml"),
    ("lowlight", 3, "lowlight/sev3/kitti_lowlight_sev3.yaml"),
    ("lowlight", 5, "lowlight/sev5/kitti_lowlight_sev5.yaml"),

    ("motion_blur", 1, "motion_blur/sev1/kitti_motion_blur_sev1.yaml"),
    ("motion_blur", 3, "motion_blur/sev3/kitti_motion_blur_sev3.yaml"),
    ("motion_blur", 5, "motion_blur/sev5/kitti_motion_blur_sev5.yaml"),

    ("gaussian_noise", 1, "gaussian_noise/sev1/kitti_gaussian_noise_sev1.yaml"),
    ("gaussian_noise", 3, "gaussian_noise/sev3/kitti_gaussian_noise_sev3.yaml"),
    ("gaussian_noise", 5, "gaussian_noise/sev5/kitti_gaussian_noise_sev5.yaml"),

    ("contrast", 1, "contrast/sev1/kitti_contrast_sev1.yaml"),
    ("contrast", 3, "contrast/sev3/kitti_contrast_sev3.yaml"),
    ("contrast", 5, "contrast/sev5/kitti_contrast_sev5.yaml"),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DETR-l on KITTI corruptions.")
    parser.add_argument("--corrupt-root", type=Path, default=CORRUPT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--raw-csv", type=Path, default=RAW_CSV)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite-detrl", action="store_true", help="Remove existing DETR-l rows before running.")
    parser.add_argument("--skip-errors", action="store_true", help="Log failed jobs and continue.")
    return parser.parse_args()


def require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("numpy is required.") from exc
    return np


def validate_inputs(corrupt_root: Path) -> None:
    if not Path(WEIGHT_PATH).is_file():
        raise FileNotFoundError(f"Missing DETR-l weight: {WEIGHT_PATH}")

    missing_yamls = [corrupt_root / rel for _, _, rel in CONDITIONS if not (corrupt_root / rel).is_file()]
    if missing_yamls:
        preview = "\n".join(str(p) for p in missing_yamls[:12])
        raise FileNotFoundError(f"Missing corruption yaml files:\n{preview}")


def read_existing_rows(raw_csv: Path) -> list[dict[str, str]]:
    if not raw_csv.is_file():
        return []
    with raw_csv.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(raw_csv: Path, rows: list[dict[str, Any]]) -> None:
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def completed_keys(raw_csv: Path) -> set[tuple[str, str, int]]:
    rows = read_existing_rows(raw_csv)
    keys = set()
    for row in rows:
        if row.get("model") and row.get("corruption") and row.get("severity"):
            keys.add((row["model"], row["corruption"], int(row["severity"])))
    return keys


def remove_existing_detrl_rows(raw_csv: Path) -> None:
    rows = read_existing_rows(raw_csv)
    rows = [row for row in rows if row.get("model") != MODEL_NAME]
    write_rows(raw_csv, rows)


def class_ap50_values(box: Any, nc: int = 3) -> tuple[list[float], str]:
    np = require_numpy()
    values = [math.nan] * nc

    ap50 = np.asarray(getattr(box, "ap50", []), dtype=float)
    class_index = np.asarray(getattr(box, "ap_class_index", []), dtype=int)

    if ap50.size and class_index.size:
        for metric_index, class_id in enumerate(class_index):
            if 0 <= int(class_id) < nc and metric_index < ap50.size:
                values[int(class_id)] = float(ap50[metric_index])
        return values, "results.box.ap50"

    if ap50.size >= nc:
        return [float(x) for x in ap50[:nc]], "results.box.ap50"

    maps = np.asarray(getattr(box, "maps", []), dtype=float)
    if maps.size >= nc:
        return [float(x) for x in maps[:nc]], "results.box.maps_fallback"

    return values, "unavailable"


def extract_metrics(results: Any) -> tuple[dict[str, float], str]:
    box = results.box
    class_values, source = class_ap50_values(box)
    return {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "P": float(box.mp),
        "R": float(box.mr),
        "Car_mAP50": class_values[0],
        "Pedestrian_mAP50": class_values[1],
        "Cyclist_mAP50": class_values[2],
    }, source


def append_row(raw_csv: Path, row: dict[str, Any]) -> None:
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = raw_csv.is_file()

    with raw_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_error(model_name: str, corruption: str, severity: int, exc: BaseException) -> None:
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ERROR_LOG.open("a", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"model={model_name}, corruption={corruption}, severity={severity}\n")
        f.write(str(exc) + "\n")
        f.write(traceback.format_exc())
        f.write("\n")


def main() -> None:
    args = parse_args()
    raw_csv = args.raw_csv

    validate_inputs(args.corrupt_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite_detrl and raw_csv.is_file():
        remove_existing_detrl_rows(raw_csv)

    skip_keys = completed_keys(raw_csv)

    from ultralytics import RTDETR

    print(f"Loading DETR-l with RTDETR: {WEIGHT_PATH}")
    model = RTDETR(WEIGHT_PATH)

    metric_sources: set[str] = set()

    for corruption, severity, yaml_rel in CONDITIONS:
        key = (MODEL_NAME, corruption, severity)
        if key in skip_keys:
            print(f"Skipping completed: model={MODEL_NAME} corruption={corruption} severity={severity}")
            continue

        data_yaml = args.corrupt_root / yaml_rel
        run_name = f"{MODEL_NAME}_{corruption}_sev{severity}"

        print(f"Validating model={MODEL_NAME} corruption={corruption} severity={severity}")

        try:
            results = model.val(
                data=str(data_yaml),
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                split="val",
                workers=args.workers,
                save_json=False,
                save_txt=False,
                save_conf=False,
                plots=False,
                verbose=False,
                project=str(args.output_dir / "val_runs"),
                name=run_name,
                exist_ok=True,
            )

            metrics, metric_source = extract_metrics(results)
            metric_sources.add(metric_source)

            row = {
                "model": MODEL_NAME,
                "weight_path": WEIGHT_PATH,
                "corruption": corruption,
                "severity": severity,
                "yaml_path": str(data_yaml),
                **metrics,
            }

            append_row(raw_csv, row)

            print(
                f"  mAP50={metrics['mAP50']:.6f} "
                f"mAP50-95={metrics['mAP50_95']:.6f} "
                f"P={metrics['P']:.6f} "
                f"R={metrics['R']:.6f}"
            )

        except Exception as exc:
            log_error(MODEL_NAME, corruption, severity, exc)
            print(f"[ERROR] Failed: model={MODEL_NAME} corruption={corruption} severity={severity}")
            print(f"[ERROR] See: {ERROR_LOG}")
            if not args.skip_errors:
                raise

    METRIC_NOTES.write_text(
        "DETR-l class AP columns are extracted from results.box.ap50 when available. "
        "If ap50 is unavailable, results.box.maps is used as fallback.\n"
        f"Sources used: {', '.join(sorted(metric_sources)) if metric_sources else 'none'}\n",
        encoding="utf-8",
    )

    print(f"Done. Raw CSV: {raw_csv}")
    print(f"Metric notes: {METRIC_NOTES}")
    if ERROR_LOG.exists():
        print(f"Error log: {ERROR_LOG}")


if __name__ == "__main__":
    main()
