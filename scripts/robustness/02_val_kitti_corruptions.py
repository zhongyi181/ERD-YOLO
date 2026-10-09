#!/usr/bin/env python3
"""Validate YOLO/RT-DETR models on KITTI corruptions."""

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

RAW_CSV_NAME = "robustness_raw.csv"
METRIC_NOTES_NAME = "robustness_raw_metric_notes.txt"
ERROR_LOG_NAME = "robustness_errors.log"


# These locations are placeholders for user-supplied checkpoints.
MODELS = {
    "yolov5": str(WEIGHTS_ROOT / 'yolov5/best.pt'),
    "yolov8": str(WEIGHTS_ROOT / 'yolov8/best.pt'),
    "yolov11": str(WEIGHTS_ROOT / 'yolov11/best.pt'),

    "yolo26n_baseline": str(WEIGHTS_ROOT / 'yolo26n_baseline/best.pt'),

    "ours_full": str(WEIGHTS_ROOT / 'ours_full/best.pt'),

    "yolo26s": str(WEIGHTS_ROOT / 'yolo26s/best.pt'),

    "yolo26l": str(WEIGHTS_ROOT / 'yolo26l/best.pt'),

    "detr_l": str(WEIGHTS_ROOT / 'detr_l/best.pt'),
}


MODEL_BACKENDS = {
    "yolov5": "yolo",
    "yolov8": "yolo",
    "yolov11": "yolo",
    "yolo26n_baseline": "yolo",
    "ours_full": "yolo",
    "yolo26s": "yolo",
    "yolo26l": "yolo",
    "detr_l": "rtdetr",
}


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
    parser = argparse.ArgumentParser(description="Validate KITTI corruption robustness.")
    parser.add_argument("--corrupt-root", type=Path, default=CORRUPT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--raw-csv", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=6)

    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODELS),
        default=list(MODELS),
        help="Selected models to validate.",
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=sorted({c[0] for c in CONDITIONS}),
        default=None,
        help="Optional condition filter, e.g. --conditions clean fog lowlight.",
    )

    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing raw CSV.")
    parser.add_argument("--resume", action="store_true", help="Append to an existing raw CSV and skip completed rows.")
    parser.add_argument("--skip-errors", action="store_true", help="Log failed validation jobs and continue.")
    return parser.parse_args()


def selected_conditions(condition_names: list[str] | None) -> list[tuple[str, int, str]]:
    if not condition_names:
        return CONDITIONS
    selected = set(condition_names)
    return [item for item in CONDITIONS if item[0] in selected]


def validate_inputs(corrupt_root: Path, selected_models: list[str], conditions: list[tuple[str, int, str]]) -> None:
    missing_yamls = [corrupt_root / rel for _, _, rel in conditions if not (corrupt_root / rel).is_file()]
    if missing_yamls:
        preview = "\n".join(str(path) for path in missing_yamls[:12])
        raise FileNotFoundError(
            "Missing corruption dataset yaml files. Run "
            "scripts/robustness/01_generate_kitti_corruptions.py first.\n"
            f"{preview}"
        )

    missing_weights = [MODELS[name] for name in selected_models if not Path(MODELS[name]).is_file()]
    if missing_weights:
        preview = "\n".join(missing_weights)
        raise FileNotFoundError(f"Missing model weights. Edit MODELS at the top of this script.\n{preview}")


def completed_keys(raw_csv: Path) -> set[tuple[str, str, int]]:
    if not raw_csv.is_file():
        return set()
    with raw_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return {
            (row["model"], row["corruption"], int(row["severity"]))
            for row in reader
            if row.get("model") and row.get("corruption") and row.get("severity")
        }


def prepare_csv(raw_csv: Path, overwrite: bool, resume: bool) -> tuple[str, set[tuple[str, str, int]]]:
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    if raw_csv.exists():
        if overwrite:
            raw_csv.unlink()
        elif resume:
            return "a", completed_keys(raw_csv)
        else:
            raise FileExistsError(f"Raw CSV already exists: {raw_csv}. Use --overwrite or --resume.")
    return "w", set()


def require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("numpy is required. Install project dependencies before running this script.") from exc
    return np


def load_model(model_name: str, weight_path: str) -> Any:
    backend = MODEL_BACKENDS.get(model_name, "yolo")

    if backend == "rtdetr":
        try:
            from ultralytics import RTDETR
            print(f"Loading RTDETR model: {model_name}")
            return RTDETR(weight_path)
        except Exception as exc:
            print(f"[WARN] RTDETR load failed for {model_name}: {exc}")
            print("[WARN] Fallback to YOLO loader.")
            from ultralytics import YOLO
            return YOLO(weight_path)

    from ultralytics import YOLO
    print(f"Loading YOLO model: {model_name}")
    return YOLO(weight_path)


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
    metrics = {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "P": float(box.mp),
        "R": float(box.mr),
        "Car_mAP50": class_values[0],
        "Pedestrian_mAP50": class_values[1],
        "Cyclist_mAP50": class_values[2],
    }
    return metrics, source


def write_metric_notes(notes_path: Path, sources: set[str]) -> None:
    sources_text = ", ".join(sorted(sources)) if sources else "none"
    notes_path.write_text(
        "Class columns Car_mAP50, Pedestrian_mAP50, and Cyclist_mAP50 are filled from "
        "results.box.ap50 when available. If results.box.ap50 is unavailable, the script "
        "falls back to results.box.maps, whose Ultralytics meaning is per-class mAP50-95.\n"
        f"Sources used in this run: {sources_text}\n",
        encoding="utf-8",
    )


def log_error(error_log: Path, model_name: str, corruption: str, severity: int, exc: BaseException) -> None:
    error_log.parent.mkdir(parents=True, exist_ok=True)
    with error_log.open("a", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"model={model_name}, corruption={corruption}, severity={severity}\n")
        f.write(str(exc) + "\n")
        f.write(traceback.format_exc())
        f.write("\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    raw_csv = args.raw_csv or output_dir / RAW_CSV_NAME
    metric_notes = output_dir / METRIC_NOTES_NAME
    error_log = output_dir / ERROR_LOG_NAME
    conditions = selected_conditions(args.conditions)

    validate_inputs(args.corrupt_root, args.models, conditions)

    mode, skip_keys = prepare_csv(raw_csv, overwrite=args.overwrite, resume=args.resume)
    write_header = mode == "w"
    metric_sources: set[str] = set()

    with raw_csv.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        for model_name in args.models:
            weight_path = MODELS[model_name]
            model = load_model(model_name, weight_path)

            for corruption, severity, yaml_rel in conditions:
                key = (model_name, corruption, severity)
                if key in skip_keys:
                    print(f"Skipping completed: model={model_name} corruption={corruption} severity={severity}")
                    continue

                data_yaml = args.corrupt_root / yaml_rel
                run_name = f"{model_name}_{corruption}_sev{severity}"
                print(f"Validating model={model_name} corruption={corruption} severity={severity}")

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
                        augment=False,
                        rect=False,
                        project=str(output_dir / "val_runs"),
                        name=run_name,
                        exist_ok=True,
                    )

                    metrics, metric_source = extract_metrics(results)
                    metric_sources.add(metric_source)

                    row = {
                        "model": model_name,
                        "weight_path": weight_path,
                        "corruption": corruption,
                        "severity": severity,
                        "yaml_path": str(data_yaml),
                        **metrics,
                    }
                    writer.writerow(row)
                    file.flush()

                    print(
                        f"  mAP50={metrics['mAP50']:.6f} "
                        f"mAP50-95={metrics['mAP50_95']:.6f} "
                        f"P={metrics['P']:.6f} R={metrics['R']:.6f}"
                    )

                except Exception as exc:
                    log_error(error_log, model_name, corruption, severity, exc)
                    print(f"[ERROR] Failed: model={model_name} corruption={corruption} severity={severity}")
                    print(f"[ERROR] See: {error_log}")
                    if not args.skip_errors:
                        raise

    write_metric_notes(metric_notes, metric_sources)
    print(f"Raw results: {raw_csv}")
    print(f"Metric notes: {metric_notes}")
    if error_log.exists():
        print(f"Error log: {error_log}")


if __name__ == "__main__":
    main()
