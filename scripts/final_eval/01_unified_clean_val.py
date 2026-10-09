#!/usr/bin/env python3
"""Run final unified clean validation for KITTI 3-class paper tables."""

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


OUTPUT_DIR = OUTPUT_ROOT / "final_unified_val"
CSV_PATH = OUTPUT_DIR / "unified_clean_val.csv"
MD_PATH = OUTPUT_DIR / "unified_clean_val.md"
PATH_CHECK_CSV = OUTPUT_DIR / "model_path_check.csv"
CLEAN_YAML = Path(str(DATA_ROOT / 'kitti_corrupt/clean/kitti_clean.yaml'))
ORIGINAL_VAL_IMAGES = Path(str(DATA_ROOT / 'images/val'))

# These locations are placeholders for user-supplied checkpoints and optional args files.
MODELS: dict[str, dict[str, str]] = {
    "yolo26n_baseline": {
        "model_name": "yolo26n_baseline",
        "best_pt_path": str(WEIGHTS_ROOT / 'yolo26n_baseline/best.pt'),
        "args_yaml_path": str(OUTPUT_ROOT / 'yolo26n_baseline/args.yaml'),
        "note": "User-supplied YOLO26n baseline checkpoint; provenance must be verified.",
    },
    "repP3": {
        "model_name": "repP3",
        "best_pt_path": str(WEIGHTS_ROOT / 'repP3/best.pt'),
        "args_yaml_path": str(OUTPUT_ROOT / 'repP3/args.yaml'),
        "note": "User-supplied RepP3/SR-P3 checkpoint; provenance must be verified.",
    },
    "autoCBSW": {
        "model_name": "autoCBSW",
        "best_pt_path": str(WEIGHTS_ROOT / 'autoCBSW/best.pt'),
        "args_yaml_path": str(OUTPUT_ROOT / 'autoCBSW/args.yaml'),
        "note": "User-supplied Auto-CBSW/CWSW-Lite checkpoint; provenance must be verified.",
    },
    "ogdkd": {
        "model_name": "ogdkd",
        "best_pt_path": str(WEIGHTS_ROOT / 'ogdkd/best.pt'),
        "args_yaml_path": str(OUTPUT_ROOT / 'ogdkd/args.yaml'),
        "note": "User-supplied OG-DKD checkpoint; provenance must be verified.",
    },
    "repP3_autoCBSW": {
        "model_name": "repP3_autoCBSW",
        "best_pt_path": str(WEIGHTS_ROOT / 'repP3_autoCBSW/best.pt'),
        "args_yaml_path": str(OUTPUT_ROOT / 'repP3_autoCBSW/args.yaml'),
        "note": "User-supplied RepP3 + Auto-CBSW checkpoint; provenance must be verified.",
    },
    "repP3_ogdkd": {
        "model_name": "repP3_ogdkd",
        "best_pt_path": str(WEIGHTS_ROOT / 'repP3_ogdkd/best.pt'),
        "args_yaml_path": str(OUTPUT_ROOT / 'repP3_ogdkd/args.yaml'),
        "note": "User-supplied RepP3 + OG-DKD checkpoint; provenance must be verified.",
    },
    "autoCBSW_ogdkd": {
        "model_name": "autoCBSW_ogdkd",
        "best_pt_path": str(WEIGHTS_ROOT / 'autoCBSW_ogdkd/best.pt'),
        "args_yaml_path": str(OUTPUT_ROOT / 'autoCBSW_ogdkd/args.yaml'),
        "note": "User-supplied Auto-CBSW + OG-DKD checkpoint; provenance must be verified.",
    },
    "ours_1_2_3": {
        "model_name": "ours_1_2_3",
        "best_pt_path": str(WEIGHTS_ROOT / 'ours_1_2_3/best.pt'),
        "args_yaml_path": str(OUTPUT_ROOT / 'ours_1_2_3/args.yaml'),
        "note": "User-supplied RepP3 + Auto-CBSW + OG-DKD checkpoint; provenance must be verified.",
    },
    "yolo26s_baseline": {
        "model_name": "yolo26s_baseline",
        "best_pt_path": str(WEIGHTS_ROOT / 'yolo26s_baseline/best.pt'),
        "args_yaml_path": str(OUTPUT_ROOT / 'yolo26s_baseline/args.yaml'),
        "note": "User-supplied YOLO26s baseline checkpoint; provenance must be verified.",
    },
    "yolo26l_teacher": {
        "model_name": "yolo26l_teacher",
        "best_pt_path": str(WEIGHTS_ROOT / 'yolo26l_teacher/best.pt'),
        "args_yaml_path": str(OUTPUT_ROOT / 'yolo26l_teacher/args.yaml'),
        "note": "User-supplied YOLO26l teacher checkpoint; provenance must be verified.",
    },
}

MODEL_ORDER = list(MODELS)

CSV_COLUMNS = [
    "model_name",
    "best_pt_path",
    "data_yaml",
    "imgsz",
    "batch",
    "split",
    "rect",
    "augment",
    "mAP50",
    "mAP50_95",
    "P",
    "R",
    "Params",
    "GFLOPs",
    "status",
    "note",
]

PATH_CHECK_COLUMNS = [
    "model_name",
    "best_pt_path",
    "args_yaml_path",
    "exists_best_pt",
    "exists_args_yaml",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified clean validation for final KITTI 3-class tables.")
    parser.add_argument("--data", type=Path, default=CLEAN_YAML, help="Clean validation yaml.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--split", default="val")
    parser.add_argument("--resume", action="store_true", help="Skip models already validated with this exact protocol.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing unified CSV/Markdown outputs.")
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=MODEL_ORDER, help="Optional subset to run.")
    return parser.parse_args()


def require_yolo():
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Run with the project Python environment that has ultralytics installed.") from exc
    return YOLO


def require_yaml():
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required to inspect the clean data yaml.") from exc
    return yaml


def bool_csv(value: bool) -> str:
    return "False" if value is False else "True"


def format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "data": str(args.data),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "split": args.split,
        "workers": args.workers,
        "save_json": False,
        "save_txt": False,
        "save_conf": False,
        "plots": False,
        "verbose": False,
        "augment": False,
        "rect": False,
    }


def existing_completed(csv_path: Path, args: argparse.Namespace) -> set[str]:
    if not csv_path.is_file():
        return set()
    expected = {
        "data_yaml": str(args.data),
        "imgsz": str(args.imgsz),
        "batch": str(args.batch),
        "split": args.split,
        "rect": bool_csv(False),
        "augment": bool_csv(False),
    }
    completed: set[str] = set()
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("status") not in {"ok", "missing"}:
                continue
            if all(str(row.get(key, "")) == value for key, value in expected.items()):
                completed.add(row.get("model_name", ""))
    completed.discard("")
    return completed


def write_path_check(path_check_csv: Path, selected_models: list[str]) -> list[str]:
    path_check_csv.parent.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    with path_check_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=PATH_CHECK_COLUMNS)
        writer.writeheader()
        for model_key in selected_models:
            item = MODELS[model_key]
            best_pt = Path(item["best_pt_path"])
            args_yaml = Path(item["args_yaml_path"])
            exists_best = best_pt.is_file()
            exists_args = args_yaml.is_file()
            if not exists_best:
                missing.append(model_key)
            writer.writerow(
                {
                    "model_name": item["model_name"],
                    "best_pt_path": str(best_pt),
                    "args_yaml_path": str(args_yaml),
                    "exists_best_pt": exists_best,
                    "exists_args_yaml": exists_args,
                }
            )
    return missing


def prepare_outputs(csv_path: Path, md_path: Path, overwrite: bool, resume: bool) -> tuple[str, bool]:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in (csv_path, md_path):
            if path.exists():
                path.unlink()
    elif csv_path.exists() and not resume:
        raise FileExistsError(f"{csv_path} already exists. Use --resume or --overwrite.")

    mode = "a" if resume and csv_path.exists() else "w"
    write_header = mode == "w"
    return mode, write_header


def empty_metric_row(
    model_key: str,
    args: argparse.Namespace,
    status: str,
    note: str,
) -> dict[str, Any]:
    item = MODELS[model_key]
    return {
        "model_name": item["model_name"],
        "best_pt_path": item["best_pt_path"],
        "data_yaml": str(args.data),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "split": args.split,
        "rect": bool_csv(False),
        "augment": bool_csv(False),
        "mAP50": "",
        "mAP50_95": "",
        "P": "",
        "R": "",
        "Params": "",
        "GFLOPs": "",
        "status": status,
        "note": note,
    }


def extract_metrics(results: Any) -> dict[str, float]:
    box = results.box
    return {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "P": float(box.mp),
        "R": float(box.mr),
    }


def model_complexity(model: Any, imgsz: int) -> tuple[int | str, float | str]:
    try:
        from ultralytics.utils.torch_utils import get_flops, get_num_params

        return int(get_num_params(model.model)), float(get_flops(model.model, imgsz=imgsz))
    except Exception:
        try:
            info = model.info(verbose=True, imgsz=imgsz)
            if info and len(info) >= 4:
                return int(info[1]), float(info[3])
        except Exception:
            pass
    return "", ""


def validate_one(model_key: str, args: argparse.Namespace, yolo_cls: Any) -> dict[str, Any]:
    item = MODELS[model_key]
    best_pt = Path(item["best_pt_path"])
    if not best_pt.is_file():
        return empty_metric_row(model_key, args, "missing", f"missing best.pt: {best_pt}")

    model = yolo_cls(str(best_pt))
    params, gflops = model_complexity(model, args.imgsz)
    results = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split=args.split,
        workers=args.workers,
        save_json=False,
        save_txt=False,
        save_conf=False,
        plots=False,
        verbose=False,
        augment=False,
        rect=False,
        project=str(args.output_dir / "val_runs"),
        name=model_key,
        exist_ok=True,
    )
    metrics = extract_metrics(results)
    return {
        **empty_metric_row(model_key, args, "ok", item.get("note", "")),
        "mAP50": format_float(metrics["mAP50"]),
        "mAP50_95": format_float(metrics["mAP50_95"]),
        "P": format_float(metrics["P"]),
        "R": format_float(metrics["R"]),
        "Params": params,
        "GFLOPs": format_float(gflops, digits=3),
    }


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.is_file():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def clean_images_source(data_yaml: Path) -> str:
    if not data_yaml.is_file():
        return "不确定"
    yaml = require_yaml()
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = Path(data.get("path") or data_yaml.parent)
    val_value = data.get("val")
    if not val_value:
        return "不确定"
    val_path = Path(val_value)
    if not val_path.is_absolute():
        val_path = root / val_path
    try:
        if val_path.resolve() == ORIGINAL_VAL_IMAGES.resolve():
            return "原始 val 图像"
    except FileNotFoundError:
        return "不确定"
    if val_path.exists() and "kitti_corrupt" in str(val_path):
        return "clean 副本"
    return "不确定"


def row_for_markdown(rows: list[dict[str, str]], model_key: str) -> list[str]:
    model_name = MODELS[model_key]["model_name"]
    candidates = [row for row in rows if row.get("model_name") == model_name]
    if not candidates:
        return [model_name, "", "", "", "", "", ""]
    row = candidates[-1]
    if row.get("status") == "missing":
        return [model_name, "MISSING", "MISSING", "MISSING", "MISSING", "MISSING", "MISSING"]
    if row.get("status") != "ok":
        return [model_name, "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR"]
    return [
        model_name,
        row.get("mAP50", ""),
        row.get("mAP50_95", ""),
        row.get("P", ""),
        row.get("R", ""),
        row.get("Params", ""),
        row.get("GFLOPs", ""),
    ]


def build_markdown(rows: list[dict[str, str]], args: argparse.Namespace, source: str) -> str:
    lines = [
        f"clean_yaml_used = {args.data}",
        f"clean_images_source = {source}",
        (
            "val_params = "
            f"imgsz={args.imgsz}, batch={args.batch}, device={args.device}, split={args.split}, "
            f"workers={args.workers}, save_json=False, save_txt=False, save_conf=False, "
            "plots=False, verbose=False, augment=False, rect=False"
        ),
        "",
        "| Model | mAP50 | mAP50-95 | P | R | Params | GFLOPs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_key in MODEL_ORDER:
        model, map50, map5095, precision, recall, params, gflops = row_for_markdown(rows, model_key)
        lines.append(f"| {model} | {map50} | {map5095} | {precision} | {recall} | {params} | {gflops} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / CSV_PATH.name
    md_path = args.output_dir / MD_PATH.name
    path_check_csv = args.output_dir / PATH_CHECK_CSV.name
    selected_models = list(args.models)

    missing = write_path_check(path_check_csv, MODEL_ORDER)
    mode, write_header = prepare_outputs(csv_path, md_path, overwrite=args.overwrite, resume=args.resume)
    completed = existing_completed(csv_path, args) if args.resume else set()

    print(f"data_yaml: {args.data}")
    print("val_params:")
    for key, value in protocol(args).items():
        print(f"  {key}: {value}")
    print(f"model_path_check_csv: {path_check_csv}")

    yolo_cls = require_yolo()
    success_count = 0
    missing_models: list[str] = []

    with csv_path.open(mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()

        for model_key in selected_models:
            if model_key in completed:
                print(f"Skipping completed: {model_key}")
                continue

            print(f"Validating: {model_key}")
            try:
                row = validate_one(model_key, args, yolo_cls)
            except Exception as exc:
                note = "".join(traceback.format_exception_only(type(exc), exc)).strip().replace("\n", " ")
                row = empty_metric_row(model_key, args, "error", note)
                print(f"  ERROR: {note}")
            writer.writerow(row)
            file.flush()

            if row["status"] == "ok":
                success_count += 1
                print(
                    f"  OK mAP50={row['mAP50']} mAP50-95={row['mAP50_95']} "
                    f"P={row['P']} R={row['R']} Params={row['Params']} GFLOPs={row['GFLOPs']}"
                )
            elif row["status"] == "missing":
                missing_models.append(model_key)
                print(f"  MISSING {row['note']}")

    rows = read_rows(csv_path)
    total_success = sum(1 for row in rows if row.get("status") == "ok")
    all_missing = sorted({*missing, *missing_models})
    source = clean_images_source(args.data)
    md_text = build_markdown(rows, args, source)
    md_path.write_text(md_text, encoding="utf-8")

    print(f"clean_yaml_used = {args.data}")
    print(f"clean_images_source = {source}")
    print(f"successful_models = {total_success}")
    print("missing_models = " + (", ".join(all_missing) if all_missing else "none"))
    print(f"unified_clean_val_csv: {csv_path}")
    print(f"unified_clean_val_md: {md_path}")
    print("\nunified_clean_val.md:")
    print(md_text)


if __name__ == "__main__":
    main()
