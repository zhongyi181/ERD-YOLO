#!/usr/bin/env python3
"""Summarize KITTI corruption robustness CSV into paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
import os

# Portable path defaults; run from the repository root or set these environment variables.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data/kitti_3cls_yolo")).resolve()
WEIGHTS_ROOT = Path(os.environ.get("WEIGHTS_ROOT", "weights")).resolve()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "outputs")).resolve()


OUTPUT_DIR = OUTPUT_ROOT / "robustness_eval"
RAW_CSV = OUTPUT_DIR / "robustness_raw.csv"
SUMMARY_CSV = OUTPUT_DIR / "robustness_summary.csv"
SUMMARY_MD = OUTPUT_DIR / "robustness_summary.md"

MODEL_ORDER = ("baseline", "ours")
CORRUPTION_COLUMNS = [
    ("fog", "Fog_mAP50"),
    ("lowlight", "Lowlight_mAP50"),
    ("motion_blur", "MotionBlur_mAP50"),
    ("gaussian_noise", "GaussianNoise_mAP50"),
    ("contrast", "Contrast_mAP50"),
]
SUMMARY_COLUMNS = [
    "model",
    "Clean_mAP50",
    "Fog_mAP50",
    "Lowlight_mAP50",
    "MotionBlur_mAP50",
    "GaussianNoise_mAP50",
    "Contrast_mAP50",
    "Avg_Corrupt_mAP50",
    "Avg_Drop_Rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize KITTI corruption robustness results.")
    parser.add_argument("--raw-csv", type=Path, default=RAW_CSV)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--summary-md", type=Path, default=SUMMARY_MD)
    return parser.parse_args()


def load_rows(raw_csv: Path) -> list[dict[str, str]]:
    if not raw_csv.is_file():
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv}")
    with raw_csv.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise RuntimeError(f"Raw CSV has no rows: {raw_csv}")
    return rows


def mean(values: list[float]) -> float:
    if not values:
        raise RuntimeError("Cannot average an empty list of values.")
    return statistics.fmean(values)


def model_sort_key(model: str) -> tuple[int, str]:
    if model in MODEL_ORDER:
        return MODEL_ORDER.index(model), model
    return len(MODEL_ORDER), model


def summarize(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    rows_by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_model[row["model"]].append(row)

    summary_rows: list[dict[str, float | str]] = []
    for model in sorted(rows_by_model, key=model_sort_key):
        model_rows = rows_by_model[model]
        clean_rows = [row for row in model_rows if row["corruption"] == "clean"]
        if len(clean_rows) != 1:
            raise RuntimeError(f"Expected exactly one clean row for model={model}, found {len(clean_rows)}")

        clean_map50 = float(clean_rows[0]["mAP50"])
        if clean_map50 <= 0:
            raise RuntimeError(f"Clean mAP50 must be positive for drop-rate calculation: model={model}")

        summary: dict[str, float | str] = {
            "model": model,
            "Clean_mAP50": clean_map50,
        }

        corrupt_means: list[float] = []
        drop_means: list[float] = []
        for corruption, column in CORRUPTION_COLUMNS:
            corrupt_rows = [row for row in model_rows if row["corruption"] == corruption]
            if len(corrupt_rows) != 3:
                raise RuntimeError(
                    f"Expected 3 severity rows for model={model} corruption={corruption}, "
                    f"found {len(corrupt_rows)}"
                )
            maps = [float(row["mAP50"]) for row in corrupt_rows]
            drops = [(clean_map50 - value) / clean_map50 for value in maps]
            corruption_mean = mean(maps)
            summary[column] = corruption_mean
            corrupt_means.append(corruption_mean)
            drop_means.append(mean(drops))

        summary["Avg_Corrupt_mAP50"] = mean(corrupt_means)
        summary["Avg_Drop_Rate"] = mean(drop_means)
        summary_rows.append(summary)

    return summary_rows


def fmt(value: float | str, digits: int = 6) -> str:
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def write_summary_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in SUMMARY_COLUMNS})


def write_summary_md(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Model | Clean | Fog | Low-light | Motion blur | Noise | Contrast | Avg Corrupt | Avg Drop Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['model']} | "
            f"{fmt(row['Clean_mAP50'])} | "
            f"{fmt(row['Fog_mAP50'])} | "
            f"{fmt(row['Lowlight_mAP50'])} | "
            f"{fmt(row['MotionBlur_mAP50'])} | "
            f"{fmt(row['GaussianNoise_mAP50'])} | "
            f"{fmt(row['Contrast_mAP50'])} | "
            f"{fmt(row['Avg_Corrupt_mAP50'])} | "
            f"{fmt(row['Avg_Drop_Rate'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_required_metrics(rows: list[dict[str, float | str]]) -> None:
    by_model = {str(row["model"]): row for row in rows}
    for model in MODEL_ORDER:
        if model in by_model:
            print(f"{model} clean mAP50: {fmt(by_model[model]['Clean_mAP50'])}")
    for model in MODEL_ORDER:
        if model in by_model:
            print(f"{model} Avg_Corrupt_mAP50: {fmt(by_model[model]['Avg_Corrupt_mAP50'])}")
    for model in MODEL_ORDER:
        if model in by_model:
            print(f"{model} Avg_Drop_Rate: {fmt(by_model[model]['Avg_Drop_Rate'])}")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.raw_csv)
    summary_rows = summarize(rows)
    write_summary_csv(args.summary_csv, summary_rows)
    write_summary_md(args.summary_md, summary_rows)
    print(f"Summary CSV: {args.summary_csv}")
    print(f"Summary Markdown: {args.summary_md}")
    print_required_metrics(summary_rows)


if __name__ == "__main__":
    main()
