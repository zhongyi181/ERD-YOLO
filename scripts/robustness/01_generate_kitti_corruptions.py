#!/usr/bin/env python3
"""Generate offline KITTI val corruptions for robustness validation.

This script only corrupts validation images. Labels are linked or copied
unchanged because all corruptions are pixel-level transforms.
"""

from __future__ import annotations

import argparse
import random
import shutil
import zlib
from pathlib import Path
import os
from typing import Any

# Portable path defaults; run from the repository root or set these environment variables.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data/kitti_3cls_yolo")).resolve()
WEIGHTS_ROOT = Path(os.environ.get("WEIGHTS_ROOT", "weights")).resolve()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "outputs")).resolve()


DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data/kitti_3cls_yolo")).resolve()
SOURCE_DATA_YAML = DATA_ROOT / "kitti_3cls.yaml"
VAL_IMAGES = DATA_ROOT / "images" / "val"
VAL_LABELS = DATA_ROOT / "labels" / "val"
OUTPUT_ROOT = DATA_ROOT / "kitti_corrupt"

NAMES = {
    0: "Car",
    1: "Pedestrian",
    2: "Cyclist",
}

SEED = 42
SEVERITIES = (1, 3, 5)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

IMAGECORRUPTIONS = {
    "fog": "fog",
    "motion_blur": "motion_blur",
    "gaussian_noise": "gaussian_noise",
    "contrast": "contrast",
}
LOWLIGHT_GAMMA = {
    1: 1.5,
    3: 2.5,
    5: 3.5,
}
ALL_CORRUPTIONS = tuple(IMAGECORRUPTIONS) + ("lowlight",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate KITTI val corruptions for robustness validation.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="KITTI YOLO dataset root.")
    parser.add_argument("--val-images", type=Path, default=None, help="Validation image directory.")
    parser.add_argument("--val-labels", type=Path, default=None, help="Validation label directory.")
    parser.add_argument("--output-root", type=Path, default=None, help="Corrupted dataset output root.")
    parser.add_argument("--seed", type=int, default=SEED, help="Base random seed.")
    parser.add_argument(
        "--corruptions",
        nargs="+",
        choices=ALL_CORRUPTIONS,
        default=list(ALL_CORRUPTIONS),
        help="Corruptions to generate.",
    )
    parser.add_argument(
        "--severities",
        nargs="+",
        type=int,
        choices=SEVERITIES,
        default=list(SEVERITIES),
        help="Severity levels to generate.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate images that already exist.")
    parser.add_argument(
        "--copy-labels",
        action="store_true",
        help="Copy label txt files instead of trying a directory symlink first.",
    )
    return parser.parse_args()


def list_images(images_dir: Path) -> list[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {images_dir}")
    images = sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No validation images found in: {images_dir}")
    return images


def list_labels(labels_dir: Path) -> list[Path]:
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Validation label directory not found: {labels_dir}")
    return sorted(p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


def count_images(images_dir: Path) -> int:
    return sum(1 for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def count_labels(labels_dir: Path) -> int:
    return sum(1 for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


def verify_source_labels(images: list[Path], labels_dir: Path) -> None:
    labels = list_labels(labels_dir)
    if len(labels) != len(images):
        raise RuntimeError(f"Source label count mismatch: images={len(images)} labels={len(labels)}")

    missing = [image.name for image in images if not (labels_dir / f"{image.stem}.txt").is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"Missing source labels for {len(missing)} images, first examples: {preview}")


def write_dataset_yaml(yaml_path: Path, dataset_path: Path, train: str = "images", val: str = "images") -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    names = "\n".join(f"  {idx}: {name}" for idx, name in NAMES.items())
    yaml_path.write_text(
        f"path: {dataset_path.as_posix()}\n"
        f"train: {train}\n"
        f"val: {val}\n\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )


def ensure_dir_symlink(source: Path, target: Path, overwrite: bool = False) -> bool:
    source = source.resolve()
    if target.exists() or target.is_symlink():
        if target.is_symlink():
            if target.resolve() == source:
                return True
            if overwrite:
                target.unlink()
            else:
                return False
        else:
            return False

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source, target_is_directory=True)
        return True
    except OSError as exc:
        print(f"Symlink failed for {target} -> {source}: {exc}")
        return False


def copy_missing_labels(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for label in list_labels(source):
        dst = target / label.name
        if not dst.exists():
            shutil.copy2(label, dst)


def ensure_labels(source: Path, target: Path, expected_count: int, copy_labels: bool, overwrite: bool = False) -> None:
    if target.exists() and not target.is_dir():
        raise RuntimeError(f"Label target exists but is not a directory: {target}")

    if copy_labels and target.is_symlink():
        target.unlink()
    elif target.is_symlink() and target.resolve() != source.resolve():
        if overwrite:
            target.unlink()
        else:
            raise RuntimeError(f"Label target is a symlink to a different directory: {target} -> {target.resolve()}")

    if not copy_labels and ensure_dir_symlink(source, target, overwrite=overwrite):
        pass
    else:
        copy_missing_labels(source, target)

    label_count = count_labels(target)
    if label_count != expected_count:
        raise RuntimeError(f"Label count mismatch for {target}: expected={expected_count} actual={label_count}")


def stable_seed(base_seed: int, corruption: str, severity: int, filename: str) -> int:
    key = f"{base_seed}:{corruption}:{severity}:{filename}".encode("utf-8")
    return zlib.crc32(key) & 0xFFFFFFFF


def require_numpy():
    try:
        import numpy as np
        # Compatibility patch: imagecorruptions still uses np.float_,
        # while NumPy >= 2.0 removed this alias.
        if not hasattr(np, "float_"):
            np.float_ = np.float64

    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("numpy is required. Install project dependencies before running this script.") from exc
    return np


def require_pil_image():
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Pillow is required. Install project dependencies before running this script.") from exc
    return Image


def read_rgb(path: Path) -> Any:
    np = require_numpy()
    Image = require_pil_image()
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_rgb(array: Any, path: Path) -> None:
    np = require_numpy()
    Image = require_pil_image()
    array = np.clip(array, 0, 255).astype(np.uint8, copy=False)
    Image.fromarray(array).save(path)


def apply_lowlight(image: Any, severity: int) -> Any:
    np = require_numpy()
    gamma = LOWLIGHT_GAMMA[severity]
    dark = 255.0 * (image.astype(np.float32) / 255.0) ** gamma
    return np.clip(dark, 0, 255).astype(np.uint8)


def get_corrupt_function(selected_corruptions: list[str]):
    if all(corruption == "lowlight" for corruption in selected_corruptions):
        return None
    try:
        from imagecorruptions import corrupt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "imagecorruptions is required for fog/motion_blur/gaussian_noise/contrast. "
            "Install it with: pip install imagecorruptions"
        ) from exc
    return corrupt


def corrupt_image(image: Any, corruption: str, severity: int, corrupt_fn) -> Any:
    if corruption == "lowlight":
        return apply_lowlight(image, severity)
    if corrupt_fn is None:
        raise RuntimeError(f"Missing imagecorruptions backend for: {corruption}")
    return corrupt_fn(image, corruption_name=IMAGECORRUPTIONS[corruption], severity=severity)


def generate_condition(
    images: list[Path],
    corruption: str,
    severity: int,
    condition_dir: Path,
    overwrite: bool,
    seed: int,
    corrupt_fn,
) -> None:
    np = require_numpy()
    image_dir = condition_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(images, start=1):
        target = image_dir / source.name
        if target.is_file() and not overwrite:
            continue

        per_image_seed = stable_seed(seed, corruption, severity, source.name)
        random.seed(per_image_seed)
        np.random.seed(per_image_seed)

        image = read_rgb(source)
        corrupted = corrupt_image(image, corruption, severity, corrupt_fn)
        if corrupted.shape != image.shape:
            raise RuntimeError(
                f"Shape changed for {source.name}: original={image.shape} corrupted={corrupted.shape}"
            )
        save_rgb(corrupted, target)

        if index % 100 == 0 or index == len(images):
            print(f"  {corruption} sev{severity}: {index}/{len(images)}")

    image_count = count_images(image_dir)
    if image_count != len(images):
        raise RuntimeError(f"Image count mismatch for {image_dir}: expected={len(images)} actual={image_count}")


def prepare_clean_yaml(
    output_root: Path,
    data_root: Path,
    val_images: Path,
    val_labels: Path,
    expected_count: int,
    copy_labels: bool,
    overwrite: bool,
) -> Path:
    clean_dir = output_root / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_images = clean_dir / "images"
    clean_labels = clean_dir / "labels"
    clean_yaml = clean_dir / "kitti_clean.yaml"

    images_ready = ensure_dir_symlink(val_images, clean_images, overwrite=overwrite)
    if images_ready:
        ensure_labels(val_labels, clean_labels, expected_count, copy_labels=copy_labels, overwrite=overwrite)
        write_dataset_yaml(clean_yaml, clean_dir)
    else:
        # Avoid copying the clean image set; point the clean yaml at the original val split.
        write_dataset_yaml(clean_yaml, data_root, train="images/val", val="images/val")
        print(f"Clean image symlink was not used; {clean_yaml} points to the original val split.")

    return clean_yaml


def verify_condition(condition_dir: Path, expected_count: int) -> None:
    image_count = count_images(condition_dir / "images")
    label_count = count_labels(condition_dir / "labels")
    if image_count != expected_count or label_count != expected_count:
        raise RuntimeError(
            f"Condition count mismatch for {condition_dir}: "
            f"expected={expected_count} images={image_count} labels={label_count}"
        )


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    val_images = args.val_images or data_root / "images" / "val"
    val_labels = args.val_labels or data_root / "labels" / "val"
    output_root = args.output_root or data_root / "kitti_corrupt"

    images = list_images(val_images)
    verify_source_labels(images, val_labels)
    expected_count = len(images)

    print(f"Source data yaml: {SOURCE_DATA_YAML if data_root == DATA_ROOT else data_root / 'kitti_3cls.yaml'}")
    print(f"Val images: {val_images} ({expected_count})")
    print(f"Val labels: {val_labels} ({count_labels(val_labels)})")
    print(f"Output root: {output_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    clean_yaml = prepare_clean_yaml(
        output_root,
        data_root,
        val_images,
        val_labels,
        expected_count,
        copy_labels=args.copy_labels,
        overwrite=args.overwrite,
    )
    generated_yamls = [clean_yaml]

    corrupt_fn = get_corrupt_function(list(args.corruptions))
    for corruption in args.corruptions:
        for severity in args.severities:
            condition_dir = output_root / corruption / f"sev{severity}"
            print(f"Generating {condition_dir}")
            ensure_labels(
                val_labels,
                condition_dir / "labels",
                expected_count,
                copy_labels=args.copy_labels,
                overwrite=args.overwrite,
            )
            generate_condition(
                images,
                corruption,
                severity,
                condition_dir,
                overwrite=args.overwrite,
                seed=args.seed,
                corrupt_fn=corrupt_fn,
            )
            verify_condition(condition_dir, expected_count)

            yaml_path = condition_dir / f"kitti_{corruption}_sev{severity}.yaml"
            write_dataset_yaml(yaml_path, condition_dir)
            generated_yamls.append(yaml_path)

    print("\nGenerated dataset yamls:")
    for yaml_path in generated_yamls:
        print(yaml_path)


if __name__ == "__main__":
    main()
