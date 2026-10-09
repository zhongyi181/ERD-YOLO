from __future__ import annotations

import argparse
import sys
from pathlib import Path
import os
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.cfg import get_cfg  # noqa: E402
from ultralytics.data import build_yolo_dataset  # noqa: E402
from ultralytics.data.utils import check_det_dataset  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils import DEFAULT_CFG, YAML  # noqa: E402
from ultralytics.utils.dsakd import (  # noqa: E402
    collect_top_level_shapes,
    feature_layers_from_detect,
    output_shape,
    selected_feature_shapes,
    spatial_candidates,
)
from ultralytics.utils.ogdkd import (  # noqa: E402
    OGDKDLoss,
    OGDKDSettings,
    count_class_instances_from_labels,
    effective_number_class_weights,
    resolve_class_weight_mode,
)
from ultralytics.utils.torch_utils import select_device, unwrap_model  # noqa: E402

# Portable path defaults; run from the repository root or set these environment variables.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data/kitti_3cls_yolo")).resolve()
WEIGHTS_ROOT = Path(os.environ.get("WEIGHTS_ROOT", "weights")).resolve()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "outputs")).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check YOLO26 CW-SW loss and optional teacher-guided OG-DKD setup.")
    parser.add_argument(
        "--model",
        "--student",
        dest="model",
        default="ultralytics/cfg/models/26/yolo26n.yaml",
        help="YOLO26n model YAML/checkpoint. --student is kept as a backward-compatible alias.",
    )
    parser.add_argument(
        "--teacher-runs",
        "--teacher",
        dest="teacher",
        default=None,
        help="Teacher checkpoint path. Required only when gap weighting or OG-DKD is enabled.",
    )
    parser.add_argument("--data", default=str(DATA_ROOT / 'kitti_3cls.yaml'), help="Dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square dummy input size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size used only for building the train dataset.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or GPU index such as 0.")
    parser.add_argument("--max-print", type=int, default=80, help="Maximum top-level layer shapes to print per model.")
    parser.add_argument("--class-weight-mode", choices=("none", "manual", "effective"), default="effective")
    parser.add_argument("--class-weights", nargs="+", default=["1.0", "1.25", "1.5"])
    parser.add_argument("--class-balance-beta", type=float, default=0.999)
    parser.add_argument("--class-weight-min", type=float, default=0.5)
    parser.add_argument("--class-weight-max", type=float, default=2.0)
    parser.add_argument("--use-class-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-size-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--beta-size", type=float, default=0.5)
    parser.add_argument("--size-weight-max", type=float, default=1.5)
    parser.add_argument("--use-gap-weight", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-ogdkd", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def parse_float_tuple(values: list[str] | str) -> tuple[float, ...]:
    if isinstance(values, str):
        values = [values]
    pieces: list[str] = []
    for value in values:
        pieces.extend(x.strip() for x in str(value).split(",") if x.strip())
    return tuple(float(x) for x in pieces)


def resolve_local_path(value: str) -> str:
    path = Path(value)
    if path.exists():
        return str(path)
    candidates = []
    if value.startswith("cfg/"):
        candidates.append(ROOT / "ultralytics" / value)
    if value.startswith("cfg/models/YOLO26/"):
        candidates.append(ROOT / "ultralytics" / "cfg" / "models" / "26" / Path(value).name)
    if value.startswith("cfg/models/26/"):
        candidates.append(ROOT / "ultralytics" / value)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return value


def map_wsl_mount_path(value: str) -> str:
    """Map '/mnt/x/...' to 'X:/...' when this check script runs under Windows Python."""
    if not sys.platform.startswith("win") or not value.startswith("/mnt/") or len(value) < 8:
        return value
    drive, sep = value[5], value[6]
    if sep != "/" or not drive.isalpha():
        return value
    return f"{drive.upper()}:/{value[7:]}"


def resolve_data_entry(root: Path, value: str) -> str:
    """Resolve one train/val/test path entry with Windows WSL-mount compatibility."""
    value = map_wsl_mount_path(value)
    path = Path(value)
    if path.is_absolute():
        return str(path.resolve())
    resolved = (root / value).resolve()
    if not resolved.exists() and value.startswith("../"):
        resolved = (root / value[3:]).resolve()
    return str(resolved)


def load_data_with_mapped_paths(data_path: str) -> dict[str, Any]:
    """Load a data YAML and map WSL mount paths for Windows-only check runs."""
    data = YAML.load(data_path, append_filename=True)
    if "train" not in data:
        raise SyntaxError(f"{data_path} 'train:' key missing.")
    if "val" not in data and "validation" in data:
        data["val"] = data.pop("validation")
    if "names" not in data and "nc" not in data:
        raise SyntaxError(f"{data_path} must define 'names' or 'nc'.")
    if "names" in data:
        data["nc"] = len(data["names"])
    else:
        data["names"] = [f"class_{i}" for i in range(data["nc"])]
    data["channels"] = data.get("channels", 3)

    root_value = str(data.get("path") or Path(data.get("yaml_file", data_path)).parent)
    root = Path(map_wsl_mount_path(root_value)).resolve()
    data["path"] = root
    for key in ("train", "val", "test", "minival"):
        if not data.get(key):
            continue
        if isinstance(data[key], str):
            data[key] = resolve_data_entry(root, data[key])
        else:
            data[key] = [resolve_data_entry(root, x) for x in data[key]]
    return data


def build_train_dataset_labels(data_path: str, imgsz: int, batch: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the train dataset and return checked data metadata plus original labels."""
    try:
        data = check_det_dataset(data_path)
    except (FileNotFoundError, PermissionError):
        if not sys.platform.startswith("win"):
            raise
        data = load_data_with_mapped_paths(data_path)
    cfg = get_cfg(
        DEFAULT_CFG,
        overrides={
            "imgsz": imgsz,
            "batch": batch,
            "task": "detect",
            "cache": False,
            "rect": False,
            "single_cls": False,
            "classes": None,
            "fraction": 1.0,
        },
    )
    dataset = build_yolo_dataset(cfg, data["train"], batch=batch, data=data, mode="train", rect=False, stride=32)
    return data, getattr(dataset, "labels", [])


def resolve_num_classes(data: dict[str, Any], model: torch.nn.Module | None = None) -> int:
    """Resolve nc from checked data metadata first, then model attributes."""
    nc = data.get("nc")
    if nc is None:
        names = data.get("names")
        if isinstance(names, dict):
            nc = len(names)
        elif isinstance(names, (list, tuple)):
            nc = len(names)
    if nc is None and model is not None:
        nc = getattr(unwrap_model(model), "nc", None)
    if nc is None or int(nc) <= 0:
        raise ValueError(f"Unable to resolve a valid number of classes, got nc={nc}.")
    return int(nc)


def load_student_model(model_path: str, data: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Load a student model the same way training does for YAML configs, including dataset nc."""
    suffix = Path(model_path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return DetectionModel(
            model_path,
            nc=int(data["nc"]),
            ch=int(data.get("channels", 3)),
            verbose=False,
        ).to(device).eval()
    return YOLO(model_path, task="detect").model.to(device).eval()


def resolve_class_weights(args: argparse.Namespace, class_counts: tuple[int, ...], nc: int) -> tuple[str, tuple[float, ...], bool]:
    """Resolve none/manual/effective class weights for the check script."""
    mode = resolve_class_weight_mode(args.class_weight_mode, args.use_class_weight)
    clipped = False
    if mode == "none":
        class_weights = tuple(1.0 for _ in range(nc))
    elif mode == "manual":
        class_weights = parse_float_tuple(args.class_weights)
        if len(class_weights) != nc:
            raise ValueError(
                f"--class-weight-mode manual expects {nc} class weights from the dataset/model, "
                f"got {len(class_weights)}: {class_weights}."
            )
    else:
        class_weights, clipped = effective_number_class_weights(
            class_counts,
            beta=args.class_balance_beta,
            weight_min=args.class_weight_min,
            weight_max=args.class_weight_max,
        )
    return mode, class_weights, clipped


def shape_to_str(shape: Any) -> str:
    if isinstance(shape, tuple):
        return str(tuple(int(x) for x in shape))
    if isinstance(shape, list):
        return "[" + ", ".join(shape_to_str(x) for x in shape) + "]"
    if isinstance(shape, dict):
        return "{" + ", ".join(f"{k}: {shape_to_str(v)}" for k, v in shape.items()) + "}"
    return str(shape)


def print_layer_shapes(name: str, model: torch.nn.Module, shapes: dict[int, Any], max_print: int) -> None:
    print(f"\n{name} top-level layer output shapes:")
    for printed, (index, shape) in enumerate(shapes.items()):
        if printed >= max_print:
            remaining = len(shapes) - max_print
            if remaining > 0:
                print(f"  ... {remaining} more layers omitted")
            break
        layer = unwrap_model(model).model[index]
        layer_type = layer.__class__.__name__
        print(f"  {index:>2} {layer_type:<24} {shape_to_str(shape)}")


def print_candidates(name: str, shapes: dict[int, Any]) -> None:
    print(f"\n{name} auto candidates for P3/P4/P5 spatial maps:")
    candidates = spatial_candidates(shapes, targets=((80, 80), (40, 40), (20, 20)))
    for hw in ((80, 80), (40, 40), (20, 20)):
        items = candidates.get(hw, [])
        if not items:
            print(f"  {hw}: none")
            continue
        pretty = ", ".join(f"{idx}:{tuple(shape)}" for idx, shape in items)
        print(f"  {hw}: {pretty}")


def print_recommended(name: str, model: torch.nn.Module, imgsz: int, device: torch.device) -> list[int]:
    layers = feature_layers_from_detect(model)
    shapes = selected_feature_shapes(model, layers, imgsz=imgsz, device=device)
    print(f"\n{name} recommended hook layers from Detect.f: {layers}")
    for index in layers:
        print(f"  layer {index}: {tuple(shapes[index])}")
    return layers


def print_forward_output(name: str, model: torch.nn.Module, imgsz: int, device: torch.device) -> None:
    module = unwrap_model(model).to(device).eval()
    x = torch.zeros(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        y = module(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(f"\n{name} dummy forward output:")
    print(f"  {shape_to_str(output_shape(y))}")


def main() -> None:
    args = parse_args()
    device = select_device(str(args.device))

    teacher_guided = args.use_gap_weight or args.use_ogdkd
    if teacher_guided and not args.teacher:
        raise ValueError("--teacher-runs/--teacher is required when --use-gap-weight or --use-ogdkd is enabled.")

    model_path = resolve_local_path(args.model)
    data, train_labels = build_train_dataset_labels(args.data, args.imgsz, args.batch)
    nc = resolve_num_classes(data)
    class_counts, invalid = count_class_instances_from_labels(train_labels, nc)
    class_weight_mode, class_weights, clipped = resolve_class_weights(args, class_counts, nc)
    settings = OGDKDSettings(
        class_weights=class_weights,
        class_weight_mode=class_weight_mode,
        class_counts=class_counts,
        class_balance_beta=args.class_balance_beta,
        class_weight_min=args.class_weight_min,
        class_weight_max=args.class_weight_max,
        use_class_weight=class_weight_mode != "none",
        use_size_weight=args.use_size_weight,
        use_gap_weight=args.use_gap_weight,
        use_ogdkd=args.use_ogdkd,
        beta_size=args.beta_size,
        size_weight_max=args.size_weight_max,
    )

    mode = "teacher-student guided loss" if teacher_guided else "YOLO26n improved detection loss only"
    print(f"Mode: {mode}")
    print(f"model: {model_path}")
    print(f"data: {args.data}")
    if teacher_guided:
        print(f"teacher: {args.teacher}")
    print(f"device: {device}")
    print(f"imgsz: {args.imgsz}")
    print(f"nc: {nc}")
    print(f"class_weight_mode: {class_weight_mode}")
    print(f"train class_counts: {list(class_counts)}")
    if invalid:
        print(f"invalid train labels ignored: {invalid}")
    if class_weight_mode == "effective":
        print(f"effective class weights: {[round(float(x), 6) for x in class_weights]}")
        if clipped:
            print(f"class weights clipped to [{args.class_weight_min}, {args.class_weight_max}]")
    elif class_weight_mode == "manual":
        print(f"manual class weights: {[round(float(x), 6) for x in class_weights]}")
    else:
        print("class weights disabled: using all ones")
    print(
        "size weight config: existing implementation, "
        f"use_size_weight={settings.use_size_weight}, beta_size={settings.beta_size}, "
        f"size_weight_max={settings.size_weight_max}"
    )
    print(f"teacher disabled: {not teacher_guided}")
    print(f"KD disabled: {not args.use_ogdkd}")
    print(f"settings: {settings}")

    model = load_student_model(model_path, data, device)
    teacher = None
    if teacher_guided:
        teacher = YOLO(args.teacher, task="detect").model.to(device).eval()
        teacher.requires_grad_(False)
        teacher.eval()

    model_shapes = collect_top_level_shapes(model, imgsz=args.imgsz, device=device)
    teacher_shapes = collect_top_level_shapes(teacher, imgsz=args.imgsz, device=device) if teacher is not None else {}

    print_layer_shapes("model", model, model_shapes, args.max_print)
    print_candidates("model", model_shapes)
    model_layers = print_recommended("model", model, args.imgsz, device)
    print_forward_output("model", model, args.imgsz, device)

    teacher_layers = []
    spatial_match = None
    if teacher is not None:
        print_layer_shapes("teacher", teacher, teacher_shapes, args.max_print)
        print_candidates("teacher", teacher_shapes)
        teacher_layers = print_recommended("teacher", teacher, args.imgsz, device)
        print_forward_output("teacher", teacher, args.imgsz, device)
        model_hw = [
            selected_feature_shapes(model, model_layers, imgsz=args.imgsz, device=device)[i][-2:] for i in model_layers
        ]
        teacher_hw = [
            selected_feature_shapes(teacher, teacher_layers, imgsz=args.imgsz, device=device)[i][-2:]
            for i in teacher_layers
        ]
        spatial_match = model_hw == teacher_hw

    criterion = OGDKDLoss(model, settings)
    print("\nCW-SW / OG-DKD checks:")
    print(f"  model end2end: {getattr(model, 'end2end', False)}")
    print(f"  criterion end2end: {criterion.end2end}")
    print(f"  custom criterion: {criterion.__class__.__name__}")
    print(f"  teacher required: {teacher_guided}")
    if teacher is not None:
        print(f"  teacher eval: {not teacher.training}")
        print(f"  teacher frozen: {all(not p.requires_grad for p in teacher.parameters())}")
        print(f"  hook spatial sizes match: {spatial_match}")
    print(f"  extra trainable OG-DKD params: {sum(p.numel() for p in criterion.trainable_parameters())}")
    if teacher_layers:
        print("\nSuggested teacher-guided hook arguments:")
        print(f"  --student-layers {','.join(map(str, model_layers))}")
        print(f"  --teacher-layers {','.join(map(str, teacher_layers))}")
    print("check_yolo26_ogdkd passed: True")


if __name__ == "__main__":
    main()
