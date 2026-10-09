#!/usr/bin/env python3
"""Audit RepP3/RepConv ONNX export and TensorRT speed."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from ultralytics.utils.torch_utils import TORCH_2_4

# Portable path defaults; run from the repository root or set these environment variables.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data/kitti_3cls_yolo")).resolve()
WEIGHTS_ROOT = Path(os.environ.get("WEIGHTS_ROOT", "weights")).resolve()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "outputs")).resolve()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# These locations are placeholders for user-supplied checkpoints.
OURS_PT = WEIGHTS_ROOT / "ours_1_2_3/best.pt"
BASELINE_PT = WEIGHTS_ROOT / "yolo26n_baseline/best.pt"
OUT_DIR = OUTPUT_ROOT / "rep_p3_deploy_audit"


def load_yolo_model(pt: Path) -> torch.nn.Module:
    from ultralytics import YOLO

    yolo = YOLO(str(pt))
    model = yolo.model
    model.eval()
    model.float()
    return model


def torch_model_stats(pt: Path, imgsz: int) -> dict[str, Any]:
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    yolo = YOLO(str(pt))
    model = yolo.model
    return {
        "pt": str(pt),
        "yaml_file": model.yaml.get("yaml_file"),
        "layers": len(model.model),
        "params": int(get_num_params(model)),
        "gflops": float(get_flops(model, imgsz=imgsz)),
        "repconv": repconv_summary(model),
    }


def package_status() -> dict[str, Any]:
    import importlib.util

    packages = {}
    for name in ["onnx", "onnxslim", "onnxsim", "onnxscript", "tensorrt", "onnxruntime"]:
        spec = importlib.util.find_spec(name)
        packages[name] = spec.origin if spec else None
    return packages


def repconv_summary(model: torch.nn.Module) -> dict[str, Any]:
    from ultralytics.nn.modules import RepConv

    items = []
    for name, module in model.named_modules():
        if isinstance(module, RepConv):
            items.append(
                {
                    "name": name,
                    "has_conv": hasattr(module, "conv"),
                    "has_conv1": hasattr(module, "conv1"),
                    "has_conv2": hasattr(module, "conv2"),
                    "has_bn": hasattr(module, "bn") and module.bn is not None,
                }
            )
    return {"count": len(items), "items": items}


def prepare_for_export(model: torch.nn.Module, fuse: bool) -> torch.nn.Module:
    model = model.cpu().eval().float()
    for p in model.parameters():
        p.requires_grad_(False)

    if fuse:
        model = model.fuse(verbose=False)

    for module in model.modules():
        if module.__class__.__name__ in {"Detect", "RTDETRDecoder"}:
            module.dynamic = False
            module.export = True
            module.format = "onnx"
            if hasattr(module, "shape"):
                module.shape = None
    return model


def export_onnx(pt: Path, out: Path, fuse: bool, opset: int, imgsz: int) -> dict[str, Any]:
    model = load_yolo_model(pt)
    before = repconv_summary(model)
    model = prepare_for_export(model, fuse=fuse)
    after = repconv_summary(model)

    dummy = torch.zeros(1, model.yaml.get("channels", 3), imgsz, imgsz, dtype=torch.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        # Warm up Detect export flags and cached shapes.
        model(dummy)
        kwargs = {"dynamo": False} if TORCH_2_4 else {}
        torch.onnx.export(
            model,
            dummy,
            str(out),
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["output0"],
            dynamic_axes=None,
            **kwargs,
        )
    try_simplify(out)
    return {"onnx": str(out), "repconv_before": before, "repconv_after": after}


def try_simplify(path: Path) -> None:
    try:
        import onnx
        import onnxslim
    except Exception as exc:
        print(f"[warn] onnxslim unavailable, skip simplify for {path.name}: {exc}")
        return

    model = onnx.load(str(path))
    simplified = onnxslim.slim(model)
    if getattr(simplified, "ir_version", 0) > 10:
        simplified.ir_version = 10
    onnx.save(simplified, str(path))


def count_params_from_onnx(path: Path) -> int:
    import onnx

    model = onnx.load(str(path))
    total = 0
    for init in model.graph.initializer:
        n = 1
        for dim in init.dims:
            n *= int(dim)
        total += n
    return total


def onnx_stats(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path))
    ops = Counter(node.op_type for node in model.graph.node)
    return {
        "onnx": str(path),
        "opset": max(imp.version for imp in model.opset_import if imp.domain in {"", "ai.onnx"}),
        "ir_version": model.ir_version,
        "params": count_params_from_onnx(path),
        "nodes": len(model.graph.node),
        "Conv": ops.get("Conv", 0),
        "BatchNormalization": ops.get("BatchNormalization", 0),
        "Add": ops.get("Add", 0),
        "Concat": ops.get("Concat", 0),
    }


def find_trtexec() -> str | None:
    candidates = [
        shutil.which("trtexec"),
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/tensorrt/bin/trtexec",
        "/usr/local/cuda/bin/trtexec",
        "/opt/tensorrt/bin/trtexec",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    return None


def run_trtexec(onnx_path: Path, engine_path: Path, trtexec: str, extra_args: list[str]) -> dict[str, Any]:
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        *extra_args,
    ]
    print("[trtexec]", " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    log = proc.stdout
    log_path = engine_path.with_suffix(".trtexec.log")
    log_path.write_text(log, encoding="utf-8", errors="replace")
    result = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_s": elapsed,
        "log": str(log_path),
        "engine": str(engine_path),
    }
    result.update(parse_trtexec_log(log))
    return result


def parse_trtexec_log(log: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    patterns = {
        "throughput_fps": r"Throughput:\s*([0-9.]+)\s*qps",
        "mean_latency_ms": r"Latency:\s*min\s*=\s*[0-9.]+\s*ms,\s*max\s*=\s*[0-9.]+\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms",
        "mean_gpu_compute_ms": r"GPU Compute Time:\s*min\s*=\s*[0-9.]+\s*ms,\s*max\s*=\s*[0-9.]+\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, log)
        if m:
            out[key] = float(m.group(1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-trt", action="store_true")
    parser.add_argument(
        "--trtexec-args",
        nargs=argparse.REMAINDER,
        default=["--shapes=images:1x3x640x640", "--warmUp=500", "--duration=10", "--iterations=1000"],
        help="Arguments appended to trtexec after --fp16.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"python={sys.executable}")
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")

    paths = {
        "ours_train": args.out_dir / "ours_train_unfused.onnx",
        "ours_deploy": args.out_dir / "ours_deploy_fused.onnx",
        "baseline": args.out_dir / "yolo26n_baseline_fused.onnx",
    }
    report: dict[str, Any] = {
        "environment": {
            "python": sys.executable,
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "packages": package_status(),
            "trtexec_path": find_trtexec(),
        },
        "paths": {k: str(v) for k, v in paths.items()},
        "torch_model_stats": {},
        "exports": {},
        "onnx_stats": {},
        "trtexec": {},
    }

    if not args.skip_export:
        if not OURS_PT.is_file():
            raise FileNotFoundError(OURS_PT)
        if not BASELINE_PT.is_file():
            raise FileNotFoundError(BASELINE_PT)
        report["torch_model_stats"]["ours"] = torch_model_stats(OURS_PT, args.imgsz)
        report["torch_model_stats"]["baseline"] = torch_model_stats(BASELINE_PT, args.imgsz)
        report["exports"]["ours_train"] = export_onnx(OURS_PT, paths["ours_train"], fuse=False, opset=args.opset, imgsz=args.imgsz)
        report["exports"]["ours_deploy"] = export_onnx(OURS_PT, paths["ours_deploy"], fuse=True, opset=args.opset, imgsz=args.imgsz)
        report["exports"]["baseline"] = export_onnx(BASELINE_PT, paths["baseline"], fuse=True, opset=args.opset, imgsz=args.imgsz)

    for key, path in paths.items():
        if path.is_file():
            report["onnx_stats"][key] = onnx_stats(path)

    if not args.skip_trt:
        trtexec = find_trtexec()
        if trtexec is None:
            report["trtexec_error"] = "trtexec not found"
            print("[warn] trtexec not found; skip TensorRT build.")
        else:
            for key, path in paths.items():
                if path.is_file():
                    report["trtexec"][key] = run_trtexec(
                        path,
                        args.out_dir / f"{key}.fp16.engine",
                        trtexec,
                        args.trtexec_args,
                    )

    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
