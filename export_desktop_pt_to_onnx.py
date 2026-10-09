from pathlib import Path
import os
import re
import shutil
from ultralytics import YOLO

# Portable path defaults; run from the repository root or set these environment variables.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data/kitti_3cls_yolo")).resolve()
WEIGHTS_ROOT = Path(os.environ.get("WEIGHTS_ROOT", "weights")).resolve()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "outputs")).resolve()

# Each weights/<model_key>/best.pt must be supplied by the user.
src_root = WEIGHTS_ROOT
out_root = Path(str(OUTPUT_ROOT / "onnx_export"))
out_root.mkdir(parents=True, exist_ok=True)

pt_files = sorted(src_root.glob("*/best.pt"))

if not pt_files:
    raise FileNotFoundError(f"No best.pt found under {src_root}")

print(f"Found {len(pt_files)} model(s):")
for p in pt_files:
    print(" -", p)

def safe_name(name: str) -> str:
    name = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", name)
    return name.strip("_")

for pt in pt_files:
    run_name = safe_name(pt.parent.name)
    export_dir = out_root / run_name
    export_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Exporting: {pt}")
    print(f"Output dir: {export_dir}")
    print("=" * 80)

    model = YOLO(str(pt))
    result = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        dynamic=False
    )

    result_path = Path(result)
    target_onnx = export_dir / f"{run_name}.onnx"
    shutil.copy2(result_path, target_onnx)

    target_pt = export_dir / f"{run_name}.pt"
    shutil.copy2(pt, target_pt)

    print(f"Saved ONNX: {target_onnx}")
    print(f"Saved PT:   {target_pt}")

print("\nAll exports finished.")
print(f"Output root: {out_root}")
