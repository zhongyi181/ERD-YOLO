# ERD-YOLO

**Edge-oriented Reparameterized Detector**

Research source for lightweight object detection in road scenes. ERD-YOLO combines RepP3, class–scale collaborative weighted loss (CS-CWL), and quality-gated knowledge distillation (QG-KD) on a YOLO26n student, with a frozen YOLO26l teacher used during training.

## Release scope

This release provides the method implementation with its current code defaults, network definitions, corruption generation and evaluation tools, and filename-only KITTI splits. It does not include historical launch commands, experiment-specific training configurations, training logs, numerical result archives, or trained weights.

**The defaults in this source are not a recovered record of the training settings used for the manuscript's results. This repository is not sufficient to reproduce all manuscript tables.** Historical runs used different training settings, and the correspondence between the reported accuracy, robustness, and deployment checkpoints still requires verification. The repository makes no claim that these historical comparisons constitute a controlled evaluation under identical training settings.

See [RELEASE_SCOPE.md](RELEASE_SCOPE.md) for the complete boundary of this release.

## Contents

| Path | Purpose |
| --- | --- |
| `ultralytics/cfg/models/26/yolo26n-rep-p3.yaml` | RepP3 student network definition |
| `ultralytics/utils/ogdkd.py` | CS-CWL and QG-KD implementation; historical internal module name retained |
| `ultralytics/utils/dsakd.py` | Feature hooks and supporting utilities |
| `train_yolo26n_ogdkd_kitti.py` | Configurable training entry point with code defaults |
| `check_yolo26_ogdkd.py` | Method and dataset checks |
| `scripts/robustness/` | Offline corruption generation, evaluation, and summaries |
| `scripts/final_eval/` | Clean-image evaluation utility |
| `tools/rep_p3_deploy_audit.py` | RepConv fusion and export inspection |
| `splits/` | KITTI image filenames; no images or labels |
| `patches/repp3_upstream.patch` | RepConv parser registration relative to the pinned upstream source |

## Installation

The vendored framework is based on Ultralytics 8.4.33 at commit `dfbb343547070756b68b156f4471acd1f7e01b7e`. Prepare a suitable Python/PyTorch environment, then install the repository:

```bash
python -m pip install -e .
```

Corruption generation additionally requires `imagecorruptions==1.1.2`, NumPy, Pillow, SciPy, and scikit-image. The motion-blur implementation requires a working Wand/ImageMagick installation. Runtime dependencies and dataset files are not bundled.

## Data and weights

Obtain KITTI or BDD100K through their providers and follow their terms. Edit the example dataset YAML files in `configs/` for your local dataset. The included split lists contain filenames only.

Paths may be configured through `DATA_ROOT`, `WEIGHTS_ROOT`, and `OUTPUT_ROOT`. The evaluation scripts contain generic weight-path placeholders. Supply your own weights and inspect the mappings before running evaluation. A directory or model alias is not evidence that a supplied checkpoint matches a manuscript result.

## Selected public parameters

See [PARAMETERS.md](PARAMETERS.md) and [configs/public_parameters.json](configs/public_parameters.json) for selected code defaults and a separate subset of BDD100K recovered method options. These records have different sources and do not form a complete historical training configuration.

## Training interface

Inspect the available settings before configuring a new experiment:

```bash
python train_yolo26n_ogdkd_kitti.py --help
python check_yolo26_ogdkd.py --help
```

Selected code defaults remain visible and configurable. The project training entry requires explicit `--lr0` and `--mosaic` values selected by the user. It provides no default values for these two options. Generic defaults and optimizer constants in the vendored upstream framework remain unchanged and do not establish manuscript run settings. A default launch enables class/scale weighting; QG-KD and gap weighting have separate explicit switches. QG-KD requires a trained teacher checkpoint. Effective-number class weights are computed from the actual training labels rather than taken from the manual fallback tuple. Choose and record a consistent protocol for any new comparison. No historical reproduction command is supplied.

## Corruption and evaluation tools

```bash
python scripts/robustness/01_generate_kitti_corruptions.py --help
python scripts/robustness/02_val_kitti_corruptions.py --help
python scripts/robustness/03_summarize_corrupt_results.py --help
```

Fog, motion blur, Gaussian noise, and contrast call imagecorruptions; low light uses a separate gamma transformation. Evaluation and corruption parameters remain in the code because they define these tools. This is not a complete official KITTI-C benchmark implementation.

The existing summary tool reports mAP50 means and relative drops using the supplied clean row. Keep the clean and corrupted evaluations tied to the same checkpoint and evaluation settings. Optional RT-DETR adapters require external model code/configuration that is not included here. The native RT-DETR-S adapter expects `RTDETR_ROOT`, a user-supplied `RTDETR_CONFIG`, `weights/detr_s/best.pth`, and COCO-format validation annotations under `DATA_ROOT/coco_rtdetrv2/annotations/instances_val.json`. Clean-evaluation `outputs/<model_key>/args.yaml` paths are optional existence checks; missing files do not provide historical settings or prevent that evaluation.

## Validation status

Release checks are limited to static source and package inspection. This publication does not establish that training, accuracy, robustness, export, or device throughput has been reproduced. No trained checkpoint or TensorRT engine is included.

## License and attribution

The inherited [AGPL-3.0 license](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md) apply. The base detector, RepConv implementation, and training framework come from Ultralytics. See [UPSTREAM_README.md](UPSTREAM_README.md) for the upstream project documentation; upstream examples are not this project's experiment records.

No publication acceptance or final bibliographic citation is claimed by this repository.

## Repository version

This selected-parameter release is published as a fresh root commit. Replacing the repository's branch history does not retract cached GitHub objects or external copies of earlier public versions.
