# Selected public parameters

This document records a selected subset of code defaults and recovered method options. It is not a complete training configuration and does not reproduce the manuscript tables. The sections have different sources and must not be merged into a claimed historical run configuration.

The machine-readable counterpart is `configs/public_parameters.json`. Its nested sections are descriptive records, not an Ultralytics configuration file to pass directly to the trainer.

## Training interface defaults

| Parameter | Default |
| --- | ---: |
| epochs | 100 |
| batch | 32 |
| imgsz | 640 |
| optimizer | MuSGD |

These values describe the current custom training interface. They do not establish the batch size, optimizer, or other settings used in a historical KITTI or BDD100K run.

## Method code defaults

| Parameter | Default |
| --- | ---: |
| class_balance_beta | 0.999 |
| class_weight_min | 0.5 |
| class_weight_max | 2.0 |
| beta_size | 0.5 |
| tau_size | 0.02 |
| size_weight_max | 1.5 |
| lambda_kd | 0.3 |
| eta_box_kd | 1.0 |
| teacher_quality_thr | 0.5 |
| quality_margin | 0.05 |

Effective-number class weights are computed from the training labels and clipped to the configured bounds. The listed QG-KD coefficients apply when QG-KD is enabled; their presence does not imply that a default launch enables distillation.

## Framework detection defaults

The selected defaults are `lrf=0.01`, `weight_decay=0.0005`, and `warmup_epochs=3.0`. Loss gains are `box=7.5`, `cls=0.5`, and the legacy framework key `dfl=1.5`. In the included YOLO26 implementation (`reg_max=1`), the third detection term is normalized LTRB distance L1.

Selected augmentation defaults are:

| Parameter | Default |
| --- | ---: |
| hsv_h | 0.015 |
| hsv_s | 0.7 |
| hsv_v | 0.4 |
| degrees | 0.0 |
| translate | 0.1 |
| scale | 0.5 |
| shear | 0.0 |
| perspective | 0.0 |
| flipud | 0.0 |
| fliplr | 0.5 |
| bgr | 0.0 |
| mixup | 0.0 |
| cutmix | 0.0 |
| copy_paste | 0.0 |

These are framework defaults; entry-point and command-line overrides can change them. They are not a recovered historical configuration.

## BDD100K recovered method options

The following subset comes from a saved recovered BDD100K method command. It does not use the QG-KD defaults listed above.

| Option | Recorded command value |
| --- | --- |
| lambda_kd | 0.2 |
| eta_box_kd | 0.1 |
| teacher_quality_thr | 0.05 |
| quality_margin | 0.0 |
| use_class_weight | true |
| use_size_weight | true |
| use_ogdkd | true |
| use_gap_weight | false |
| student_layers | 17,20,23 |
| teacher_layers | 16,19,22 |

The original record still contains unresolved model, data, and teacher path variables. This subset does not identify a released teacher checkpoint or establish the complete executed training settings. The current code has not been verified as an exact frozen copy of the historical training source.

## Omitted project settings

Project-specific values for `lr0` and `mosaic` are intentionally omitted from this disclosure. Users must select their own values for a new experiment; the project training entry requires both options explicitly. The vendored upstream framework retains its generic defaults and optimizer algorithms, which do not identify historical manuscript run settings. Other source implementation constants and framework interfaces remain visible; this document does not claim that every numerical parameter has been removed from the repository.

Historical runs used different training settings. The correspondence between the reported accuracy, corruption robustness, and deployment checkpoints still requires verification. Omission of settings does not resolve those differences or support a claim of controlled comparisons under identical settings.

This selected-parameter release starts a new branch history at a fresh root commit. Replacing branch history cannot retract old cached GitHub objects or external copies.
