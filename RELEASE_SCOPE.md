# Release scope

## Included

- Method source, selected configurable defaults, and implementation constants.
- Selected parameter documentation and a machine-readable partial record, with code defaults and BDD100K recovered method options identified separately.
- RepP3 network definitions and the vendored framework.
- Dataset YAML examples and filename-only KITTI splits.
- Image corruption, validation, export, and fusion inspection utilities.
- Third-party licensing and attribution.

## Not published in this release

- Complete recovered historical training commands and experiment-specific configuration files. Only the selected BDD100K method-option subset is disclosed.
- Training logs, per-epoch CSV files, numerical experiment archives, and checkpoint hash indexes.
- Archived frozen-protocol manifests, historical launch scripts, and the associated enforcement branch in the training entry point.
- Internal working notes, source-path audit records, and historical deployment provenance.
- Dataset images/labels, predictions, student/teacher weights, ONNX files, or TensorRT engines.

The omission of these assets does not resolve differences between historical experimental protocols. Historical runs used different settings, and the links between the accuracy, robustness, and deployment checkpoints remain under verification. Current source defaults must not be described as the exact historical configuration. This release does not substantiate a full reproduction or a controlled re-evaluation of the manuscript tables.

Selected code defaults and upstream framework defaults are retained. The project training entry omits its `lr0` and `mosaic` defaults and requires both options explicitly. Vendored upstream defaults and optimizer algorithms remain unchanged; their constants do not identify the manuscript's actual run settings. Corruption and evaluation settings are also retained so that the utilities keep a defined interface. This is not a package with every numerical parameter removed.

All evaluation weight paths are generic placeholders. Users must supply and identify their own checkpoints and required optional external model assets. Syntax/static checks do not establish runtime correctness or reproduced scientific results.

The branch history for this selected-parameter version starts at a fresh root commit. This does not guarantee removal of old GitHub object caches or external copies.
