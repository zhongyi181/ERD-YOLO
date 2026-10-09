from __future__ import annotations

import argparse
import hashlib
import math
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.detect import DetectionTrainer  # noqa: E402
from ultralytics.utils import DEFAULT_CFG, LOGGER, LOCAL_RANK  # noqa: E402
from ultralytics.utils.dsakd import FeatureHookStore, resolve_distill_layers, selected_feature_shapes  # noqa: E402
from ultralytics.utils.ogdkd import (  # noqa: E402
    OGDKDLoss,
    OGDKDSettings,
    count_class_instances_from_labels,
    effective_number_class_weights,
    resolve_class_weight_mode,
)
from ultralytics.utils.torch_utils import unwrap_model  # noqa: E402

# Portable path defaults; run from the repository root or set these environment variables.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data/kitti_3cls_yolo")).resolve()
WEIGHTS_ROOT = Path(os.environ.get("WEIGHTS_ROOT", "weights")).resolve()
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "outputs")).resolve()


@dataclass
class OGDKDConfig:
    teacher: str | None
    student_layers: str | None = None
    teacher_layers: str | None = None
    class_weights: tuple[float, ...] = (1.0, 1.25, 1.5)
    class_weight_mode: str = "effective"
    class_counts: tuple[int, ...] = ()
    class_balance_beta: float = 0.999
    class_weight_min: float = 0.5
    class_weight_max: float = 2.0
    use_class_weight: bool = True
    use_size_weight: bool = True
    use_gap_weight: bool = True
    use_ogdkd: bool = True
    beta_size: float = 0.5
    tau_size: float = 0.02
    size_weight_max: float = 1.5
    gamma_gap: float = 0.5
    gap_weight_max: float = 1.5
    lambda_kd: float = 0.3
    eta_box_kd: float = 1.0
    teacher_quality_thr: float = 0.5
    quality_margin: float = 0.05
    teacher_sha256: str | None = None
    experiment_manifest: str | None = None
    dry_run: bool = False

    @property
    def teacher_guided(self) -> bool:
        """Return whether this run needs teacher predictions."""
        return self.use_gap_weight or self.use_ogdkd

    @property
    def mode_name(self) -> str:
        """Return the user-facing experiment mode."""
        if self.teacher_guided:
            return "teacher-student guided loss"
        return "YOLO26n improved detection loss only"

    def to_settings(self) -> OGDKDSettings:
        return OGDKDSettings(
            class_weights=self.class_weights,
            class_weight_mode=self.class_weight_mode,
            class_counts=self.class_counts,
            class_balance_beta=self.class_balance_beta,
            class_weight_min=self.class_weight_min,
            class_weight_max=self.class_weight_max,
            use_class_weight=self.use_class_weight,
            use_size_weight=self.use_size_weight,
            use_gap_weight=self.use_gap_weight,
            use_ogdkd=self.use_ogdkd,
            beta_size=self.beta_size,
            tau_size=self.tau_size,
            size_weight_max=self.size_weight_max,
            gamma_gap=self.gamma_gap,
            gap_weight_max=self.gap_weight_max,
            lambda_kd=self.lambda_kd,
            eta_box_kd=self.eta_box_kd,
            teacher_quality_thr=self.teacher_quality_thr,
            quality_margin=self.quality_margin,
        )


class OGDKDTrainer(DetectionTrainer):
    """Detection trainer for Optimization-Guided Dynamic Knowledge Distillation."""

    def __init__(
        self,
        cfg: str | dict = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
        ogdkd: OGDKDConfig | None = None,
        provenance: dict[str, Any] | None = None,
    ):
        self.ogdkd_cfg = ogdkd
        self.provenance = provenance or {}
        self.teacher: torch.nn.Module | None = None
        self.ogdkd_loss: OGDKDLoss | None = None
        self.student_hooks: FeatureHookStore | None = None
        self.teacher_hooks: FeatureHookStore | None = None
        self.student_layers: list[int] = []
        self.teacher_layers: list[int] = []
        self.is_dry_run = bool(ogdkd.dry_run) if ogdkd is not None else False
        self._loss_wrapper_installed = False
        self._last_ogdkd_diagnostics: dict[str, Any] = {}
        self._method_epoch_sums: dict[str, float] = {}
        self._method_epoch_weight_count = 0
        self._method_epoch_batches = 0
        self._method_epoch_all_finite = True
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        if self.ddp:
            raise NotImplementedError("OG-DKD script currently supports single-device training. Use --device 0.")
        self.add_callback("on_train_epoch_start", lambda trainer: trainer._reset_method_epoch_stats())
        self._write_experiment_config(stage="trainer_initialized")

    def set_model_attributes(self) -> None:
        """Set detection attributes, then prepare the frozen teacher and OG-DKD criterion."""
        super().set_model_attributes()
        self._set_loss_names()
        if self.ogdkd_cfg is None:
            return
        self._setup_ogdkd()

    def _set_loss_names(self) -> None:
        self.loss_names = (
            "box_loss",
            "cls_loss",
            "dist_l1_loss",
            "qg_kd_cls_loss",
            "qg_kd_box_loss",
        )

    def _setup_ogdkd(self) -> None:
        """Load teacher, resolve P3/P4/P5 hooks, and replace the student criterion for training."""
        student = unwrap_model(self.model)
        settings = self.ogdkd_cfg.to_settings()
        self.ogdkd_loss = OGDKDLoss(student, settings)
        student.criterion = self.ogdkd_loss

        LOGGER.info(f"Mode: {self.ogdkd_cfg.mode_name}")
        LOGGER.info(
            "OG-DKD: "
            f"class_weight_mode={settings.class_weight_mode}, "
            f"class_weights={settings.class_weights if settings.class_weight_mode != 'effective' else 'pending'}, "
            f"use_class_weight={settings.use_class_weight}, "
            f"use_size_weight={settings.use_size_weight}, use_gap_weight={settings.use_gap_weight}, "
            f"use_ogdkd={settings.use_ogdkd}, lambda_kd={settings.lambda_kd}, "
            f"class_balance_beta={settings.class_balance_beta}, "
            f"class_weight_bounds=[{settings.class_weight_min}, {settings.class_weight_max}], "
            f"beta_size={settings.beta_size}, tau_size={settings.tau_size}, gamma_gap={settings.gamma_gap}, "
            f"teacher_quality_thr={settings.teacher_quality_thr}, quality_margin={settings.quality_margin}"
        )

        needs_teacher = self.ogdkd_cfg.teacher_guided
        if not needs_teacher:
            LOGGER.info("CW-SW: teacher is not required because gap weighting and KD are disabled.")
            return

        LOGGER.info(f"OG-DKD: loading teacher from {self.ogdkd_cfg.teacher}")
        self.teacher = YOLO(self.ogdkd_cfg.teacher, task="detect").model.to(self.device).eval()
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        self.student_layers = resolve_distill_layers(student, self.ogdkd_cfg.student_layers)
        self.teacher_layers = resolve_distill_layers(self.teacher, self.ogdkd_cfg.teacher_layers)
        if len(self.student_layers) != len(self.teacher_layers):
            raise ValueError(
                "Student and teacher hook layers must have the same count: "
                f"{self.student_layers} vs {self.teacher_layers}"
            )
        if len(self.student_layers) != 3:
            raise ValueError(f"OG-DKD expects 3 P3/P4/P5 hook layers, got {self.student_layers}.")

        imgsz = int(self.args.imgsz if isinstance(self.args.imgsz, int) else self.args.imgsz[0])
        student_shapes = selected_feature_shapes(student, self.student_layers, imgsz=imgsz, device=self.device)
        teacher_shapes = selected_feature_shapes(self.teacher, self.teacher_layers, imgsz=imgsz, device=self.device)
        student_hw = [student_shapes[i][-2:] for i in self.student_layers]
        teacher_hw = [teacher_shapes[i][-2:] for i in self.teacher_layers]
        if student_hw != teacher_hw:
            raise ValueError(f"Student/teacher feature spatial sizes mismatch: {student_hw} vs {teacher_hw}.")

        LOGGER.info(f"OG-DKD: student layers {self._format_shapes(student_shapes)}")
        LOGGER.info(f"OG-DKD: teacher layers {self._format_shapes(teacher_shapes)}")

    @staticmethod
    def _format_shapes(shapes: dict[int, tuple[int, ...]]) -> str:
        return ", ".join(f"{i}:{tuple(shape)}" for i, shape in shapes.items())

    def _verify_teacher_not_in_optimizer(self) -> None:
        """Prove that the frozen teacher has no parameter in the student optimizer."""
        if self.teacher is None:
            return
        teacher_ids = {id(p) for p in self.teacher.parameters()}
        optimizer_ids = {
            id(p)
            for group in self.optimizer.param_groups
            for p in group.get("params", [])
            if isinstance(p, torch.nn.Parameter)
        }
        overlap = teacher_ids & optimizer_ids
        if overlap:
            raise RuntimeError(f"Teacher leaked into optimizer: {len(overlap)} parameter tensors overlap.")
        if any(p.requires_grad for p in self.teacher.parameters()) or self.teacher.training:
            raise RuntimeError("Teacher must remain frozen and in eval mode before training starts.")
        LOGGER.info("Teacher optimizer isolation verified: frozen/eval and 0 parameter overlap.")

    @staticmethod
    def _plain(value: Any) -> Any:
        """Convert runtime values into YAML-safe scalars and containers."""
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): OGDKDTrainer._plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [OGDKDTrainer._plain(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _write_experiment_config(self, stage: str) -> None:
        """Write a self-contained method/protocol snapshot into the new run directory."""
        if self.ogdkd_cfg is None:
            return
        payload = self._plain(self.provenance)
        payload["snapshot_stage"] = stage
        payload["run_directory"] = str(self.save_dir)
        payload["method"] = {
            "RepP3_enabled": bool(payload.get("method", {}).get("RepP3_enabled", False)),
            "CS_CWL_enabled": bool(self.ogdkd_cfg.use_class_weight and self.ogdkd_cfg.use_size_weight),
            "QG_KD_enabled": bool(self.ogdkd_cfg.use_ogdkd),
            "gap_weight_enabled": bool(self.ogdkd_cfg.use_gap_weight),
        }
        payload["teacher_path"] = self.ogdkd_cfg.teacher
        payload["teacher_sha256"] = self.ogdkd_cfg.teacher_sha256
        payload["cs_cwl"] = {
            "class_weight_mode": self.ogdkd_cfg.class_weight_mode,
            "class_counts": list(self.ogdkd_cfg.class_counts),
            "class_weights": list(self.ogdkd_cfg.class_weights),
            "class_balance_beta": self.ogdkd_cfg.class_balance_beta,
            "class_weight_min": self.ogdkd_cfg.class_weight_min,
            "class_weight_max": self.ogdkd_cfg.class_weight_max,
            "scale_beta": self.ogdkd_cfg.beta_size,
            "scale_tau": self.ogdkd_cfg.tau_size,
            "scale_weight_max": self.ogdkd_cfg.size_weight_max,
            "classification_weight": "class_weight (gap disabled)",
            "box_weight": "clamp(class_weight * scale_weight, 1.0, 2.0)",
            "dist_l1_weight": "clamp(scale_weight, 1.0, 1.8)",
        }
        payload["qg_kd"] = {
            "tau_teacher_quality": self.ogdkd_cfg.teacher_quality_thr,
            "delta_quality_margin": self.ogdkd_cfg.quality_margin,
            "classification_coefficient": self.ogdkd_cfg.lambda_kd,
            "box_relative_coefficient": self.ogdkd_cfg.eta_box_kd,
            "box_effective_coefficient": self.ogdkd_cfg.lambda_kd * self.ogdkd_cfg.eta_box_kd,
            "student_layers": list(self.student_layers),
            "teacher_layers": list(self.teacher_layers),
        }
        payload["loss_names"] = list(self.loss_names)
        payload["loss_semantics"] = {
            "third_detection_term": "normalized LTRB distance L1",
            "traditional_DFL": False,
            "reg_max": 1,
            "legacy_ultralytics_gain_key": "dfl",
        }
        config_path = Path(self.save_dir) / "experiment_config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def _setup_train(self) -> None:
        """Run normal setup, then install the training-only loss wrapper."""
        super()._setup_train()
        self._install_loss_wrapper()

    def _build_train_pipeline(self) -> None:
        """Build dataloaders and optimizer; dry-run exits after one optimizer step."""
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train")
        self._configure_auto_cbsw()
        self.test_loader = self.get_dataloader(
            self.data.get("val") or self.data.get("test"),
            batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
            rank=LOCAL_RANK,
            mode="val",
        )
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        if self.is_dry_run:
            self.accumulate = 1
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        self._setup_scheduler()
        self._verify_teacher_not_in_optimizer()
        self._write_experiment_config(stage="class_balance_resolved")

    def _configure_auto_cbsw(self) -> None:
        """Resolve class weights from the train dataloader labels and update the active criterion."""
        if self.ogdkd_cfg is None or self.ogdkd_loss is None:
            return

        nc = self._resolve_num_classes()
        labels = getattr(getattr(self.train_loader, "dataset", None), "labels", [])
        class_counts, invalid = count_class_instances_from_labels(labels, nc)
        if invalid:
            LOGGER.warning(f"[AUTO-CBSW] ignored {invalid} train labels outside class range [0, {nc - 1}].")

        mode = self.ogdkd_cfg.class_weight_mode
        clipped = False
        if mode == "none":
            class_weights = tuple(1.0 for _ in range(nc))
            LOGGER.info("[AUTO-CBSW] class_weight_mode: none")
            LOGGER.info("[AUTO-CBSW] class weights disabled; using all ones.")
        elif mode == "manual":
            class_weights = self.ogdkd_cfg.class_weights
            if len(class_weights) != nc:
                raise ValueError(
                    f"--class-weight-mode manual expects {nc} class weights from the dataset/model, "
                    f"got {len(class_weights)}: {class_weights}."
                )
            LOGGER.info("[AUTO-CBSW] class_weight_mode: manual")
            LOGGER.info(f"[AUTO-CBSW] manual class weights: {[round(float(x), 6) for x in class_weights]}")
        else:
            class_weights, clipped = effective_number_class_weights(
                class_counts,
                beta=self.ogdkd_cfg.class_balance_beta,
                weight_min=self.ogdkd_cfg.class_weight_min,
                weight_max=self.ogdkd_cfg.class_weight_max,
            )
            LOGGER.info("[AUTO-CBSW] class_weight_mode: effective")
            LOGGER.info(f"[AUTO-CBSW] effective class weights: {[round(float(x), 6) for x in class_weights]}")

        self.ogdkd_cfg.class_counts = class_counts
        self.ogdkd_cfg.class_weights = class_weights
        self.ogdkd_cfg.use_class_weight = mode != "none"
        self.ogdkd_loss.set_class_balance(class_weights, class_counts, mode)

        LOGGER.info(f"[AUTO-CBSW] train class counts: {list(class_counts)}")
        if clipped:
            LOGGER.info(
                "[AUTO-CBSW] some class weights were clipped to "
                f"[{self.ogdkd_cfg.class_weight_min}, {self.ogdkd_cfg.class_weight_max}]"
            )
        LOGGER.info("[AUTO-CBSW] size weight: keep existing implementation")

    def _resolve_num_classes(self) -> int:
        """Resolve nc from dataset metadata first, then model attributes."""
        nc = self.data.get("nc") if isinstance(self.data, dict) else None
        if nc is None and isinstance(self.data, dict):
            names = self.data.get("names")
            if isinstance(names, dict):
                nc = len(names)
            elif isinstance(names, (list, tuple)):
                nc = len(names)
        if nc is None:
            nc = getattr(unwrap_model(self.model), "nc", None)
        if nc is None or int(nc) <= 0:
            raise ValueError(f"Unable to resolve a valid number of classes, got nc={nc}.")
        return int(nc)

    def _install_loss_wrapper(self) -> None:
        """Wrap student loss so teacher forward and hooks are used only during training."""
        if self._loss_wrapper_installed or self.ogdkd_loss is None:
            return

        student = unwrap_model(self.model)
        if self.teacher is not None:
            self.student_hooks = FeatureHookStore(student, self.student_layers)
            self.teacher_hooks = FeatureHookStore(self.teacher, self.teacher_layers)

        def loss_with_ogdkd(batch: dict[str, torch.Tensor], preds: Any = None):
            student_was_training = student.training
            teacher_output = None
            teacher_head = None
            student_feature_shapes, teacher_feature_shapes = [], []

            if preds is None:
                if self.student_hooks is not None:
                    self.student_hooks.clear()
                preds = student.forward(batch["img"])

            try:
                if student_was_training and self.teacher is not None and self._needs_teacher_forward():
                    self.teacher.eval()
                    if self.teacher_hooks is not None:
                        self.teacher_hooks.clear()
                    with torch.no_grad():
                        teacher_output = self.teacher(batch["img"])

                    student_missing = [i for i in self.student_layers if i not in self.student_hooks.outputs]
                    teacher_missing = [i for i in self.teacher_layers if i not in self.teacher_hooks.outputs]
                    assert not student_missing, f"Student hook missed layers {student_missing}"
                    assert not teacher_missing, f"Teacher hook missed layers {teacher_missing}"
                    student_feature_shapes = [
                        (i, tuple(self.student_hooks.outputs[i].shape)) for i in self.student_layers
                    ]
                    teacher_feature_shapes = [
                        (i, tuple(self.teacher_hooks.outputs[i].shape)) for i in self.teacher_layers
                    ]
                    teacher_head = unwrap_model(self.teacher).model[-1]

                loss_vec, items = self.ogdkd_loss(
                    preds,
                    batch,
                    teacher_output=teacher_output,
                    teacher_head=teacher_head,
                )
                diag = dict(self.ogdkd_loss.last_diag)
                diag.update(
                    {
                        "student_distill_layers": list(self.student_layers),
                        "teacher_distill_layers": list(self.teacher_layers),
                        "student_feature_shapes": student_feature_shapes,
                        "teacher_feature_shapes": teacher_feature_shapes,
                        "batch_idx_shape": tuple(batch["batch_idx"].shape) if "batch_idx" in batch else None,
                        "cls_shape": tuple(batch["cls"].shape) if "cls" in batch else None,
                        "bboxes_shape": tuple(batch["bboxes"].shape) if "bboxes" in batch else None,
                        "bboxes_format": "normalized xywh",
                        "train_batch_size": batch["img"].shape[0],
                        "teacher_forward": teacher_output is not None,
                    }
                )
                self._assert_finite_diagnostics(diag, loss_vec, items)
                self._accumulate_method_diagnostics(diag)
                self._last_ogdkd_diagnostics = diag
                return loss_vec, items
            finally:
                if self.student_hooks is not None:
                    self.student_hooks.clear()
                if self.teacher_hooks is not None:
                    self.teacher_hooks.clear()

        student.loss = loss_with_ogdkd
        student.criterion = self.ogdkd_loss
        student._ogdkd_loss_active = True
        self._loss_wrapper_installed = True
        LOGGER.info("Custom CW-SW / OG-DKD loss wrapper installed; EMA/best.pt remain plain YOLO26n.")

    def _needs_teacher_forward(self) -> bool:
        if self.ogdkd_cfg is None:
            return False
        return self.ogdkd_cfg.teacher_guided

    def _reset_method_epoch_stats(self) -> None:
        """Reset proof-oriented statistics at the beginning of every training epoch."""
        self._method_epoch_sums = {
            "class": 0.0,
            "scale": 0.0,
            "cls": 0.0,
            "box": 0.0,
            "dist": 0.0,
            "kd_candidates": 0.0,
            "kd_gate_pass": 0.0,
            "kd_valid_pairs": 0.0,
        }
        self._method_epoch_weight_count = 0
        self._method_epoch_batches = 0
        self._method_epoch_all_finite = True

    @staticmethod
    def _diag_scalar(value: Any) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

    @classmethod
    def _all_checks_true(cls, checks: dict[str, Any]) -> bool:
        return all(bool(v.detach().cpu().item()) if isinstance(v, torch.Tensor) else bool(v) for v in checks.values())

    def _assert_finite_diagnostics(
        self, diag: dict[str, Any], loss_vec: torch.Tensor, items: torch.Tensor
    ) -> None:
        """Abort immediately on any non-finite loss, weight, or diagnostic check."""
        failures: list[str] = []
        if not bool(torch.isfinite(loss_vec).all().detach().cpu()):
            failures.append("loss_vec")
        if not bool(torch.isfinite(items).all().detach().cpu()):
            failures.append("loss_items")
        for scope in ("finite_checks",):
            checks = diag.get(scope) or {}
            if checks and not self._all_checks_true(checks):
                failures.append(scope)
        for branch_name in ("one2many", "one2one"):
            branch = diag.get(branch_name) or {}
            checks = branch.get("finite_checks") or {}
            if checks and not self._all_checks_true(checks):
                failures.append(f"{branch_name}.finite_checks")
        self._method_epoch_all_finite = self._method_epoch_all_finite and not failures
        if failures:
            raise FloatingPointError(f"Non-finite method loss/diagnostics detected: {failures}")

    def _accumulate_method_diagnostics(self, diag: dict[str, Any]) -> None:
        """Aggregate batch diagnostics for direct per-epoch evidence in results.csv."""
        if not self._method_epoch_sums:
            self._reset_method_epoch_stats()
        for branch_name in ("one2many", "one2one"):
            branch = diag.get(branch_name) or {}
            fg_count = int(branch.get("fg_count") or 0)
            if fg_count <= 0:
                continue
            self._method_epoch_weight_count += fg_count
            for short, key in (
                ("class", "mean_class_weight"),
                ("scale", "mean_size_weight"),
                ("cls", "mean_cls_weight"),
                ("box", "mean_box_weight"),
                ("dist", "mean_dist_weight"),
            ):
                self._method_epoch_sums[short] += self._diag_scalar(branch.get(key, 0.0)) * fg_count

        kd_branch = diag.get("one2many") or diag.get("detect") or {}
        self._method_epoch_sums["kd_candidates"] += int(kd_branch.get("kd_candidate_count") or 0)
        self._method_epoch_sums["kd_gate_pass"] += int(kd_branch.get("kd_gate_pass_count") or 0)
        self._method_epoch_sums["kd_valid_pairs"] += int(kd_branch.get("kd_positive_count") or 0)
        self._method_epoch_batches += 1

    def _method_epoch_metrics(self) -> dict[str, float]:
        """Return numeric method proof fields suitable for the standard results.csv writer."""
        count = max(self._method_epoch_weight_count, 1)
        candidates = self._method_epoch_sums.get("kd_candidates", 0.0)
        gate_pass = self._method_epoch_sums.get("kd_gate_pass", 0.0)
        return {
            "method/RepP3_enabled": float(bool(self.provenance.get("method", {}).get("RepP3_enabled", False))),
            "method/CS_CWL_enabled": float(
                bool(self.ogdkd_cfg.use_class_weight and self.ogdkd_cfg.use_size_weight)
            ),
            "method/QG_KD_enabled": float(bool(self.ogdkd_cfg.use_ogdkd)),
            "method/cs_mean_class_weight": self._method_epoch_sums.get("class", 0.0) / count,
            "method/cs_mean_scale_weight": self._method_epoch_sums.get("scale", 0.0) / count,
            "method/cs_mean_cls_weight": self._method_epoch_sums.get("cls", 0.0) / count,
            "method/cs_mean_box_weight": self._method_epoch_sums.get("box", 0.0) / count,
            "method/cs_mean_dist_weight": self._method_epoch_sums.get("dist", 0.0) / count,
            "method/qg_kd_candidate_count": candidates,
            "method/qg_kd_gate_pass_count": gate_pass,
            "method/qg_kd_valid_pair_count": self._method_epoch_sums.get("kd_valid_pairs", 0.0),
            "method/qg_kd_gate_pass_ratio": gate_pass / candidates if candidates > 0 else 0.0,
            "method/all_finite": float(self._method_epoch_all_finite),
        }

    def _model_train(self) -> None:
        """Keep student trainable and teacher frozen/eval."""
        super()._model_train()
        if self.teacher is not None:
            self.teacher.eval()

    def optimizer_step(self) -> None:
        """Run dry-run diagnostics before any optimizer mutation."""
        if self.is_dry_run:
            self._print_dry_run_diagnostics()
            return
        super().optimizer_step()

    def _print_dry_run_diagnostics(self) -> None:
        """Print graph, hook, teacher and OG-DKD diagnostics for one backward pass."""
        diag = self._last_ogdkd_diagnostics
        settings = self.ogdkd_cfg.to_settings()
        student_grad = any(p.grad is not None for p in self.model.parameters())
        teacher_grad = any(p.grad is not None for p in self.teacher.parameters()) if self.teacher is not None else False
        teacher_frozen = (
            all(not p.requires_grad for p in self.teacher.parameters()) if self.teacher is not None else False
        )
        teacher_eval = self.teacher is not None and not self.teacher.training
        student_training = unwrap_model(self.model).training
        extra_params = self.ogdkd_loss.trainable_parameters() if self.ogdkd_loss is not None else []
        extra_in_optimizer, extra_tensors = self._optimizer_extra_coverage(extra_params)

        def scalar(value: Any, default: float = 0.0) -> float:
            if value is None:
                return default
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu())
            return float(value)

        branch = diag.get("one2many") or {}
        branch2 = diag.get("one2one") or {}
        finite_root = diag.get("finite_checks") or {}
        finite_ok = all(bool(v.item()) if isinstance(v, torch.Tensor) else bool(v) for v in finite_root.values())
        branch_finite = all(
            bool(v.item()) if isinstance(v, torch.Tensor) else bool(v)
            for v in (branch.get("finite_checks") or {}).values()
        )
        one2one_finite = all(
            bool(v.item()) if isinstance(v, torch.Tensor) else bool(v)
            for v in (branch2.get("finite_checks") or {}).values()
        )

        print("\n" + "=" * 44)
        print("CW-SW / OG-DKD DRY-RUN DIAGNOSTICS")
        print(f"Mode: {self.ogdkd_cfg.mode_name}")
        self._print_run_config()
        print(f"Student training: {student_training}")
        print(f"Student gradients active: {student_grad}")
        if self.ogdkd_cfg.teacher_guided:
            print(f"Teacher loaded: {self.teacher is not None}")
            print(f"Teacher eval: {teacher_eval}")
            print(f"Teacher frozen: {teacher_frozen}")
            print(f"Teacher gradients active: {teacher_grad}")
            print(f"Student P3/P4/P5 hook layers: {diag.get('student_distill_layers')}")
            print(f"Teacher P3/P4/P5 hook layers: {diag.get('teacher_distill_layers')}")
            print(f"Student feature shapes: {diag.get('student_feature_shapes')}")
            print(f"Teacher feature shapes: {diag.get('teacher_feature_shapes')}")
        print(f"batch_idx shape: {diag.get('batch_idx_shape')}")
        print(f"cls shape: {diag.get('cls_shape')}")
        print(f"bboxes shape: {diag.get('bboxes_shape')}")
        print(f"class_weight_mode: {settings.class_weight_mode}")
        print(f"train class_counts: {settings.class_counts}")
        print(f"class_weights: {settings.class_weights}")
        print(
            "switches: "
            f"class={settings.use_class_weight}, size={settings.use_size_weight}, "
            f"gap={settings.use_gap_weight}, ogdkd={settings.use_ogdkd}"
        )
        print(
            "hyperparams: "
            f"lambda_kd={settings.lambda_kd}, eta_box_kd={settings.eta_box_kd}, "
            f"beta_size={settings.beta_size}, tau_size={settings.tau_size}, "
            f"gamma_gap={settings.gamma_gap}, teacher_quality_thr={settings.teacher_quality_thr}, "
            f"quality_margin={settings.quality_margin}"
        )
        print("size weight implementation: existing")
        print(f"one2many fg_count: {branch.get('fg_count')}")
        if self.ogdkd_cfg.teacher_guided:
            print(f"one2many teacher_available: {branch.get('teacher_available')}")
            print(f"one2many valid_teacher_count: {branch.get('valid_teacher_count')}")
        print(f"mean class/size/gap weights: {scalar(branch.get('mean_class_weight')):.4f} / "
              f"{scalar(branch.get('mean_size_weight')):.4f} / {scalar(branch.get('mean_gap_weight')):.4f}")
        print(f"mean box/cls/dist_l1 weights: {scalar(branch.get('mean_box_weight')):.4f} / "
              f"{scalar(branch.get('mean_cls_weight')):.4f} / {scalar(branch.get('mean_dist_weight')):.4f}")
        if self.ogdkd_cfg.teacher_guided:
            print(f"mean q_t/q_s: {scalar(branch.get('mean_q_t')):.6f} / {scalar(branch.get('mean_q_s')):.6f}")
            print(f"KD candidates/positives: {branch.get('kd_candidate_count')} / {branch.get('kd_positive_count')}")
            print(f"OG cls KD / box KD: {scalar(branch.get('cls_kd')):.6f} / {scalar(branch.get('box_kd')):.6f}")
        print(f"Loss total: {scalar(diag.get('loss_total')):.6f}")
        print(f"End2End weights o2m/o2o: {diag.get('o2m')} / {diag.get('o2o')}")
        print(f"one2many finite checks: {branch_finite}")
        print(f"one2one finite checks: {one2one_finite}")
        print(f"root finite checks: {finite_ok}")
        print(f"Extra train modules: {sum(p.numel() for p in extra_params)} params / {len(extra_params)} tensors")
        print(f"Optimizer extra params: {extra_in_optimizer}/{extra_tensors}")

        guided_ok = True
        if self.ogdkd_cfg.teacher_guided:
            guided_ok = (
                self.teacher is not None
                and teacher_eval
                and teacher_frozen
                and not teacher_grad
                and bool(diag.get("student_feature_shapes"))
                and bool(diag.get("teacher_feature_shapes"))
            )

        passed = (
            student_training
            and student_grad
            and guided_ok
            and finite_ok
            and branch_finite
            and (not branch2 or one2one_finite)
            and extra_in_optimizer == extra_tensors
        )
        if branch.get("teacher_error"):
            print(f"DRY-RUN teacher decode error: {branch.get('teacher_error')}")
            passed = False
        if passed:
            print("CW-SW / OG-DKD DRY-RUN PASSED: graph and loss are healthy.")
        else:
            print("CW-SW / OG-DKD DRY-RUN FAILED: inspect the diagnostics before training.")
        print("=" * 44 + "\n")
        raise SystemExit(0 if passed else 1)

    def _print_run_config(self) -> None:
        """Print the compact run settings requested for dry-run diagnostics."""
        print(f"model path: {self.args.model}")
        print(f"data path: {self.args.data}")
        print(f"epochs: {self.args.epochs}")
        print(f"batch: {self.args.batch}")
        print(f"optimizer: {self.args.optimizer}")
        print(f"pretrained: {bool(self.args.pretrained)}")
        print(f"use_class_weight: {self.ogdkd_cfg.use_class_weight}")
        print(f"class_weight_mode: {self.ogdkd_cfg.class_weight_mode}")
        print(f"use_size_weight: {self.ogdkd_cfg.use_size_weight}")
        print(f"use_gap_weight: {self.ogdkd_cfg.use_gap_weight}")
        print(f"use_ogdkd: {self.ogdkd_cfg.use_ogdkd}")

    def _optimizer_extra_coverage(self, extra_params: list[torch.nn.Parameter]) -> tuple[int, int]:
        extra_ids = {id(p) for p in extra_params}
        optimizer_ids = {
            id(p)
            for group in getattr(self.optimizer, "param_groups", [])
            for p in group.get("params", [])
            if isinstance(p, torch.nn.Parameter)
        }
        return len(extra_ids & optimizer_ids), len(extra_ids)

    def get_validator(self):
        """Return standard detector validator while keeping OG-DKD train loss names."""
        validator = super().get_validator()
        self._set_loss_names()
        return validator

    def label_loss_items(self, loss_items: list[float] | torch.Tensor | None = None, prefix: str = "train"):
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        values = loss_items.detach().flatten().cpu().tolist() if isinstance(loss_items, torch.Tensor) else list(loss_items)
        if len(values) < len(keys):
            values.extend([0.0] * (len(keys) - len(values)))
        elif len(values) > len(keys):
            values = values[: len(keys)]
        return dict(zip(keys, [round(float(x), 5) for x in values]))

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        """Append proof-oriented CS-CWL/QG-KD statistics to the normal results.csv row."""
        proof = self._method_epoch_metrics()
        if not bool(proof["method/all_finite"]) or not all(math.isfinite(float(value)) for value in proof.values()):
            raise FloatingPointError("Epoch method diagnostics are non-finite; refusing to continue training.")
        merged = {**metrics, **proof}
        LOGGER.info(
            "[METHOD-EPOCH] "
            f"epoch={self.epoch + 1} RepP3={int(proof['method/RepP3_enabled'])} "
            f"CS-CWL={int(proof['method/CS_CWL_enabled'])} QG-KD={int(proof['method/QG_KD_enabled'])} "
            f"class/scale/cls/box/dist={proof['method/cs_mean_class_weight']:.6f}/"
            f"{proof['method/cs_mean_scale_weight']:.6f}/{proof['method/cs_mean_cls_weight']:.6f}/"
            f"{proof['method/cs_mean_box_weight']:.6f}/{proof['method/cs_mean_dist_weight']:.6f} "
            f"qg_cls={float(merged.get('train/qg_kd_cls_loss', 0.0)):.6f} "
            f"qg_box={float(merged.get('train/qg_kd_box_loss', 0.0)):.6f} "
            f"valid_pairs={int(proof['method/qg_kd_valid_pair_count'])} "
            f"gate={int(proof['method/qg_kd_gate_pass_count'])}/{int(proof['method/qg_kd_candidate_count'])} "
            f"ratio={proof['method/qg_kd_gate_pass_ratio']:.6f} finite=1"
        )
        super().save_metrics(merged)

    def validate(self):
        """Validate the plain student/EMA model and pad OG-DKD val items with zeros."""
        original_loss_items = self.loss_items
        if isinstance(self.loss_items, torch.Tensor) and self.loss_items.numel() > 3:
            self.loss_items = self.loss_items[:3]
        try:
            return super().validate()
        finally:
            self.loss_items = original_loss_items


def parse_float_tuple(values: list[str] | tuple[str, ...] | str) -> tuple[float, ...]:
    """Parse class weights from '--class-weights 1 1.25 1.5' or '1,1.25,1.5'."""
    if isinstance(values, str):
        values = [values]
    pieces: list[str] = []
    for value in values:
        pieces.extend(x.strip() for x in str(value).split(",") if x.strip())
    return tuple(float(x) for x in pieces)


def resolve_local_path(value: str) -> str:
    """Resolve common repo-relative model paths while preserving external paths and model names."""
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


def sha256_file(path: str | Path) -> str:
    """Hash a file without modifying it."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    """Return the repository commit while keeping untracked source hashes separate."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def build_provenance(
    args: argparse.Namespace, model_path: str, overrides: dict[str, Any], ogdkd: OGDKDConfig
) -> dict[str, Any]:
    """Record the current run and optional user-provided metadata without prescribing a protocol."""
    import ultralytics

    source_paths = [
        Path(__file__).resolve(),
        ROOT / "ultralytics/utils/ogdkd.py",
        Path(model_path).resolve(),
        ROOT / "ultralytics/nn/modules/conv.py",
        ROOT / "ultralytics/nn/tasks.py",
    ]
    source_files = [
        {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in source_paths
    ]
    manifest_path = Path(args.experiment_manifest).resolve() if args.experiment_manifest else None
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) if manifest_path else {}
    data_manifest = manifest.get("inputs", {}).get("data_yaml", {}) if isinstance(manifest, dict) else {}
    return {
        "schema_version": 1,
        "identity": "CUSTOM_OGDKD_RUN",
        "launch_command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "pytorch_version": str(torch.__version__),
            "pytorch_cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
            "cudnn_version": int(torch.backends.cudnn.version()) if torch.backends.cudnn.version() else None,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ultralytics_version": ultralytics.__version__,
            "git_commit": git_commit(),
        },
        "experiment_manifest": {
            "path": str(manifest_path) if manifest_path else None,
            "sha256": sha256_file(manifest_path) if manifest_path else None,
        },
        "inputs": {
            "data_yaml": {"path": str(Path(args.data).resolve()), "sha256": sha256_file(args.data)},
            "model_yaml": {"path": str(Path(model_path).resolve()), "sha256": sha256_file(model_path)},
            "teacher_checkpoint": {
                "path": str(Path(args.teacher).resolve()) if args.teacher else None,
                "sha256": sha256_file(args.teacher) if args.teacher else None,
            },
            "split_manifests": {
                "train_images": data_manifest.get("train_images"),
                "val_images": data_manifest.get("val_images"),
                "train_sorted_basename_manifest_sha256": data_manifest.get(
                    "train_sorted_basename_manifest_sha256"
                ),
                "val_sorted_basename_manifest_sha256": data_manifest.get("val_sorted_basename_manifest_sha256"),
            },
        },
        "method": {
            "RepP3_enabled": "rep-p3" in Path(model_path).name.lower(),
            "CS_CWL_enabled": bool(ogdkd.use_class_weight and ogdkd.use_size_weight),
            "QG_KD_enabled": bool(ogdkd.use_ogdkd),
        },
        "training_overrides": overrides,
        "source_files": source_files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26n on KITTI with CW-SW loss or optional OG-DKD.")
    parser.add_argument(
        "--model",
        "--student",
        dest="model",
        default="ultralytics/cfg/models/26/yolo26n.yaml",
        help="Student model YAML/checkpoint. --student is kept as a backward-compatible alias.",
    )
    parser.add_argument(
        "--teacher-runs",
        "--teacher",
        dest="teacher",
        default=None,
        help="Trained YOLO26l teacher checkpoint. Required only when gap weighting or OG-DKD is enabled.",
    )
    parser.add_argument(
        "--teacher-sha256", default=None, help="Optional user-supplied teacher digest recorded in the run snapshot."
    )
    parser.add_argument(
        "--experiment-manifest", default=None, help="Optional user-provided metadata YAML recorded by hash for this run."
    )
    parser.add_argument("--data", default=str(DATA_ROOT / 'kitti_3cls.yaml'), help="Dataset YAML.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--project", default=str(OUTPUT_ROOT / 'training'))
    parser.add_argument("--name", default="experiment")
    parser.add_argument("--optimizer", default="MuSGD")
    parser.add_argument("--lr0", type=float, required=True, help="Initial learning rate for your experiment; no project default is provided.")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mosaic", type=float, required=True, help="Mosaic probability for your experiment; no project default is provided.")
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--class-weight-mode", choices=("none", "manual", "effective"), default="effective")
    parser.add_argument("--class-weights", nargs="+", default=["1.0", "1.25", "1.5"])
    parser.add_argument("--class-balance-beta", type=float, default=0.999)
    parser.add_argument("--class-weight-min", type=float, default=0.5)
    parser.add_argument("--class-weight-max", type=float, default=2.0)
    parser.add_argument("--use-class-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-size-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-gap-weight", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-ogdkd", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--beta-size", type=float, default=0.5)
    parser.add_argument("--tau-size", type=float, default=0.02)
    parser.add_argument("--size-weight-max", type=float, default=1.5)
    parser.add_argument("--gamma-gap", type=float, default=0.5)
    parser.add_argument("--gap-weight-max", type=float, default=1.5)
    parser.add_argument("--lambda-kd", "--qg-cls-coeff", dest="lambda_kd", type=float, default=0.3)
    parser.add_argument("--eta-box-kd", "--qg-box-relative-coeff", dest="eta_box_kd", type=float, default=1.0)
    parser.add_argument("--teacher-quality-thr", "--qg-tau", dest="teacher_quality_thr", type=float, default=0.5)
    parser.add_argument("--quality-margin", "--qg-delta", dest="quality_margin", type=float, default=0.05)
    parser.add_argument("--student-layers", default=None, help="Optional comma layer list, e.g. 16,19,22.")
    parser.add_argument("--teacher-layers", default=None, help="Optional comma layer list, e.g. 16,19,22.")
    parser.add_argument("--dry-run", action="store_true", help="Run one backward pass, print diagnostics, then exit.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--save-period", type=int, default=-1)
    return parser.parse_args()


def log_startup_config(args: argparse.Namespace, model_path: str, ogdkd: OGDKDConfig) -> None:
    """Print a compact run summary for both dry-run and training."""
    LOGGER.info(f"Mode: {ogdkd.mode_name}")
    LOGGER.info(f"model path: {model_path}")
    LOGGER.info(f"data path: {args.data}")
    LOGGER.info(f"epochs: {args.epochs}")
    LOGGER.info(f"batch: {args.batch}")
    LOGGER.info(f"optimizer: {args.optimizer}")
    LOGGER.info(f"pretrained: {bool(args.pretrained)}")
    LOGGER.info(f"RepP3_enabled: {'rep-p3' in Path(model_path).name.lower()}")
    LOGGER.info(f"CS_CWL_enabled: {bool(ogdkd.use_class_weight and ogdkd.use_size_weight)}")
    LOGGER.info(f"QG_KD_enabled: {bool(ogdkd.use_ogdkd)}")
    LOGGER.info(f"teacher_path: {ogdkd.teacher}")
    LOGGER.info(f"teacher_sha256: {ogdkd.teacher_sha256}")
    LOGGER.info(f"class_weight_mode: {ogdkd.class_weight_mode}")
    LOGGER.info(f"use_class_weight: {ogdkd.use_class_weight}")
    LOGGER.info(f"use_size_weight: {ogdkd.use_size_weight}")
    LOGGER.info(f"use_gap_weight: {ogdkd.use_gap_weight}")
    LOGGER.info(f"use_ogdkd: {ogdkd.use_ogdkd}")
    LOGGER.info(
        "class balance: "
        f"beta={ogdkd.class_balance_beta}, bounds=[{ogdkd.class_weight_min}, {ogdkd.class_weight_max}]"
    )
    LOGGER.info(
        "QG-KD gate/loss: "
        f"tau={ogdkd.teacher_quality_thr}, delta={ogdkd.quality_margin}, "
        f"cls_coeff={ogdkd.lambda_kd}, box_relative_coeff={ogdkd.eta_box_kd}"
    )
    LOGGER.info("YOLO26 third detection term: normalized LTRB distance L1 (reg_max=1), not traditional DFL.")


def main() -> None:
    args = parse_args()
    class_weights = parse_float_tuple(args.class_weights)
    class_weight_mode = resolve_class_weight_mode(args.class_weight_mode, args.use_class_weight)

    ogdkd = OGDKDConfig(
        teacher=args.teacher,
        student_layers=args.student_layers,
        teacher_layers=args.teacher_layers,
        class_weights=class_weights,
        class_weight_mode=class_weight_mode,
        class_balance_beta=args.class_balance_beta,
        class_weight_min=args.class_weight_min,
        class_weight_max=args.class_weight_max,
        use_class_weight=class_weight_mode != "none",
        use_size_weight=args.use_size_weight,
        use_gap_weight=args.use_gap_weight,
        use_ogdkd=args.use_ogdkd,
        beta_size=args.beta_size,
        tau_size=args.tau_size,
        size_weight_max=args.size_weight_max,
        gamma_gap=args.gamma_gap,
        gap_weight_max=args.gap_weight_max,
        lambda_kd=args.lambda_kd,
        eta_box_kd=args.eta_box_kd,
        teacher_quality_thr=args.teacher_quality_thr,
        quality_margin=args.quality_margin,
        teacher_sha256=args.teacher_sha256,
        experiment_manifest=args.experiment_manifest,
        dry_run=args.dry_run,
    )
    if ogdkd.teacher_guided and not ogdkd.teacher:
        raise ValueError("--teacher-runs/--teacher is required when --use-gap-weight or --use-ogdkd is enabled.")
    model_path = resolve_local_path(args.model)
    log_startup_config(args, model_path, ogdkd)
    overrides = {
        "model": model_path,
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "momentum": args.momentum,
        "pretrained": args.pretrained,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "amp": args.amp,
        "plots": args.plots,
        "val": args.val,
        "patience": args.patience,
        "save_period": args.save_period,
        "task": "detect",
        "compile": False,
    }
    provenance = build_provenance(args, model_path, overrides, ogdkd)
    trainer = OGDKDTrainer(overrides=overrides, ogdkd=ogdkd, provenance=provenance)
    trainer.train()


if __name__ == "__main__":
    main()
