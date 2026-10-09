# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Optimization-guided dynamic knowledge distillation utilities for YOLO detection training."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist, dist2bbox, make_anchors


def resolve_class_weight_mode(class_weight_mode: str, use_class_weight: bool) -> str:
    """Resolve legacy class-weight switches into the new explicit mode."""
    allowed = {"none", "manual", "effective"}
    if class_weight_mode not in allowed:
        raise ValueError(f"class_weight_mode must be one of {sorted(allowed)}, got {class_weight_mode!r}.")
    return class_weight_mode if use_class_weight else "none"


def count_class_instances_from_labels(labels: list[dict[str, Any]], nc: int) -> tuple[tuple[int, ...], int]:
    """Count train-set instances per class from YOLO dataset labels."""
    if nc <= 0:
        raise ValueError(f"Number of classes must be positive, got nc={nc}.")

    counts = torch.zeros(nc, dtype=torch.long)
    invalid = 0
    for label in labels or []:
        cls = label.get("cls") if isinstance(label, dict) else None
        if cls is None:
            continue
        cls_tensor = torch.as_tensor(cls).view(-1)
        if cls_tensor.numel() == 0:
            continue
        cls_tensor = cls_tensor.long()
        valid = (cls_tensor >= 0) & (cls_tensor < nc)
        invalid += int((~valid).sum().item())
        if valid.any():
            counts += torch.bincount(cls_tensor[valid], minlength=nc)[:nc]
    return tuple(int(x) for x in counts.tolist()), invalid


def effective_number_class_weights(
    class_counts: tuple[int, ...],
    beta: float = 0.999,
    weight_min: float = 0.5,
    weight_max: float = 2.0,
) -> tuple[tuple[float, ...], bool]:
    """Compute effective-number class-balanced weights normalized to mean 1, then clipped."""
    if not 0.0 <= beta < 1.0:
        raise ValueError(f"class_balance_beta must be in [0, 1), got {beta}.")
    if weight_min <= 0.0 or weight_max < weight_min:
        raise ValueError(
            f"class weight bounds must satisfy 0 < min <= max, got [{weight_min}, {weight_max}]."
        )

    counts = torch.tensor(class_counts, dtype=torch.float64)
    if counts.numel() == 0:
        return (), False

    weights = torch.ones_like(counts)
    positive = counts > 0
    if positive.any():
        beta_t = torch.tensor(beta, dtype=counts.dtype)
        effective_num = (1.0 - beta_t.pow(counts[positive])) / (1.0 - beta)
        weights[positive] = 1.0 / effective_num.clamp_min(1e-12)
        if (~positive).any():
            weights[~positive] = weights[positive].mean()

    weights = weights / weights.mean().clamp_min(1e-12)
    clipped = bool(((weights < weight_min) | (weights > weight_max)).any().item())
    weights = weights.clamp(weight_min, weight_max)
    return tuple(float(x) for x in weights.to(dtype=torch.float32).tolist()), clipped


@dataclass
class OGDKDSettings:
    """Runtime switches and hyperparameters for OG-DKD.

    The three positive-sample weights have separate switches so ablations can keep the same code path:
    class weights bias rare/hard KITTI classes, size weights emphasize small boxes, and gap weights emphasize anchors
    where the frozen teacher is measurably better than the student.
    """

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
    eps: float = 1e-6


@dataclass
class AnchorWeightPack:
    """Per-positive-anchor OG-DKD weights and quality diagnostics."""

    fg_b: torch.Tensor
    fg_a: torch.Tensor
    assigned_cls: torch.Tensor
    assigned_gt_boxes: torch.Tensor
    class_weight: torch.Tensor
    size_weight: torch.Tensor
    gap_weight: torch.Tensor
    box_weight: torch.Tensor
    cls_weight: torch.Tensor
    dist_weight: torch.Tensor
    kd_base_weight: torch.Tensor
    q_t: torch.Tensor
    q_s: torch.Tensor
    teacher_boxes: torch.Tensor | None
    student_boxes: torch.Tensor | None
    teacher_logits: torch.Tensor | None
    student_logits: torch.Tensor | None
    valid_teacher: torch.Tensor


class OGDKDBranchLoss(v8DetectionLoss):
    """One Detect branch loss with OG dynamic positive-anchor weighting and optional KD."""

    def __init__(
        self,
        model: torch.nn.Module,
        settings: OGDKDSettings,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
        branch_name: str = "one2many",
    ):
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.settings = settings
        self.branch_name = branch_name
        self.class_weights_tensor = torch.tensor(settings.class_weights, dtype=torch.float32, device=self.device)
        self.last_diag: dict[str, Any] = {}

    def set_class_weights(self, class_weights: tuple[float, ...]) -> None:
        """Update class weights after train-set label statistics are available."""
        self.class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)

    def forward_og(
        self,
        preds: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        teacher_preds: dict[str, torch.Tensor] | None = None,
        teacher_head: torch.nn.Module | None = None,
        enable_kd: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Return train loss vector, log items, KD train vector, KD log items, and diagnostics."""
        loss = torch.zeros(3, device=self.device)  # box, cls, normalized LTRB distance L1 for YOLO26 reg_max=1
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy in pixels
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy in stride units
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = target_scores.sum().clamp_min(1.0)

        teacher_pack, teacher_error = None, None
        needs_teacher = self.settings.use_gap_weight or (enable_kd and self.settings.use_ogdkd)
        if needs_teacher and teacher_preds is not None and teacher_head is not None:
            try:
                teacher_pack = self.decode_peer_predictions(teacher_preds, teacher_head, pred_scores, pred_bboxes)
            except (KeyError, TypeError, ValueError, FloatingPointError) as exc:
                teacher_error = str(exc)
                if self.training_requires_teacher(enable_kd):
                    raise

        weights = self.build_anchor_weights(
            fg_mask=fg_mask,
            target_gt_idx=target_gt_idx,
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            pred_scores=pred_scores,
            pred_bboxes=pred_bboxes,
            stride_tensor=stride_tensor,
            imgsz=imgsz,
            teacher_pack=teacher_pack,
        )

        cls_weight_map = torch.ones_like(target_scores[..., 0])
        if weights.fg_a.numel():
            cls_weight_map[weights.fg_b, weights.fg_a] = weights.cls_weight.to(dtype=cls_weight_map.dtype)
        loss[1] = (self.bce(pred_scores, target_scores.to(dtype)).sum(-1) * cls_weight_map).sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.weighted_bbox_loss(
                pred_distri=pred_distri,
                pred_bboxes=pred_bboxes,
                anchor_points=anchor_points,
                target_bboxes=target_bboxes / stride_tensor,
                target_scores=target_scores,
                target_scores_sum=target_scores_sum,
                fg_mask=fg_mask,
                imgsz=imgsz,
                stride_tensor=stride_tensor,
                box_weight=weights.box_weight,
                dist_weight=weights.dist_weight,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        cls_kd, box_kd, kd_diag = self.ogdkd_loss(weights, enable_kd=enable_kd, loss_ref=loss)
        kd_train_vec = torch.stack(
            (
                self.settings.lambda_kd * cls_kd,
                self.settings.lambda_kd * self.settings.eta_box_kd * box_kd,
            )
        ) * batch_size
        kd_items = torch.stack((cls_kd.detach(), box_kd.detach()))

        train_vec = loss * batch_size
        items = loss.detach()
        self.last_diag = self.build_diag(
            loss=loss,
            train_vec=train_vec,
            kd_train_vec=kd_train_vec,
            kd_items=kd_items,
            weights=weights,
            fg_mask=fg_mask,
            target_scores_sum=target_scores_sum,
            teacher_pack=teacher_pack,
            teacher_error=teacher_error,
            kd_diag=kd_diag,
            batch_size=batch_size,
            imgsz=imgsz,
        )
        return train_vec, items, kd_train_vec, kd_items, self.last_diag

    def training_requires_teacher(self, enable_kd: bool) -> bool:
        """Return whether a teacher decoding failure should stop the run."""
        return self.settings.use_gap_weight or (enable_kd and self.settings.use_ogdkd)

    def decode_peer_predictions(
        self,
        peer_preds: dict[str, torch.Tensor],
        peer_head: torch.nn.Module,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Decode teacher boxes and logits on the same flattened anchor layout as the student."""
        required = {"boxes", "scores", "feats"}
        missing = required.difference(peer_preds)
        if missing:
            raise KeyError(f"{self.branch_name} teacher predictions missing keys {sorted(missing)}")

        raw_boxes = peer_preds["boxes"]
        raw_scores = peer_preds["scores"]
        if raw_boxes.ndim != 3 or raw_scores.ndim != 3:
            raise ValueError(
                f"Unexpected {self.branch_name} teacher shapes: boxes={tuple(raw_boxes.shape)}, "
                f"scores={tuple(raw_scores.shape)}"
            )

        peer_distri = raw_boxes.permute(0, 2, 1).contiguous()
        peer_scores = raw_scores.permute(0, 2, 1).contiguous()
        peer_anchor_points, peer_stride_tensor = make_anchors(
            peer_preds["feats"], peer_head.stride.to(device=raw_boxes.device), 0.5
        )
        peer_bboxes = self.decode_distances(peer_anchor_points, peer_distri, int(peer_head.reg_max))

        if peer_scores.shape != pred_scores.shape:
            raise ValueError(
                f"Teacher/student score shape mismatch on {self.branch_name}: "
                f"{tuple(peer_scores.shape)} vs {tuple(pred_scores.shape)}"
            )
        if peer_bboxes.shape != pred_bboxes.shape:
            raise ValueError(
                f"Teacher/student box shape mismatch on {self.branch_name}: "
                f"{tuple(peer_bboxes.shape)} vs {tuple(pred_bboxes.shape)}"
            )
        if peer_stride_tensor.shape[0] != pred_bboxes.shape[1]:
            raise ValueError(
                f"Teacher stride count mismatch on {self.branch_name}: "
                f"{tuple(peer_stride_tensor.shape)} vs anchors={pred_bboxes.shape[1]}"
            )
        if not torch.isfinite(peer_bboxes).all() or not torch.isfinite(peer_scores).all():
            raise FloatingPointError(f"Teacher predictions contain NaN/Inf on {self.branch_name}.")
        return {"boxes": peer_bboxes, "scores": peer_scores, "stride_tensor": peer_stride_tensor}

    def decode_distances(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, reg_max: int) -> torch.Tensor:
        """Decode Detect distance predictions using the teacher's reg_max."""
        if reg_max > 1:
            b, a, c = pred_dist.shape
            proj = torch.arange(reg_max, dtype=pred_dist.dtype, device=pred_dist.device)
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(proj)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def build_anchor_weights(
        self,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        stride_tensor: torch.Tensor,
        imgsz: torch.Tensor,
        teacher_pack: dict[str, torch.Tensor] | None,
    ) -> AnchorWeightPack:
        """Build class, size and teacher-student gap weights for positive anchors."""
        fg_b, fg_a = fg_mask.nonzero(as_tuple=True)
        zero = pred_scores.new_zeros((0,))
        empty_long = fg_b.new_zeros((0,), dtype=torch.long)
        if fg_a.numel() == 0:
            return AnchorWeightPack(
                fg_b=fg_b,
                fg_a=fg_a,
                assigned_cls=empty_long,
                assigned_gt_boxes=pred_scores.new_zeros((0, 4)),
                class_weight=zero,
                size_weight=zero,
                gap_weight=zero,
                box_weight=zero,
                cls_weight=zero,
                dist_weight=zero,
                kd_base_weight=zero,
                q_t=zero,
                q_s=zero,
                teacher_boxes=None,
                student_boxes=None,
                teacher_logits=None,
                student_logits=None,
                valid_teacher=fg_b.new_zeros((0,), dtype=torch.bool),
            )

        gt_idx = target_gt_idx[fg_b, fg_a].long().clamp_min(0)
        assigned_cls = gt_labels[fg_b, gt_idx, 0].long().clamp_min(0)
        assigned_gt_boxes = gt_bboxes[fg_b, gt_idx].to(device=pred_scores.device, dtype=pred_scores.dtype)

        class_weight = torch.ones_like(fg_a, dtype=pred_scores.dtype, device=pred_scores.device)
        if self.settings.use_class_weight and self.class_weights_tensor.numel():
            class_weights = self.class_weights_tensor.to(device=pred_scores.device, dtype=pred_scores.dtype)
            valid_cls = assigned_cls < class_weights.numel()
            if valid_cls.any():
                class_weight[valid_cls] = class_weights[assigned_cls[valid_cls]]

        size_weight = torch.ones_like(class_weight)
        if self.settings.use_size_weight:
            gt_w = (assigned_gt_boxes[:, 2] - assigned_gt_boxes[:, 0]).clamp_min(0) / imgsz[1].clamp_min(
                self.settings.eps
            )
            gt_h = (assigned_gt_boxes[:, 3] - assigned_gt_boxes[:, 1]).clamp_min(0) / imgsz[0].clamp_min(
                self.settings.eps
            )
            area = (gt_w * gt_h).clamp_min(self.settings.eps)
            size_weight = 1.0 + self.settings.beta_size * torch.exp(-area / self.settings.tau_size)
            size_weight = size_weight.clamp(1.0, self.settings.size_weight_max)

        student_boxes = pred_bboxes[fg_b, fg_a] * stride_tensor[fg_a]
        student_logits = pred_scores[fg_b, fg_a]
        q_s = pred_scores.new_zeros(fg_a.shape)
        q_t = pred_scores.new_zeros(fg_a.shape)
        valid_teacher = torch.zeros_like(fg_a, dtype=torch.bool, device=pred_scores.device)
        teacher_boxes, teacher_logits = None, None

        gap_weight = torch.ones_like(class_weight)
        if teacher_pack is not None:
            teacher_boxes = teacher_pack["boxes"][fg_b, fg_a] * teacher_pack["stride_tensor"][fg_a]
            teacher_logits = teacher_pack["scores"][fg_b, fg_a]
            valid_teacher = self.valid_xyxy(student_boxes) & self.valid_xyxy(teacher_boxes) & self.valid_xyxy(
                assigned_gt_boxes
            )
            cls_idx = assigned_cls.clamp_max(pred_scores.shape[-1] - 1)
            student_conf = student_logits.sigmoid().gather(1, cls_idx.view(-1, 1)).squeeze(1)
            teacher_conf = teacher_logits.sigmoid().gather(1, cls_idx.view(-1, 1)).squeeze(1)
            student_iou = self.box_iou_aligned(student_boxes, assigned_gt_boxes)
            teacher_iou = self.box_iou_aligned(teacher_boxes, assigned_gt_boxes)
            q_s = torch.where(valid_teacher, student_conf * student_iou, q_s)
            q_t = torch.where(valid_teacher, teacher_conf * teacher_iou, q_t)
            if self.settings.use_gap_weight:
                gap = (q_t - q_s).clamp_min(0)
                gap_weight = (1.0 + self.settings.gamma_gap * gap).clamp(1.0, self.settings.gap_weight_max)

        box_weight = (class_weight * size_weight * gap_weight).clamp(1.0, 2.0)
        cls_weight = (class_weight * gap_weight).clamp(1.0, 1.8)
        dist_weight = (size_weight * gap_weight).clamp(1.0, 1.8)
        kd_base_weight = class_weight * size_weight
        return AnchorWeightPack(
            fg_b=fg_b,
            fg_a=fg_a,
            assigned_cls=assigned_cls,
            assigned_gt_boxes=assigned_gt_boxes,
            class_weight=class_weight.detach(),
            size_weight=size_weight.detach(),
            gap_weight=gap_weight.detach(),
            box_weight=box_weight.detach(),
            cls_weight=cls_weight.detach(),
            dist_weight=dist_weight.detach(),
            kd_base_weight=kd_base_weight.detach(),
            q_t=q_t.detach(),
            q_s=q_s.detach(),
            teacher_boxes=teacher_boxes.detach() if teacher_boxes is not None else None,
            student_boxes=student_boxes,
            teacher_logits=teacher_logits.detach() if teacher_logits is not None else None,
            student_logits=student_logits,
            valid_teacher=valid_teacher.detach(),
        )

    def weighted_bbox_loss(
        self,
        pred_distri: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride_tensor: torch.Tensor,
        box_weight: torch.Tensor,
        dist_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute YOLO box and distribution/distance terms with separate OG weights.

        This keeps the original YOLO assignment and reductions, changing only positive-anchor weights. When reg_max=1
        (YOLO26n in this repo), the local YOLO loss uses normalized LTRB distance L1, not Distribution Focal Loss.
        ``dist_weight`` is therefore the public and diagnostic name used by the controlled experiment.
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        box_weight = box_weight.to(dtype=weight.dtype).unsqueeze(-1)
        dist_weight = dist_weight.to(dtype=weight.dtype).unsqueeze(-1)

        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight * box_weight).sum() / target_scores_sum

        if self.bbox_loss.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.bbox_loss.dfl_loss.reg_max - 1)
            loss_dfl = self.bbox_loss.dfl_loss(
                pred_distri[fg_mask].view(-1, self.bbox_loss.dfl_loss.reg_max), target_ltrb[fg_mask]
            )
            loss_dfl = (loss_dfl * weight * dist_weight).sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride_tensor
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_distri * stride_tensor
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True)
            loss_dfl = (loss_dfl * weight * dist_weight).sum() / target_scores_sum
        return loss_iou, loss_dfl

    def ogdkd_loss(
        self, weights: AnchorWeightPack, enable_kd: bool, loss_ref: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Compute teacher-guided classification KL and box IoU distillation."""
        zero = loss_ref.new_zeros(()) if loss_ref is not None else weights.kd_base_weight.new_zeros(())

        def finite_mean_or_zero(x: torch.Tensor) -> torch.Tensor:
            if not x.numel():
                return zero.detach()
            x = x.detach()
            finite = torch.isfinite(x)
            return x[finite].mean() if bool(finite.any().detach().cpu()) else zero.detach()

        def finish(cls_kd: torch.Tensor, box_kd: torch.Tensor, valid_pair_count: int) -> tuple[torch.Tensor, torch.Tensor]:
            """Return finite KD scalars; epoch-level pair logging is handled by the trainer."""
            return cls_kd, box_kd

        kd_diag: dict[str, Any] = {
            "kd_candidate_count": 0,
            "kd_gate_pass_count": 0,
            "kd_positive_count": 0,
            "kd_raw_pairs_finite": True,
            "kd_reduction_finite": True,
            "mean_kd_weight": zero.detach(),
            "mean_q_t": zero.detach(),
            "mean_q_s": zero.detach(),
        }
        if not enable_kd or not self.settings.use_ogdkd or weights.fg_a.numel() == 0:
            cls_kd, box_kd = finish(zero, zero, 0)
            return cls_kd, box_kd, kd_diag
        if weights.teacher_logits is None or weights.teacher_boxes is None or weights.student_boxes is None:
            cls_kd, box_kd = finish(zero, zero, 0)
            return cls_kd, box_kd, kd_diag

        kd_mask = (
            weights.valid_teacher
            & torch.isfinite(weights.q_t)
            & torch.isfinite(weights.q_s)
            & torch.isfinite(weights.kd_base_weight)
            & (weights.kd_base_weight > 0)
            & torch.isfinite(weights.student_logits).all(dim=-1)
            & torch.isfinite(weights.teacher_logits).all(dim=-1)
            & self.valid_xyxy(weights.student_boxes)
            & self.valid_xyxy(weights.teacher_boxes)
            & (weights.q_t > self.settings.teacher_quality_thr)
            & (weights.q_t > weights.q_s + self.settings.quality_margin)
        )
        kd_diag.update(
            {
                "kd_candidate_count": int(weights.fg_a.numel()),
                "mean_q_t": finite_mean_or_zero(weights.q_t),
                "mean_q_s": finite_mean_or_zero(weights.q_s),
            }
        )
        if not bool(kd_mask.any().detach().cpu()):
            cls_kd, box_kd = finish(zero, zero, 0)
            return cls_kd, box_kd, kd_diag

        kd_diag["kd_gate_pass_count"] = int(kd_mask.sum().detach().cpu())

        # YOLO class heads are independent logits, but the requested first version uses KL on the aligned class vector.
        student_logits = weights.student_logits[kd_mask]
        teacher_logits = weights.teacher_logits[kd_mask]
        student_boxes = weights.student_boxes[kd_mask]
        teacher_boxes = weights.teacher_boxes[kd_mask]
        kd_weight = weights.kd_base_weight[kd_mask].to(dtype=zero.dtype)
        cls_kd = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="none",
        ).sum(-1)
        box_kd = 1.0 - self.box_iou_aligned(student_boxes, teacher_boxes)
        raw_finite_pair = torch.isfinite(cls_kd) & torch.isfinite(box_kd) & torch.isfinite(kd_weight)
        kd_diag["kd_raw_pairs_finite"] = bool(raw_finite_pair.all().detach().cpu())
        finite_pair = raw_finite_pair & (kd_weight > 0)
        if not bool(finite_pair.any().detach().cpu()):
            cls_kd, box_kd = finish(zero, zero, 0)
            return cls_kd, box_kd, kd_diag

        cls_kd = cls_kd[finite_pair].to(dtype=zero.dtype)
        box_kd = box_kd[finite_pair].to(dtype=zero.dtype)
        kd_weight = kd_weight[finite_pair]
        kd_sum = kd_weight.sum()
        valid_pair_count = int(finite_pair.sum().detach().cpu())
        kd_diag["kd_positive_count"] = valid_pair_count
        if not bool(torch.isfinite(kd_sum).detach().cpu()) or bool((kd_sum <= 0).detach().cpu()):
            kd_diag["kd_reduction_finite"] = False
            cls_kd, box_kd = finish(zero, zero, 0)
            return cls_kd, box_kd, kd_diag

        cls_kd = (cls_kd * kd_weight).sum() / kd_sum.clamp_min(self.settings.eps)
        box_kd = (box_kd * kd_weight).sum() / kd_sum.clamp_min(self.settings.eps)
        reduction_finite = bool(torch.isfinite(cls_kd).detach().cpu()) and bool(
            torch.isfinite(box_kd).detach().cpu()
        )
        kd_diag["kd_reduction_finite"] = reduction_finite
        if not bool(torch.isfinite(cls_kd).detach().cpu()):
            cls_kd = zero
        if not bool(torch.isfinite(box_kd).detach().cpu()):
            box_kd = zero
        kd_diag["mean_kd_weight"] = kd_weight.mean().detach() if valid_pair_count else zero.detach()
        cls_kd, box_kd = finish(cls_kd, box_kd, valid_pair_count)
        return cls_kd, box_kd, kd_diag

    def build_diag(
        self,
        loss: torch.Tensor,
        train_vec: torch.Tensor,
        kd_train_vec: torch.Tensor,
        kd_items: torch.Tensor,
        weights: AnchorWeightPack,
        fg_mask: torch.Tensor,
        target_scores_sum: torch.Tensor,
        teacher_pack: dict[str, torch.Tensor] | None,
        teacher_error: str | None,
        kd_diag: dict[str, Any],
        batch_size: int,
        imgsz: torch.Tensor,
    ) -> dict[str, Any]:
        """Collect scalar diagnostics for dry-run and shape checks."""

        def mean_or_zero(x: torch.Tensor) -> torch.Tensor:
            if not x.numel():
                return loss.new_zeros(())
            x = x.detach()
            finite = torch.isfinite(x)
            return x[finite].mean() if bool(finite.any().detach().cpu()) else loss.new_zeros(())

        def stats_or_zero(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if not x.numel():
                zero = loss.new_zeros(())
                return zero, zero, zero
            x = x.detach()
            finite = torch.isfinite(x)
            if not bool(finite.any().detach().cpu()):
                zero = loss.new_zeros(())
                return zero, zero, zero
            x = x[finite]
            return x.mean(), x.min(), x.max()

        box_mean, box_min, box_max = stats_or_zero(weights.box_weight)
        cls_mean, cls_min, cls_max = stats_or_zero(weights.cls_weight)
        dist_mean, dist_min, dist_max = stats_or_zero(weights.dist_weight)

        finite_checks = {
            "det_loss": torch.isfinite(loss).all(),
            "kd_train_vec": torch.isfinite(kd_train_vec).all(),
            "kd_items": torch.isfinite(kd_items).all(),
            "weights": all(
                torch.isfinite(x).all()
                for x in (
                    weights.class_weight,
                    weights.size_weight,
                    weights.gap_weight,
                    weights.box_weight,
                    weights.cls_weight,
                    weights.dist_weight,
                    weights.kd_base_weight,
                    weights.q_t,
                    weights.q_s,
                )
            ),
            "kd_raw_pairs": bool(kd_diag.get("kd_raw_pairs_finite", True)),
            "kd_reduction": bool(kd_diag.get("kd_reduction_finite", True)),
        }
        return {
            "branch": self.branch_name,
            "batch_size": batch_size,
            "imgsz": tuple(int(x) for x in imgsz.detach().cpu().tolist()),
            "fg_count": int(fg_mask.sum().detach().cpu()),
            "target_scores_sum": target_scores_sum.detach(),
            "teacher_available": teacher_pack is not None,
            "teacher_error": teacher_error,
            "box_loss": loss[0].detach(),
            "cls_loss": loss[1].detach(),
            "dist_l1_loss": loss[2].detach(),
            "det_train_scaled": train_vec.detach().sum(),
            "kd_train_scaled": kd_train_vec.detach().sum(),
            "cls_kd": kd_items[0].detach(),
            "box_kd": kd_items[1].detach(),
            "mean_class_weight": mean_or_zero(weights.class_weight),
            "mean_size_weight": mean_or_zero(weights.size_weight),
            "mean_gap_weight": mean_or_zero(weights.gap_weight),
            "mean_box_weight": box_mean,
            "min_box_weight": box_min,
            "max_box_weight": box_max,
            "mean_cls_weight": cls_mean,
            "min_cls_weight": cls_min,
            "max_cls_weight": cls_max,
            "mean_dist_weight": dist_mean,
            "min_dist_weight": dist_min,
            "max_dist_weight": dist_max,
            "mean_q_t": mean_or_zero(weights.q_t),
            "mean_q_s": mean_or_zero(weights.q_s),
            "valid_teacher_count": int(weights.valid_teacher.sum().detach().cpu()),
            "finite_checks": finite_checks,
            **kd_diag,
        }

    @staticmethod
    def valid_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Return a mask for finite xyxy boxes with positive width and height."""
        eps = torch.finfo(boxes.dtype).eps if boxes.dtype.is_floating_point else 1e-6
        finite = torch.isfinite(boxes).all(dim=-1)
        return finite & (boxes[..., 2] > boxes[..., 0] + eps) & (boxes[..., 3] > boxes[..., 1] + eps)

    def box_iou_aligned(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute aligned IoU for xyxy box pairs in FP32, including under AMP."""
        with torch.autocast(device_type=boxes1.device.type, enabled=False):
            boxes1 = boxes1.float()
            boxes2 = boxes2.float()
            lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
            rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
            wh = (rb - lt).clamp_min(0)
            inter = wh[:, 0] * wh[:, 1]
            area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
            area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
            return inter / (area1 + area2 - inter + self.settings.eps)


class OGDKDLoss:
    """YOLO detection criterion wrapper for OG-DKD.

    For YOLO26 end-to-end heads, both one2many and one2one detection branches receive OG detection weights. KD is
    applied only once on the one2many branch to avoid double-counting teacher guidance.
    """

    def __init__(self, model: torch.nn.Module, settings: OGDKDSettings):
        self.ensure_loss_args(model)
        self.settings = settings
        self.end2end = bool(getattr(model, "end2end", False))
        self.last_diag: dict[str, Any] = {}
        self._loss_check_printed = False
        if self.end2end:
            self.one2many = OGDKDBranchLoss(model, settings, tal_topk=10, branch_name="one2many")
            self.one2one = OGDKDBranchLoss(model, settings, tal_topk=7, tal_topk2=1, branch_name="one2one")
            self.updates = 0
            self.total = 1.0
            self.o2m = 0.8
            self.o2o = self.total - self.o2m
            self.o2m_copy = self.o2m
            self.final_o2m = 0.1
        else:
            self.one2many = OGDKDBranchLoss(model, settings, tal_topk=10, branch_name="detect")

    def __call__(
        self,
        preds: Any,
        batch: dict[str, torch.Tensor],
        teacher_output: Any | None = None,
        teacher_head: torch.nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        preds = self.parse_output(preds)
        teacher_preds = self.parse_output(teacher_output) if teacher_output is not None else None
        if self.end2end:
            one2many = preds["one2many"]
            one2one = preds["one2one"]
            teacher_one2many = self.select_teacher_branch(teacher_preds, "one2many")
            teacher_one2one = self.select_teacher_branch(teacher_preds, "one2one") or teacher_one2many
            o2m_loss, o2m_items, kd_vec, kd_items, o2m_diag = self.one2many.forward_og(
                one2many, batch, teacher_one2many, teacher_head, enable_kd=True
            )
            o2o_loss, o2o_items, _, _, o2o_diag = self.one2one.forward_og(
                one2one, batch, teacher_one2one, teacher_head, enable_kd=False
            )
            det_loss = o2m_loss * self.o2m + o2o_loss * self.o2o
            det_items = o2m_items * self.o2m + o2o_items * self.o2o
            loss_vec = torch.cat((det_loss, kd_vec))
            items = torch.cat((det_items, kd_items))
            self.last_diag = {
                "end2end": True,
                "o2m": self.o2m,
                "o2o": self.o2o,
                "one2many": o2m_diag,
                "one2one": o2o_diag,
                "loss_total": loss_vec.detach().sum(),
                "finite_checks": {
                    "loss_vec": torch.isfinite(loss_vec).all(),
                    "items": torch.isfinite(items).all(),
                    "one2many": self.all_checks_pass(o2m_diag["finite_checks"]),
                    "one2one": self.all_checks_pass(o2o_diag["finite_checks"]),
                },
            }
            self.print_loss_check_once()
            return self.format_return(loss_vec, items)

        teacher_branch = self.select_teacher_branch(teacher_preds, "one2many")
        det_loss, det_items, kd_vec, kd_items, diag = self.one2many.forward_og(
            preds, batch, teacher_branch, teacher_head, enable_kd=True
        )
        loss_vec = torch.cat((det_loss, kd_vec))
        items = torch.cat((det_items, kd_items))
        self.last_diag = {
            "end2end": False,
            "one2many": diag,
            "loss_total": loss_vec.detach().sum(),
            "finite_checks": {
                "loss_vec": torch.isfinite(loss_vec).all(),
                "items": torch.isfinite(items).all(),
                "one2many": self.all_checks_pass(diag["finite_checks"]),
            },
        }
        self.print_loss_check_once()
        return self.format_return(loss_vec, items)

    @staticmethod
    def parse_output(preds: Any) -> Any:
        """Select train prediction dict from Ultralytics train/eval return values."""
        if isinstance(preds, (tuple, list)) and len(preds) > 1 and isinstance(preds[1], dict):
            return preds[1]
        return preds

    @staticmethod
    def ensure_loss_args(model: torch.nn.Module) -> None:
        """Provide minimal loss hyperparameters for standalone layer-check scripts."""
        defaults = {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 1}
        if not hasattr(model, "args"):
            model.args = SimpleNamespace(**defaults)
            return
        if isinstance(model.args, dict):
            merged = {**defaults, **model.args}
            model.args = SimpleNamespace(**merged)
            return
        for name, value in defaults.items():
            if not hasattr(model.args, name):
                setattr(model.args, name, value)

    @staticmethod
    def select_teacher_branch(preds: Any, branch: str) -> dict[str, torch.Tensor] | None:
        """Select a teacher Detect branch, falling back to one2many for non-E2E teachers."""
        if preds is None:
            return None
        if isinstance(preds, dict) and branch in preds:
            return preds[branch]
        if isinstance(preds, dict) and "one2many" in preds:
            return preds["one2many"]
        return preds if isinstance(preds, dict) else None

    @staticmethod
    def all_checks_pass(checks: dict[str, Any]) -> bool:
        """Return True when every diagnostic check is true, accepting tensors or bools."""
        passed = []
        for value in checks.values():
            passed.append(bool(value.item()) if isinstance(value, torch.Tensor) else bool(value))
        return all(passed)

    @staticmethod
    def format_return(loss_vec: torch.Tensor, items: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep validation loss_items compatible with Ultralytics DetectionValidator.

        Training runs with gradients enabled and may log 5 items (box, cls, dfl, cls_kd, box_kd). Validation is executed
        under no_grad/inference_mode and DetectionValidator expects exactly 3 items, so only the detection items are
        returned there.
        """
        if torch.is_grad_enabled():
            return loss_vec, items
        return loss_vec[:3], items[:3]

    def set_class_balance(
        self,
        class_weights: tuple[float, ...],
        class_counts: tuple[int, ...],
        class_weight_mode: str,
    ) -> None:
        """Update settings and branch tensors with resolved train-set class balance."""
        self.settings.class_weights = tuple(float(x) for x in class_weights)
        self.settings.class_counts = tuple(int(x) for x in class_counts)
        self.settings.class_weight_mode = class_weight_mode
        self.settings.use_class_weight = class_weight_mode != "none"
        self.one2many.set_class_weights(self.settings.class_weights)
        if self.end2end:
            self.one2one.set_class_weights(self.settings.class_weights)

    def print_loss_check_once(self) -> None:
        """Print a one-time runtime proof that the custom loss path is active."""
        if self._loss_check_printed:
            return
        print("[LOSS-CHECK] Custom Auto-CBSW / OG-DKD loss is ACTIVE.")
        print(f"[LOSS-CHECK] class_weight_mode = {self.settings.class_weight_mode}")
        print(f"[LOSS-CHECK] use_class_weight = {self.settings.use_class_weight}")
        print(f"[LOSS-CHECK] use_size_weight  = {self.settings.use_size_weight}")
        print(f"[LOSS-CHECK] use_gap_weight   = {self.settings.use_gap_weight}")
        print(f"[LOSS-CHECK] use_ogdkd        = {self.settings.use_ogdkd}")
        print(f"[LOSS-CHECK] train class_counts = {self.settings.class_counts}")
        print(f"[LOSS-CHECK] class_weights      = {self.settings.class_weights}")
        print("[LOSS-CHECK] size weight implementation = existing")

        branch = self.last_diag.get("one2many") or {}
        if not branch and "detect" in self.last_diag:
            branch = self.last_diag["detect"]

        def scalar(value: Any) -> float:
            if isinstance(value, torch.Tensor):
                return float(value.detach().cpu())
            return float(value)

        if int(branch.get("fg_count") or 0) <= 0:
            print("[LOSS-CHECK] no positive samples in this batch, skip weight stats.")
        else:
            print(
                "[LOSS-CHECK] w_box mean/min/max: "
                f"{scalar(branch['mean_box_weight']):.4f} / "
                f"{scalar(branch['min_box_weight']):.4f} / "
                f"{scalar(branch['max_box_weight']):.4f}"
            )
            print(
                "[LOSS-CHECK] w_cls_gap mean/min/max: "
                f"{scalar(branch['mean_cls_weight']):.4f} / "
                f"{scalar(branch['min_cls_weight']):.4f} / "
                f"{scalar(branch['max_cls_weight']):.4f}"
            )
            print(
                "[LOSS-CHECK] w_dist_l1 mean/min/max: "
                f"{scalar(branch['mean_dist_weight']):.4f} / "
                f"{scalar(branch['min_dist_weight']):.4f} / "
                f"{scalar(branch['max_dist_weight']):.4f}"
            )
        self._loss_check_printed = True

    def update(self) -> None:
        """Match the local E2ELoss one2many/one2one decay schedule."""
        if not self.end2end:
            return
        self.updates += 1
        self.o2m = self.decay(self.updates)
        self.o2o = max(self.total - self.o2m, 0)

    def decay(self, x: int) -> float:
        """Decay one2many weight across epochs, mirroring E2ELoss."""
        epochs = max(getattr(self.one2one.hyp, "epochs", 1) - 1, 1)
        return max(1 - x / epochs, 0) * (self.o2m_copy - self.final_o2m) + self.final_o2m

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        """Return extra train-time parameters. OG-DKD has none in this first version."""
        return []

    def extra_repr(self) -> str:
        """Return a compact readable settings summary."""
        return (
            f"use_class_weight={self.settings.use_class_weight}, use_size_weight={self.settings.use_size_weight}, "
            f"use_gap_weight={self.settings.use_gap_weight}, use_ogdkd={self.settings.use_ogdkd}, "
            f"lambda_kd={self.settings.lambda_kd}"
        )
