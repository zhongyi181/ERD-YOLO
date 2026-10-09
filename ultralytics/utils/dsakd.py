# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Directional and scale-aware distillation utilities for YOLO detection training."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn

from ultralytics.utils.tal import dist2bbox, make_anchors
from ultralytics.utils.torch_utils import unwrap_model


def parse_layer_indices(value: str | Iterable[int] | None) -> list[int] | None:
    """Parse comma-separated layer indices for feature distillation hooks."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def feature_layers_from_detect(model: nn.Module) -> list[int]:
    """Return the feature-source layers used by the final Detect head."""
    module = unwrap_model(model)
    if not hasattr(module, "model") or not len(module.model):
        raise AttributeError("Model does not expose a top-level .model sequence.")
    head = module.model[-1]
    layers = getattr(head, "f", None)
    if isinstance(layers, int):
        return [layers]
    if isinstance(layers, (list, tuple)):
        return [int(x) for x in layers]
    raise AttributeError("Final head does not expose Detect source layers through .f.")


def resolve_distill_layers(model: nn.Module, layers: str | Iterable[int] | None = None) -> list[int]:
    """Resolve explicit distillation layers or fall back to Detect(P3, P4, P5) input layers."""
    parsed = parse_layer_indices(layers)
    return parsed if parsed is not None else feature_layers_from_detect(model)


def first_tensor(output: Any) -> torch.Tensor | None:
    """Return the first BCHW tensor contained in a nested module output."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for value in output.values():
            tensor = first_tensor(value)
            if tensor is not None:
                return tensor
    if isinstance(output, (list, tuple)):
        for value in output:
            tensor = first_tensor(value)
            if tensor is not None:
                return tensor
    return None


def output_shape(output: Any) -> Any:
    """Convert nested module outputs to lightweight shape metadata."""
    if isinstance(output, torch.Tensor):
        return tuple(output.shape)
    if isinstance(output, dict):
        return {k: output_shape(v) for k, v in output.items()}
    if isinstance(output, (list, tuple)):
        return [output_shape(v) for v in output]
    return type(output).__name__


class FeatureHookStore:
    """Store outputs from selected top-level YOLO layers."""

    def __init__(self, model: nn.Module, layers: Iterable[int]):
        self.model = unwrap_model(model)
        self.layers = [int(x) for x in layers]
        self.outputs: dict[int, torch.Tensor] = {}
        self.handles = []
        for i in self.layers:
            self.handles.append(self.model.model[i].register_forward_hook(self._make_hook(i)))

    def _make_hook(self, index: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = first_tensor(output)
            if tensor is None:
                raise TypeError(f"Layer {index} did not produce a tensor output for DSAKD.")
            self.outputs[index] = tensor

        return hook

    def clear(self) -> None:
        """Clear stored feature references."""
        self.outputs.clear()

    def get(self, layers: Iterable[int] | None = None) -> list[torch.Tensor]:
        """Return stored tensors in layer order."""
        layers = self.layers if layers is None else [int(x) for x in layers]
        missing = [i for i in layers if i not in self.outputs]
        if missing:
            raise RuntimeError(f"Missing hooked feature outputs for layers {missing}.")
        return [self.outputs[i] for i in layers]

    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()


def collect_top_level_shapes(model: nn.Module, imgsz: int = 640, device: torch.device | str = "cpu") -> dict[int, Any]:
    """Run a dummy forward and collect top-level layer output shapes."""
    module = unwrap_model(model)
    device = torch.device(device)
    was_training = module.training
    shapes: dict[int, Any] = {}
    handles = []

    def make_hook(index: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            shapes[index] = output_shape(output)

        return hook

    for i, layer in enumerate(module.model):
        handles.append(layer.register_forward_hook(make_hook(i)))

    try:
        module.eval()
        x = torch.zeros(1, 3, imgsz, imgsz, device=device)
        with torch.no_grad():
            module(x)
    finally:
        for handle in handles:
            handle.remove()
        module.train(was_training)
    return shapes


def selected_feature_shapes(
    model: nn.Module, layers: Iterable[int], imgsz: int = 640, device: torch.device | str = "cpu"
) -> dict[int, tuple[int, ...]]:
    """Collect tensor shapes for selected feature layers."""
    module = unwrap_model(model)
    device = torch.device(device)
    layers = [int(x) for x in layers]
    was_training = module.training
    store = FeatureHookStore(module, layers)
    try:
        module.eval()
        x = torch.zeros(1, 3, imgsz, imgsz, device=device)
        with torch.no_grad():
            module(x)
        return {i: tuple(t.shape) for i, t in zip(layers, store.get(layers))}
    finally:
        store.remove()
        module.train(was_training)


def spatial_candidates(
    shapes: dict[int, Any], targets: Iterable[tuple[int, int]] = ((80, 80), (40, 40), (20, 20))
) -> dict[tuple[int, int], list[tuple[int, tuple[int, ...]]]]:
    """Find tensor-producing layers with spatial sizes matching target P3/P4/P5 maps."""
    targets = {tuple(x) for x in targets}
    found = {x: [] for x in targets}
    for index, shape in shapes.items():
        if isinstance(shape, tuple) and len(shape) == 4:
            hw = (int(shape[-2]), int(shape[-1]))
            if hw in found:
                found[hw].append((index, shape))
    return found


class DSAKDLoss(nn.Module):
    """Directional and scale-aware feature distillation loss.

    Student features are adapted to teacher channels through per-level 1x1 convolutions. The module is intended for
    training only and is kept outside the student model so exported checkpoints remain ordinary YOLO models.
    """

    def __init__(
        self,
        student_channels: Iterable[int],
        teacher_channels: Iterable[int],
        lambda_fg: float = 0.1,
        lambda_dir: float = 0.05,
        lambda_loc: float = 0.05,
        lambda_score: float = 0.0,
        teacher_score_thr: float = 0.25,
        teacher_iou_thr: float = 0.50,
        class_weights: Iterable[float] = (1.0, 1.5, 1.5),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.lambda_fg = float(lambda_fg)
        self.lambda_dir = float(lambda_dir)
        self.lambda_loc = float(lambda_loc)
        self.lambda_score = float(lambda_score)
        self.teacher_score_thr = float(teacher_score_thr)
        self.teacher_iou_thr = float(teacher_iou_thr)
        self.eps = float(eps)
        self.last_mask_stats: list[dict[str, float | int | tuple[int, ...]]] = []
        self.last_fd_layer_losses: list[float] = []
        adapters = []
        for sc, tc in zip(student_channels, teacher_channels):
            adapters.append(nn.Identity() if int(sc) == int(tc) else nn.Conv2d(int(sc), int(tc), 1, bias=False))
        self.adapters = nn.ModuleList(adapters)
        self.register_buffer("class_weights", torch.tensor(list(class_weights), dtype=torch.float32), persistent=False)

    def forward(
        self, student_feats: list[torch.Tensor], teacher_feats: list[torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return weighted distillation loss vector plus raw foreground and directional losses."""
        if len(student_feats) != len(teacher_feats) or len(student_feats) != len(self.adapters):
            raise ValueError("Student features, teacher features and adapters must have the same length.")

        fg_total = student_feats[0].new_zeros(())
        dir_total = student_feats[0].new_zeros(())
        valid_layers = 0
        self.last_mask_stats = []

        for student_feat, teacher_feat, adapter in zip(student_feats, teacher_feats, self.adapters):
            student_aligned = adapter(student_feat)
            teacher_feat = teacher_feat.detach().to(dtype=student_aligned.dtype)
            if student_aligned.shape != teacher_feat.shape:
                raise ValueError(
                    "Aligned student feature shape does not match teacher feature shape: "
                    f"{tuple(student_aligned.shape)} vs {tuple(teacher_feat.shape)}"
                )

            mask = self.build_mask(batch, student_aligned.shape[-2], student_aligned.shape[-1], student_aligned)
            mask_sum = mask.sum()
            positive = mask > 0
            self.last_mask_stats.append(
                {
                    "shape": tuple(mask.shape),
                    "min": float(mask.detach().min().cpu()),
                    "max": float(mask.detach().max().cpu()),
                    "mean": float(mask.detach().mean().cpu()),
                    "foreground_pixels": int(positive.sum().detach().cpu()),
                }
            )
            if mask_sum <= 0:
                continue

            channels = student_aligned.shape[1]
            diff = student_aligned - teacher_feat
            fg_loss = (diff.square() * mask).sum() / (mask_sum * channels + self.eps)

            mask_h_sum = mask.sum(dim=3, keepdim=True)
            valid_h = mask_h_sum.gt(0).to(dtype=student_aligned.dtype)
            student_h = (student_aligned * mask).sum(dim=3, keepdim=True) / mask_h_sum.clamp_min(self.eps)
            teacher_h = (teacher_feat * mask).sum(dim=3, keepdim=True) / mask_h_sum.clamp_min(self.eps)
            h_loss = (torch.abs(student_h - teacher_h) * valid_h).sum() / (valid_h.sum() * channels + self.eps)

            mask_w_sum = mask.sum(dim=2, keepdim=True)
            valid_w = mask_w_sum.gt(0).to(dtype=student_aligned.dtype)
            student_w = (student_aligned * mask).sum(dim=2, keepdim=True) / mask_w_sum.clamp_min(self.eps)
            teacher_w = (teacher_feat * mask).sum(dim=2, keepdim=True) / mask_w_sum.clamp_min(self.eps)
            w_loss = (torch.abs(student_w - teacher_w) * valid_w).sum() / (valid_w.sum() * channels + self.eps)

            fg_total = fg_total + fg_loss
            dir_total = dir_total + h_loss + w_loss
            valid_layers += 1

        if valid_layers == 0:
            weighted = torch.stack((fg_total, dir_total))
            return weighted, fg_total.detach(), dir_total.detach(), weighted.detach()

        batch_size = student_feats[0].shape[0]
        weighted = torch.stack((self.lambda_fg * fg_total, self.lambda_dir * dir_total)) * batch_size
        raw_items = torch.stack((fg_total.detach(), dir_total.detach()))
        weighted_items = torch.stack(
            ((self.lambda_fg * fg_total).detach(), (self.lambda_dir * dir_total).detach())
        )
        return weighted, raw_items[0], raw_items[1], weighted_items

    def forward_vanilla(
        self, student_feats: list[torch.Tensor], teacher_feats: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return vanilla full-map feature distillation loss.

        This baseline intentionally avoids foreground masks, class/scale weights, and directional pooling. It averages
        full-feature MSE over the selected P3/P4/P5 layers, then scales the train loss by batch size to match YOLO loss
        reduction style.
        """
        if len(student_feats) != len(teacher_feats) or len(student_feats) != len(self.adapters):
            raise ValueError("Student features, teacher features and adapters must have the same length.")

        fd_total = student_feats[0].new_zeros(())
        self.last_mask_stats = []
        self.last_fd_layer_losses = []
        for student_feat, teacher_feat, adapter in zip(student_feats, teacher_feats, self.adapters):
            student_aligned = adapter(student_feat)
            teacher_feat = teacher_feat.detach().to(dtype=student_aligned.dtype)
            if student_aligned.shape != teacher_feat.shape:
                raise ValueError(
                    "Aligned student feature shape does not match teacher feature shape: "
                    f"{tuple(student_aligned.shape)} vs {tuple(teacher_feat.shape)}"
                )
            layer_loss = (student_aligned - teacher_feat).square().mean()
            fd_total = fd_total + layer_loss
            self.last_fd_layer_losses.append(float(layer_loss.detach().cpu()))

        raw_fd = fd_total / max(len(student_feats), 1)
        weighted_fd = self.lambda_fg * raw_fd
        batch_size = student_feats[0].shape[0]
        train_vec = weighted_fd.reshape(1) * batch_size
        return train_vec, raw_fd.detach(), weighted_fd.detach(), train_vec.detach()

    def forward_loc_fd(
        self,
        student_feats: list[torch.Tensor],
        teacher_feats: list[torch.Tensor],
        student_output: Any,
        teacher_output: Any,
        batch: dict[str, torch.Tensor],
        student_head: nn.Module,
        teacher_head: nn.Module,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return localization-aware feature distillation loss and diagnostics."""
        fd_vec, raw_fd, weighted_fd, _fd_scaled = self.forward_vanilla(student_feats, teacher_feats)
        img_h, img_w = batch["img"].shape[-2:]

        student_decoded = self.decode_detect_output(student_output, student_head, image_shape=(img_h, img_w))
        teacher_decoded = self.decode_detect_output(teacher_output, teacher_head, image_shape=(img_h, img_w))
        loc_result = self.localization_loss(
            student_boxes=student_decoded["boxes"],
            student_scores=student_decoded["scores"],
            teacher_boxes=teacher_decoded["boxes"],
            teacher_scores=teacher_decoded["scores"],
            batch=batch,
            image_shape=(img_h, img_w),
        )

        raw_loc = loc_result["raw_loc"]
        raw_score = loc_result["raw_score"]
        weighted_loc = self.lambda_loc * raw_loc
        weighted_score = self.lambda_score * raw_score
        batch_size = student_feats[0].shape[0]
        weighted_fd_for_grad = fd_vec.sum() / batch_size
        kd_loss = weighted_fd_for_grad + weighted_loc + weighted_score
        kd_loss_train_scaled = fd_vec + (weighted_loc + weighted_score).reshape(1) * batch_size
        diagnostics = {
            "raw_fd": raw_fd,
            "weighted_fd": weighted_fd,
            "raw_loc": raw_loc.detach(),
            "weighted_loc": weighted_loc.detach(),
            "raw_score": raw_score.detach(),
            "weighted_score": weighted_score.detach(),
            "kd_loss": kd_loss.detach(),
            "kd_loss_train_scaled": kd_loss_train_scaled.detach().sum(),
            "loc_fd_formula_delta": (kd_loss.detach() - (weighted_fd + weighted_loc.detach() + weighted_score.detach()))
            .abs(),
            "teacher_score_thr": self.teacher_score_thr,
            "teacher_iou_thr": self.teacher_iou_thr,
            "lambda_loc": self.lambda_loc,
            "lambda_score": self.lambda_score,
            "fd_layer_losses": list(self.last_fd_layer_losses),
            "student_box_stats": student_decoded["stats"],
            "teacher_box_stats": teacher_decoded["stats"],
            **{k: v for k, v in loc_result.items() if k not in {"raw_loc", "raw_score"}},
        }
        return kd_loss_train_scaled, diagnostics

    def decode_detect_output(
        self, output: Any, detect_head: nn.Module, image_shape: tuple[int, int]
    ) -> dict[str, torch.Tensor | dict[str, float | str]]:
        """Extract one2many Detect predictions and decode boxes with the Detect head's official logic."""
        preds = self.select_one2many(output)
        boxes = preds["boxes"]
        scores = preds["scores"]
        feats = preds["feats"]
        if not isinstance(boxes, torch.Tensor) or not isinstance(scores, torch.Tensor):
            raise TypeError("Detect one2many 'boxes' and 'scores' must be tensors.")
        if boxes.ndim != 3 or scores.ndim != 3:
            raise ValueError(f"Unexpected Detect output shapes: boxes={boxes.shape}, scores={scores.shape}")
        if boxes.shape[1] % 4:
            raise ValueError(f"Detect boxes channel dimension must be divisible by 4, got {boxes.shape}.")

        raw_min, raw_max = float(boxes.detach().min().cpu()), float(boxes.detach().max().cpu())
        anchors, stride_tensor = make_anchors(feats, detect_head.stride.to(device=boxes.device), 0.5)
        anchors = anchors.transpose(0, 1)
        stride_tensor = stride_tensor.transpose(0, 1)
        decoded = detect_head.decode_bboxes(detect_head.dfl(boxes), anchors.unsqueeze(0)) * stride_tensor
        decoded = decoded.transpose(1, 2).contiguous()
        if not torch.isfinite(decoded).all():
            raise FloatingPointError("Decoded Detect boxes contain NaN or Inf.")

        img_h, img_w = image_shape
        decoded_min, decoded_max = float(decoded.detach().min().cpu()), float(decoded.detach().max().cpu())
        scale = "pixel" if decoded_max > 2.0 else "normalized"
        if scale != "pixel" or decoded_max > max(img_h, img_w) * 8:
            raise ValueError(
                "Unable to reliably validate decoded box scale. "
                f"decoded min/max=({decoded_min:.6f}, {decoded_max:.6f}), image_shape={image_shape}"
            )
        valid_xyxy = (decoded[..., 2] > decoded[..., 0]) & (decoded[..., 3] > decoded[..., 1])
        valid_ratio = float(valid_xyxy.float().mean().detach().cpu())
        return {
            "boxes": decoded,
            "scores": scores.sigmoid(),
            "stats": {
                "raw_boxes_min": raw_min,
                "raw_boxes_max": raw_max,
                "boxes_min": decoded_min,
                "boxes_max": decoded_max,
                "box_format_detected": "decoded_xyxy",
                "box_scale_detected": scale,
                "valid_xyxy_ratio": valid_ratio,
                "decode_source": "Detect.decode_bboxes(Detect.dfl(boxes), anchors) * strides",
            },
        }

    @staticmethod
    def select_one2many(output: Any) -> dict[str, torch.Tensor]:
        """Select output[1]['one2many'] style Detect predictions, raising a clear error on mismatch."""
        preds = output[1] if isinstance(output, (list, tuple)) and len(output) > 1 and isinstance(output[1], dict) else output
        if not isinstance(preds, dict):
            raise TypeError(f"Expected Detect prediction dict or tuple(_, dict), got {type(output).__name__}.")
        if "one2many" in preds:
            preds = preds["one2many"]
        required = {"boxes", "scores", "feats"}
        missing = required.difference(preds)
        if missing:
            raise KeyError(f"Detect one2many output is missing keys {sorted(missing)}.")
        return preds

    def localization_loss(
        self,
        student_boxes: torch.Tensor,
        student_scores: torch.Tensor,
        teacher_boxes: torch.Tensor,
        teacher_scores: torch.Tensor,
        batch: dict[str, torch.Tensor],
        image_shape: tuple[int, int],
    ) -> dict[str, Any]:
        """Match high-quality teacher predictions to GT boxes and distill localization to the student."""
        batch_idx = batch["batch_idx"].view(-1).long()
        cls = batch["cls"].view(-1).long()
        gt_boxes = self.normalized_xywh_to_pixel_xyxy(batch["bboxes"], image_shape).to(student_boxes.device)
        total_gt = int(gt_boxes.shape[0])
        zero = student_boxes.sum() * 0.0
        if total_gt == 0:
            return self.empty_loc_result(zero, total_gt)

        matched_b, matched_c, matched_i = [], [], []
        qualities, teacher_score_values, teacher_iou_values = [], [], []
        with torch.no_grad():
            for j in range(total_gt):
                b = int(batch_idx[j])
                c = int(cls[j])
                if b < 0 or b >= teacher_boxes.shape[0] or c < 0 or c >= teacher_scores.shape[1]:
                    continue
                ious = self.box_iou_many_to_one(teacher_boxes[b].detach(), gt_boxes[j].detach())
                scores = teacher_scores[b, c].detach()
                teacher_valid = self.valid_xyxy(teacher_boxes[b]).detach()
                candidate = (scores > self.teacher_score_thr) & (ious > self.teacher_iou_thr) & teacher_valid
                if not candidate.any():
                    continue
                quality = scores * ious
                quality = quality.masked_fill(~candidate, -1.0)
                index = int(quality.argmax().item())
                matched_b.append(b)
                matched_c.append(c)
                matched_i.append(index)
                qualities.append((scores[index] * ious[index]).detach())
                teacher_score_values.append(scores[index].detach())
                teacher_iou_values.append(ious[index].detach())

        matched = len(matched_i)
        if matched == 0:
            return self.empty_loc_result(zero, total_gt)

        device = student_boxes.device
        matched_b_t = torch.tensor(matched_b, dtype=torch.long, device=device)
        matched_c_t = torch.tensor(matched_c, dtype=torch.long, device=device)
        matched_i_t = torch.tensor(matched_i, dtype=torch.long, device=device)
        quality = torch.stack(qualities).to(device=device, dtype=student_boxes.dtype)
        teacher_score_t = torch.stack(teacher_score_values).to(device=device, dtype=student_boxes.dtype)
        teacher_iou_t = torch.stack(teacher_iou_values).to(device=device, dtype=student_boxes.dtype)

        student_sel = student_boxes[matched_b_t, matched_i_t]
        teacher_sel = teacher_boxes[matched_b_t, matched_i_t].detach()
        student_valid = self.valid_xyxy(student_sel)
        teacher_valid = self.valid_xyxy(teacher_sel)
        valid_pair = student_valid & teacher_valid
        valid_count = int(valid_pair.sum().detach().cpu())
        invalid_count = matched - valid_count
        invalid_ratio = invalid_count / max(matched, 1)
        if valid_count == 0:
            result = self.empty_loc_result(zero, total_gt)
            result.update(
                {
                    "matched_gt_count": matched,
                    "matched_ratio": matched / max(total_gt, 1),
                    "loc_matched_count": matched,
                    "loc_valid_count": 0,
                    "loc_invalid_count": invalid_count,
                    "loc_invalid_ratio": invalid_ratio,
                    "mean_teacher_score": teacher_score_t.mean().detach(),
                    "mean_teacher_iou": teacher_iou_t.mean().detach(),
                }
            )
            return result

        student_sel = student_sel[valid_pair]
        teacher_sel = teacher_sel[valid_pair]
        matched_b_t = matched_b_t[valid_pair]
        matched_c_t = matched_c_t[valid_pair]
        matched_i_t = matched_i_t[valid_pair]
        quality = quality[valid_pair]
        teacher_score_t = teacher_score_t[valid_pair]
        teacher_iou_t = teacher_iou_t[valid_pair]
        loc_iou = self.box_iou_aligned(student_sel, teacher_sel)
        raw_loc = ((1.0 - loc_iou) * quality).sum() / quality.sum().clamp_min(self.eps)

        student_score_sel = student_scores[matched_b_t, matched_c_t, matched_i_t]
        raw_score = ((student_score_sel - teacher_score_t.detach()).square() * quality).sum() / quality.sum().clamp_min(
            self.eps
        )
        return {
            "raw_loc": raw_loc,
            "raw_score": raw_score,
            "matched_gt_count": matched,
            "total_gt_count": total_gt,
            "matched_ratio": matched / max(total_gt, 1),
            "loc_matched_count": matched,
            "loc_valid_count": valid_count,
            "loc_invalid_count": invalid_count,
            "loc_invalid_ratio": invalid_ratio,
            "mean_teacher_score": teacher_score_t.mean().detach(),
            "mean_teacher_iou": teacher_iou_t.mean().detach(),
        }

    @staticmethod
    def normalized_xywh_to_pixel_xyxy(boxes: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
        """Convert normalized xywh GT boxes to pixel-scale xyxy."""
        img_h, img_w = image_shape
        boxes = boxes.clamp(0, 1)
        x, y, w, h = boxes.unbind(1)
        return torch.stack(((x - w / 2) * img_w, (y - h / 2) * img_h, (x + w / 2) * img_w, (y + h / 2) * img_h), 1)

    def empty_loc_result(self, zero: torch.Tensor, total_gt: int) -> dict[str, Any]:
        """Return a finite zero localization result for batches with no valid teacher matches."""
        return {
            "raw_loc": zero,
            "raw_score": zero,
            "matched_gt_count": 0,
            "total_gt_count": total_gt,
            "matched_ratio": 0.0,
            "loc_matched_count": 0,
            "loc_valid_count": 0,
            "loc_invalid_count": 0,
            "loc_invalid_ratio": 0.0,
            "mean_teacher_score": zero.detach(),
            "mean_teacher_iou": zero.detach(),
        }

    def box_iou_many_to_one(self, boxes: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
        """Compute IoU between many xyxy boxes and one xyxy box."""
        lt = torch.maximum(boxes[:, :2], box[:2])
        rb = torch.minimum(boxes[:, 2:], box[2:])
        wh = (rb - lt).clamp_min(0)
        inter = wh[:, 0] * wh[:, 1]
        area1 = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
        area2 = (box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)
        return inter / (area1 + area2 - inter + self.eps)

    @staticmethod
    def valid_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        """Return a mask for boxes with positive xyxy width and height."""
        eps = torch.finfo(boxes.dtype).eps if boxes.dtype.is_floating_point else 1e-6
        finite = torch.isfinite(boxes).all(dim=-1)
        valid_w = boxes[..., 2] > boxes[..., 0] + eps
        valid_h = boxes[..., 3] > boxes[..., 1] + eps
        return finite & valid_w & valid_h

    def box_iou_aligned(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Compute aligned IoU for pairs of xyxy boxes."""
        lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
        rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
        wh = (rb - lt).clamp_min(0)
        inter = wh[:, 0] * wh[:, 1]
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp_min(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp_min(0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp_min(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp_min(0)
        return inter / (area1 + area2 - inter + self.eps)

    def build_mask(
        self, batch: dict[str, torch.Tensor], height: int, width: int, ref: torch.Tensor
    ) -> torch.Tensor:
        """Build normalized class- and scale-aware foreground masks on a feature map."""
        batch_size = ref.shape[0]
        mask = ref.new_zeros((batch_size, 1, height, width))
        if not {"batch_idx", "cls", "bboxes"}.issubset(batch):
            return mask

        batch_idx = batch["batch_idx"].view(-1).long()
        cls = batch["cls"].view(-1).long()
        boxes = batch["bboxes"].to(device=ref.device, dtype=ref.dtype)
        if boxes.numel() == 0:
            return mask

        boxes = boxes.clamp(0, 1)
        areas = (boxes[:, 2] * boxes[:, 3]).clamp_min(self.eps)
        mean_area = areas.mean().detach().clamp_min(self.eps)
        scale_weights = torch.sqrt(mean_area / areas).clamp(0.5, 2.0)
        cls_clamped = cls.to(ref.device).clamp(0, len(self.class_weights) - 1)
        class_weights = self.class_weights.to(device=ref.device, dtype=ref.dtype)[cls_clamped]
        weights = class_weights * scale_weights

        for i in range(boxes.shape[0]):
            b = int(batch_idx[i])
            if b < 0 or b >= batch_size:
                continue
            x, y, w, h = boxes[i]
            x1 = int(torch.floor((x - w / 2) * width).clamp(0, width - 1).item())
            y1 = int(torch.floor((y - h / 2) * height).clamp(0, height - 1).item())
            x2 = int(torch.ceil((x + w / 2) * width).clamp(1, width).item())
            y2 = int(torch.ceil((y + h / 2) * height).clamp(1, height).item())
            if x2 <= x1 or y2 <= y1:
                continue
            weight = weights[i]
            mask[b, :, y1:y2, x1:x2] = torch.maximum(mask[b, :, y1:y2, x1:x2], weight)

        positive = mask > 0
        if positive.any():
            mask = mask / mask[positive].mean().clamp_min(self.eps)
        return mask
