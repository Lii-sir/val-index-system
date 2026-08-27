# -*- coding: utf-8 -*-
"""目标检测的混淆矩阵与 TP/FP/FN/TN 统计。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..adapters.common import validate_category_id, validate_score
from ..iou import box_iou_matrix

DetectionItem = Mapping[str, Any]
DetectionItems = Sequence[DetectionItem]


def detection_confusion_matrix(
    predictions: DetectionItems,
    targets: DetectionItems,
    *,
    iou_threshold: float = 0.5,
    class_ids: Sequence[int] | None = None,
    ignore_class_ids: Sequence[int] | None = None,
    merge_classes: bool = False,
    class_names: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    """计算目标检测混淆矩阵。

    参数说明：
    - ``class_ids``：指定参与统计的类别顺序；
    - ``ignore_class_ids``：指定需要忽略的类别；
    - ``merge_classes``：将所有非忽略类别合并为一类。
    """

    _validate_iou_threshold(iou_threshold)
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length.")

    ignore_set = _normalize_class_id_set(ignore_class_ids, "ignore_class_ids")

    if merge_classes:
        return _build_merged_confusion_matrix(
            predictions,
            targets,
            iou_threshold=iou_threshold,
            ignore_set=ignore_set,
        )

    class_order = _resolve_class_order(
        predictions,
        targets,
        class_ids=class_ids,
        ignore_set=ignore_set,
    )
    return _build_multiclass_confusion_matrix(
        predictions,
        targets,
        class_order=class_order,
        iou_threshold=iou_threshold,
        ignore_set=ignore_set,
        class_names=class_names,
    )


def detection_tp_fp_fn_tn(
    predictions: DetectionItems,
    targets: DetectionItems,
    *,
    iou_threshold: float = 0.5,
    class_ids: Sequence[int] | None = None,
    ignore_class_ids: Sequence[int] | None = None,
    merge_classes: bool = False,
) -> dict[str, int]:
    """直接返回 TP/FP/FN/TN 统计。"""

    result = detection_confusion_matrix(
        predictions,
        targets,
        iou_threshold=iou_threshold,
        class_ids=class_ids,
        ignore_class_ids=ignore_class_ids,
        merge_classes=merge_classes,
    )
    return {
        "tp": int(result["tp"]),
        "fp": int(result["fp"]),
        "fn": int(result["fn"]),
        "tn": int(result["tn"]),
    }


def _build_multiclass_confusion_matrix(
    predictions: DetectionItems,
    targets: DetectionItems,
    *,
    class_order: list[int],
    iou_threshold: float,
    ignore_set: set[int],
    class_names: Mapping[int, str] | None,
) -> dict[str, Any]:
    label_to_index = {class_id: index for index, class_id in enumerate(class_order)}
    background_index = len(class_order)
    matrix = np.zeros((background_index + 1, background_index + 1), dtype=np.int64)
    per_class: dict[str, dict[str, int]] = {
        _display_label(class_id, class_names): {"tp": 0, "fp": 0, "fn": 0}
        for class_id in class_order
    }

    tp = fp = fn = tn = 0
    for index, (pred_item, target_item) in enumerate(zip(predictions, targets, strict=True)):
        pred = _normalize_detection_item(pred_item, f"predictions[{index}]")
        target = _normalize_detection_item(target_item, f"targets[{index}]")
        _validate_image_id(pred_item, target_item, index)

        pred = _filter_detection_item(pred, ignore_set, allowed_set=set(class_order))
        target = _filter_detection_item(target, ignore_set, allowed_set=set(class_order))

        image_tp, image_fp, image_fn = _match_image(
            pred,
            target,
            iou_threshold=iou_threshold,
            label_to_index=label_to_index,
            background_index=background_index,
            merge_classes=False,
            matrix=matrix,
        )

        tp += image_tp
        fp += image_fp
        fn += image_fn
        if not pred["boxes"] and not target["boxes"]:
            tn += 1

    _fill_per_class_stats(matrix, class_order, per_class, class_names)

    return {
        "matrix": matrix,
        "labels": [_display_label(class_id, class_names) for class_id in class_order] + ["background"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "details": {
            "per_class": per_class,
        },
    }


def _build_merged_confusion_matrix(
    predictions: DetectionItems,
    targets: DetectionItems,
    *,
    iou_threshold: float,
    ignore_set: set[int],
) -> dict[str, Any]:
    matrix = np.zeros((2, 2), dtype=np.int64)
    tp = fp = fn = tn = 0

    for index, (pred_item, target_item) in enumerate(zip(predictions, targets, strict=True)):
        pred = _normalize_detection_item(pred_item, f"predictions[{index}]")
        target = _normalize_detection_item(target_item, f"targets[{index}]")
        _validate_image_id(pred_item, target_item, index)

        pred = _filter_detection_item(pred, ignore_set, allowed_set=None)
        target = _filter_detection_item(target, ignore_set, allowed_set=None)

        image_tp, image_fp, image_fn = _match_image(
            pred,
            target,
            iou_threshold=iou_threshold,
            label_to_index={0: 0},
            background_index=1,
            merge_classes=True,
            matrix=matrix,
        )

        tp += image_tp
        fp += image_fp
        fn += image_fn
        if not pred["boxes"] and not target["boxes"]:
            tn += 1

    return {
        "matrix": matrix,
        "labels": ["all", "background"],
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "details": {
            "per_class": {
                "all": {
                    "tp": int(matrix[0, 0]),
                    "fp": int(matrix[:, 0].sum() - matrix[0, 0]),
                    "fn": int(matrix[0, :].sum() - matrix[0, 0]),
                }
            },
        },
    }


def _match_image(
    pred: DetectionItem,
    target: DetectionItem,
    *,
    iou_threshold: float,
    label_to_index: dict[int, int],
    background_index: int,
    merge_classes: bool,
    matrix: NDArray[np.int64],
) -> tuple[int, int, int]:
    pred_boxes = pred["boxes"]
    target_boxes = target["boxes"]
    pred_labels = pred["labels"]
    target_labels = target["labels"]
    pred_scores = pred["scores"]

    if not pred_boxes and not target_boxes:
        return 0, 0, 0

    if not pred_boxes:
        for target_label in target_labels:
            target_index = label_to_index.get(target_label, background_index)
            if target_index == background_index:
                continue
            matrix[target_index, background_index] += 1
        return 0, 0, len(target_labels)

    if not target_boxes:
        for pred_label in pred_labels:
            pred_index = label_to_index.get(pred_label, background_index)
            if pred_index == background_index:
                continue
            matrix[background_index, pred_index] += 1
        return 0, len(pred_labels), 0

    iou_matrix = box_iou_matrix(pred_boxes, target_boxes)
    order = sorted(range(len(pred_scores)), key=pred_scores.__getitem__, reverse=True)
    matched_targets: set[int] = set()

    tp = fp = fn = 0
    for pred_index in order:
        candidate_indices = [index for index in range(len(target_boxes)) if index not in matched_targets]
        if not candidate_indices:
            pred_label = pred_labels[pred_index]
            pred_matrix_index = label_to_index.get(pred_label, background_index)
            if pred_matrix_index != background_index:
                matrix[background_index, pred_matrix_index] += 1
            fp += 1
            continue

        candidate_ious = iou_matrix[pred_index, candidate_indices]
        best_pos = int(np.argmax(candidate_ious))
        best_target_index = candidate_indices[best_pos]
        best_iou = float(candidate_ious[best_pos])
        if best_iou < iou_threshold:
            pred_label = pred_labels[pred_index]
            pred_matrix_index = label_to_index.get(pred_label, background_index)
            if pred_matrix_index != background_index:
                matrix[background_index, pred_matrix_index] += 1
            fp += 1
            continue

        matched_targets.add(best_target_index)
        target_label = target_labels[best_target_index]
        pred_label = pred_labels[pred_index]

        if merge_classes:
            matrix[0, 0] += 1
            tp += 1
            continue

        target_matrix_index = label_to_index[target_label]
        pred_matrix_index = label_to_index[pred_label]
        matrix[target_matrix_index, pred_matrix_index] += 1
        if pred_label == target_label:
            tp += 1
        else:
            fp += 1
            fn += 1

    for target_index in range(len(target_boxes)):
        if target_index in matched_targets:
            continue
        target_label = target_labels[target_index]
        target_matrix_index = label_to_index.get(target_label, background_index)
        if target_matrix_index != background_index:
            matrix[target_matrix_index, background_index] += 1
        fn += 1

    return tp, fp, fn


def _resolve_class_order(
    predictions: DetectionItems,
    targets: DetectionItems,
    *,
    class_ids: Sequence[int] | None,
    ignore_set: set[int],
) -> list[int]:
    if class_ids is not None:
        resolved: list[int] = []
        seen: set[int] = set()
        allowed = set()
        for index, class_id in enumerate(class_ids):
            resolved_class_id = validate_category_id(class_id, f"class_ids[{index}]")
            if resolved_class_id in ignore_set or resolved_class_id in seen:
                continue
            resolved.append(resolved_class_id)
            seen.add(resolved_class_id)
            allowed.add(resolved_class_id)
        _validate_labels_in_allowed_set(predictions, targets, allowed, ignore_set)
        return resolved

    class_set: set[int] = set()
    for item_index, item in enumerate(list(predictions) + list(targets)):
        normalized = _normalize_detection_item(item, f"detections[{item_index}]")
        for label in normalized["labels"]:
            if label not in ignore_set:
                class_set.add(label)
    return sorted(class_set)


def _validate_labels_in_allowed_set(
    predictions: DetectionItems,
    targets: DetectionItems,
    allowed: set[int],
    ignore_set: set[int],
) -> None:
    for item_index, item in enumerate(list(predictions) + list(targets)):
        normalized = _normalize_detection_item(item, f"detections[{item_index}]")
        for label in normalized["labels"]:
            if label in ignore_set:
                continue
            if label not in allowed:
                raise ValueError(
                    f"label {label} is not included in class_ids and is not ignored."
                )


def _fill_per_class_stats(
    matrix: NDArray[np.int64],
    class_order: list[int],
    per_class: dict[str, dict[str, int]],
    class_names: Mapping[int, str] | None,
) -> None:
    for index, class_id in enumerate(class_order):
        label = _display_label(class_id, class_names)
        per_class[label] = {
            "tp": int(matrix[index, index]),
            "fp": int(matrix[:, index].sum() - matrix[index, index]),
            "fn": int(matrix[index, :].sum() - matrix[index, index]),
        }


def _normalize_detection_item(item: DetectionItem, name: str) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    for key in ("boxes", "labels", "scores"):
        if key not in item:
            raise ValueError(f"{name} is missing '{key}'.")

    boxes = _normalize_boxes(item["boxes"], f"{name}['boxes']")
    labels = _normalize_labels(item["labels"], f"{name}['labels']")
    scores = _normalize_scores(item["scores"], f"{name}['scores']")

    if len(boxes) != len(labels) or len(labels) != len(scores):
        raise ValueError(f"{name} requires boxes, labels, and scores to have the same length.")

    return {
        "boxes": boxes,
        "labels": labels,
        "scores": scores,
        "image_id": item.get("image_id"),
    }


def _normalize_boxes(boxes: Any, name: str) -> list[list[float]]:
    array = np.asarray(boxes, dtype=np.float64)
    if array.ndim == 1 and array.size == 0:
        return []
    if array.ndim == 1:
        if array.shape[0] != 4:
            raise ValueError(f"{name} must contain boxes with four values.")
        array = array.reshape(1, 4)
    elif array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(f"{name} must have shape (N, 4).")

    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numeric values.")
    if np.any(array[:, 2] < array[:, 0]) or np.any(array[:, 3] < array[:, 1]):
        raise ValueError(f"{name} boxes must satisfy x2 >= x1 and y2 >= y1.")

    return [[float(value) for value in row] for row in array.tolist()]


def _normalize_labels(labels: Any, name: str) -> list[int]:
    if isinstance(labels, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers.")
    normalized: list[int] = []
    for index, label in enumerate(labels):
        normalized.append(validate_category_id(label, f"{name}[{index}]"))
    return normalized


def _normalize_scores(scores: Any, name: str) -> list[float]:
    if isinstance(scores, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numbers.")
    normalized: list[float] = []
    for index, score in enumerate(scores):
        normalized.append(validate_score(score, f"{name}[{index}]"))
    return normalized


def _filter_detection_item(
    item: dict[str, Any],
    ignore_set: set[int],
    *,
    allowed_set: set[int] | None,
) -> dict[str, Any]:
    if not ignore_set and allowed_set is None:
        return item

    boxes: list[list[float]] = []
    labels: list[int] = []
    scores: list[float] = []
    for box, label, score in zip(item["boxes"], item["labels"], item["scores"], strict=True):
        if label in ignore_set:
            continue
        if allowed_set is not None and label not in allowed_set:
            raise ValueError(f"label {label} is not included in the allowed class set.")
        boxes.append(box)
        labels.append(label)
        scores.append(score)

    return {
        "boxes": boxes,
        "labels": labels,
        "scores": scores,
        "image_id": item.get("image_id"),
    }


def _validate_iou_threshold(iou_threshold: float) -> None:
    if not isinstance(iou_threshold, (int, float)) or not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must be a number in [0, 1].")


def _normalize_class_id_set(values: Sequence[int] | None, name: str) -> set[int]:
    if values is None:
        return set()
    normalized: set[int] = set()
    for index, value in enumerate(values):
        normalized.add(validate_category_id(value, f"{name}[{index}]"))
    return normalized


def _display_label(class_id: int, class_names: Mapping[int, str] | None) -> str:
    if class_names is None:
        return str(class_id)
    return class_names.get(class_id, str(class_id))


def _validate_image_id(pred_item: DetectionItem, target_item: DetectionItem, index: int) -> None:
    pred_image_id = pred_item.get("image_id")
    target_image_id = target_item.get("image_id")
    if pred_image_id is not None and target_image_id is not None and pred_image_id != target_image_id:
        raise ValueError(
            f"predictions[{index}] and targets[{index}] have different image_id values."
        )
