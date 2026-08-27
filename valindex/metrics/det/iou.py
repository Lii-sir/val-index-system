# -*- coding: utf-8 -*-
"""目标检测框的交并比计算。"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

BoxLike: TypeAlias = Sequence[Real]
BoxesLike: TypeAlias = Sequence[BoxLike] | NDArray[np.floating] | NDArray[np.integer]
FloatArray: TypeAlias = NDArray[np.float64]


def iou(pred_boxes: BoxLike | BoxesLike, target_boxes: BoxLike | BoxesLike) -> float | FloatArray:
    """计算预测框与真实框之间的 IoU。

    如果两边都是单个框，返回 ``float``；否则返回形状为
    ``(预测框数量, 真实框数量)`` 的 IoU 矩阵。
    """

    iou_matrix, pred_is_single, target_is_single = _calculate_iou_matrix(pred_boxes, target_boxes)
    return float(iou_matrix[0, 0]) if pred_is_single and target_is_single else iou_matrix


def box_iou_matrix(pred_boxes: BoxLike | BoxesLike, target_boxes: BoxLike | BoxesLike) -> FloatArray:
    """计算预测框与真实框之间的两两 IoU 矩阵。

    返回数组形状为 ``(预测框数量, 真实框数量)``。
    输入框格式统一为 ``xyxy``，即 ``[x1, y1, x2, y2]``。
    """

    iou_matrix, _, _ = _calculate_iou_matrix(pred_boxes, target_boxes)
    return iou_matrix


def box_intersection_matrix(pred_boxes: BoxLike | BoxesLike, target_boxes: BoxLike | BoxesLike) -> FloatArray:
    """计算预测框与真实框之间的两两交集面积矩阵。"""

    intersection_matrix, _, _, _ = _calculate_overlap_matrices(pred_boxes, target_boxes)
    return intersection_matrix


def box_union_matrix(pred_boxes: BoxLike | BoxesLike, target_boxes: BoxLike | BoxesLike) -> FloatArray:
    """计算预测框与真实框之间的两两并集面积矩阵。"""

    _, union_matrix, _, _ = _calculate_overlap_matrices(pred_boxes, target_boxes)
    return union_matrix


def _calculate_iou_matrix(
    pred_boxes: BoxLike | BoxesLike,
    target_boxes: BoxLike | BoxesLike,
) -> tuple[FloatArray, bool, bool]:
    intersection_matrix, union_matrix, pred_is_single, target_is_single = _calculate_overlap_matrices(
        pred_boxes,
        target_boxes,
    )

    iou_matrix = np.divide(
        intersection_matrix,
        union_matrix,
        out=np.zeros_like(intersection_matrix, dtype=np.float64),
        where=union_matrix > 0,
    )
    return iou_matrix, pred_is_single, target_is_single


def _calculate_overlap_matrices(
    pred_boxes: BoxLike | BoxesLike,
    target_boxes: BoxLike | BoxesLike,
) -> tuple[FloatArray, FloatArray, bool, bool]:
    pred_array, pred_is_single = _normalize_boxes(pred_boxes, "pred_boxes")
    target_array, target_is_single = _normalize_boxes(target_boxes, "target_boxes")

    if pred_array.size == 0 or target_array.size == 0:
        # 任一侧为空时，直接返回对应形状的空矩阵。
        empty_shape = (pred_array.shape[0], target_array.shape[0])
        empty_matrix = np.zeros(empty_shape, dtype=np.float64)
        return empty_matrix, empty_matrix.copy(), pred_is_single, target_is_single

    # 逐对计算交集和并集，IoU 在上层统一相除。
    pred_area = _box_area(pred_array)
    target_area = _box_area(target_array)

    inter_left_top = np.maximum(pred_array[:, None, :2], target_array[None, :, :2])
    inter_right_bottom = np.minimum(pred_array[:, None, 2:], target_array[None, :, 2:])
    inter_wh = np.clip(inter_right_bottom - inter_left_top, a_min=0.0, a_max=None)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    union_area = pred_area[:, None] + target_area[None, :] - inter_area

    return inter_area, union_area, pred_is_single, target_is_single


def _normalize_boxes(boxes: BoxLike | BoxesLike, name: str) -> tuple[FloatArray, bool]:
    array = np.asarray(boxes, dtype=np.float64)

    if array.ndim == 1 and array.size == 0:
        return array.reshape(0, 4), False

    if array.ndim == 1:
        if array.shape[0] != 4:
            raise ValueError(f"{name} must be a single box with shape (4,) or boxes with shape (N, 4).")
        array = array.reshape(1, 4)
        is_single = True
    elif array.ndim == 2:
        if array.shape[1] != 4:
            raise ValueError(f"{name} must have shape (N, 4).")
        is_single = False
    else:
        raise ValueError(f"{name} must be a single box with shape (4,) or boxes with shape (N, 4).")

    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite numeric values.")

    # 统一要求右下角坐标不小于左上角坐标。
    invalid_width = array[:, 2] < array[:, 0]
    invalid_height = array[:, 3] < array[:, 1]
    if np.any(invalid_width | invalid_height):
        raise ValueError(f"{name} boxes must satisfy x2 >= x1 and y2 >= y1.")

    return array, is_single


def _box_area(boxes: FloatArray) -> FloatArray:
    wh = boxes[:, 2:] - boxes[:, :2]
    return wh[:, 0] * wh[:, 1]
