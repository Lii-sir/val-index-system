# -*- coding: utf-8 -*-
"""目标检测适配器的公共类型和校验方法。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any, TypeAlias

DetectionInput: TypeAlias = dict[str, Any]
CategoryIdMap: TypeAlias = Mapping[int, int]
ImageSize: TypeAlias = tuple[int, int]


def validate_image_size(image_size: ImageSize) -> tuple[int, int]:
    """校验并返回图像尺寸，尺寸格式为 ``(height, width)``。"""

    if not isinstance(image_size, Sequence) or len(image_size) != 2:
        raise ValueError("image_size must be a pair in the format (height, width).")

    height, width = image_size
    if (
        isinstance(height, bool)
        or isinstance(width, bool)
        or not isinstance(height, Integral)
        or not isinstance(width, Integral)
        or height <= 0
        or width <= 0
    ):
        raise ValueError("image_size must contain two positive integers: (height, width).")

    return int(height), int(width)


def validate_category_id(category_id: Any, name: str = "category_id") -> int:
    """校验类别编号并转换为整数。"""

    if isinstance(category_id, bool) or not isinstance(category_id, Integral):
        raise ValueError(f"{name} must be an integer.")

    return int(category_id)


def validate_score(score: Any, name: str = "score") -> float:
    """校验置信度并转换为浮点数。"""

    if isinstance(score, bool) or not isinstance(score, Real):
        raise ValueError(f"{name} must be a real number.")

    score = float(score)
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"{name} must be in the range [0, 1].")

    return score


def resolve_category_id(category_id: Any, category_id_map: CategoryIdMap | None) -> int:
    """将外部类别编号转换为内部类别编号。"""

    category_id = validate_category_id(category_id)
    if category_id_map is None:
        return category_id

    if category_id not in category_id_map:
        raise ValueError(f"category_id {category_id} is missing from category_id_map.")

    return validate_category_id(category_id_map[category_id], "mapped_category_id")


def build_detection(
    boxes: list[list[float]],
    labels: list[int],
    scores: list[float],
    *,
    image_id: int | None = None,
    include_scores: bool = True,
) -> DetectionInput:
    """构造统一的目标检测输入字典。"""

    result: DetectionInput = {
        "boxes": boxes,
        "labels": labels,
    }
    if include_scores:
        result["scores"] = scores
    if image_id is not None:
        result["image_id"] = image_id
    return result
