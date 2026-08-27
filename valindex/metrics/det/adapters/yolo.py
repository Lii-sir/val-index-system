# -*- coding: utf-8 -*-
"""YOLO 标签到统一目标检测输入格式的适配器。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .common import DetectionInput, build_detection, validate_image_size, validate_score

YoloData = str | Path | Iterable[str] | NDArray[Any]
YoloItem = dict[str, Any]


def yolo_to_detection(
    labels: YoloData,
    image_size: tuple[int, int],
    *,
    default_score: float = 1.0,
    clip_boxes: bool = False,
    include_scores: bool = True,
) -> DetectionInput:
    """将 YOLO 标签转换为统一的目标检测输入格式。

    YOLO 标签每行支持以下两种格式：

    ``class_id x_center y_center width height``
    ``class_id x_center y_center width height score``

    坐标使用相对图像尺寸归一化后的值，``image_size`` 格式为
    ``(height, width)``。转换后的框使用像素坐标和 ``xyxy`` 格式。

    五列标签通常用于真实标注，此时使用 ``default_score`` 填充
    ``scores``；六列标签通常用于预测结果，最后一列作为置信度。
    """

    height, width = validate_image_size(image_size)
    default_score = validate_score(default_score, "default_score")
    rows = _read_yolo_rows(labels)

    boxes: list[list[float]] = []
    class_ids: list[int] = []
    scores: list[float] = []

    for line_number, row in enumerate(rows, start=1):
        if len(row) not in (5, 6):
            raise ValueError(
                f"YOLO label at line {line_number} must contain 5 or 6 values."
            )

        class_id = _parse_class_id(row[0], line_number)
        center_x, center_y, box_width, box_height = _parse_coordinates(
            row[1:5],
            line_number,
        )
        score = default_score if len(row) == 5 else validate_score(
            row[5],
            f"score at line {line_number}",
        )

        x1 = (center_x - box_width / 2.0) * width
        y1 = (center_y - box_height / 2.0) * height
        x2 = (center_x + box_width / 2.0) * width
        y2 = (center_y + box_height / 2.0) * height

        if clip_boxes:
            x1 = min(max(x1, 0.0), float(width))
            y1 = min(max(y1, 0.0), float(height))
            x2 = min(max(x2, 0.0), float(width))
            y2 = min(max(y2, 0.0), float(height))
        elif x1 < 0.0 or y1 < 0.0 or x2 > width or y2 > height:
            raise ValueError(
                f"YOLO box at line {line_number} falls outside the image. "
                "Set clip_boxes=True to clip it."
            )

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"YOLO box at line {line_number} has no positive area.")

        boxes.append([x1, y1, x2, y2])
        class_ids.append(class_id)
        scores.append(score)

    return build_detection(
        boxes=boxes,
        labels=class_ids,
        scores=scores,
        include_scores=include_scores,
    )


def yolo_to_detections(
    items: Iterable[YoloItem],
    *,
    default_score: float = 1.0,
    clip_boxes: bool = False,
    include_scores: bool = True,
) -> list[DetectionInput]:
    """批量转换多张图片的 YOLO 标签。

    每个元素必须包含：

    - ``labels``：YOLO 标签文本、文件路径、数组或行列表；
    - ``image_size``：图像尺寸，格式为 ``(height, width)``；

    可选字段：

    - ``image_id``：图片编号，会原样写入输出结果。
    """

    detections: list[DetectionInput] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise TypeError(f"YOLO item at index {index} must be a dict.")
        if "labels" not in item:
            raise ValueError(f"YOLO item at index {index} is missing 'labels'.")
        if "image_size" not in item:
            raise ValueError(f"YOLO item at index {index} is missing 'image_size'.")

        detection = yolo_to_detection(
            labels=item["labels"],
            image_size=item["image_size"],
            default_score=default_score,
            clip_boxes=clip_boxes,
            include_scores=include_scores,
        )
        if "image_id" in item:
            detection["image_id"] = item["image_id"]
        detections.append(detection)

    return detections


def _read_yolo_rows(labels: YoloData) -> list[list[float]]:
    if isinstance(labels, Path):
        text = labels.read_text(encoding="utf-8")
        return _parse_yolo_text(text)

    if isinstance(labels, str):
        # 含换行时按标签文本处理，避免把整段文本当成文件路径。
        if "\n" in labels or "\r" in labels:
            return _parse_yolo_text(labels)
        path = Path(labels)
        if path.is_file():
            return _parse_yolo_text(path.read_text(encoding="utf-8"))
        return _parse_yolo_text(labels)

    if isinstance(labels, np.ndarray):
        return _parse_yolo_array(labels)

    if isinstance(labels, Iterable):
        values = list(labels)
        if not values:
            return []
        if _is_scalar_row(values):
            return [[float(value) for value in values]]
        return [_parse_yolo_row(row) for row in values]

    raise TypeError("labels must be a file path, text, iterable, or numpy array.")


def _parse_yolo_text(text: str) -> list[list[float]]:
    rows: list[list[float]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append([float(value) for value in line.split()])
        except ValueError as exc:
            raise ValueError(f"YOLO label at line {line_number} contains non-numeric values.") from exc
    return rows


def _parse_yolo_array(array: NDArray[Any]) -> list[list[float]]:
    if array.ndim == 1:
        if array.size == 0:
            return []
        return [[float(value) for value in array.tolist()]]
    if array.ndim == 2:
        return [[float(value) for value in row] for row in array.tolist()]
    raise ValueError("YOLO labels array must have shape (5,), (6,), (N, 5), or (N, 6).")


def _parse_yolo_row(row: Any) -> list[float]:
    try:
        return [float(value) for value in row]
    except (TypeError, ValueError) as exc:
        raise ValueError("Each YOLO label row must contain numeric values.") from exc


def _is_scalar_row(values: list[Any]) -> bool:
    return all(np.isscalar(value) for value in values)


def _parse_class_id(value: float, line_number: int) -> int:
    if not np.isfinite(value) or not float(value).is_integer() or value < 0:
        raise ValueError(f"class_id at line {line_number} must be a non-negative integer.")
    return int(value)


def _parse_coordinates(values: list[float], line_number: int) -> tuple[float, float, float, float]:
    if not all(np.isfinite(value) for value in values):
        raise ValueError(f"YOLO coordinates at line {line_number} must be finite.")

    center_x, center_y, box_width, box_height = values
    if not 0.0 <= center_x <= 1.0 or not 0.0 <= center_y <= 1.0:
        raise ValueError(f"YOLO center coordinates at line {line_number} must be in [0, 1].")
    if not 0.0 < box_width <= 1.0 or not 0.0 < box_height <= 1.0:
        raise ValueError(f"YOLO width and height at line {line_number} must be in (0, 1].")

    return center_x, center_y, box_width, box_height
