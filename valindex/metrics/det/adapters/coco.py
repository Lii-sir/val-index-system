# -*- coding: utf-8 -*-
"""COCO 标注到统一目标检测输入格式的适配器。"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .common import (
    CategoryIdMap,
    DetectionInput,
    build_detection,
    resolve_category_id,
    validate_score,
)

CocoData = Mapping[str, Any] | Sequence[Mapping[str, Any]] | str | Path


def build_coco_category_id_map(
    categories: Sequence[Mapping[str, Any]],
    *,
    start: int = 0,
) -> dict[int, int]:
    """根据 COCO ``categories`` 列表生成连续的内部类别编号映射。"""

    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("start must be a non-negative integer.")

    category_id_map: dict[int, int] = {}
    for index, category in enumerate(categories):
        if "id" not in category:
            raise ValueError(f"COCO category at index {index} is missing 'id'.")
        category_id = int(category["id"])
        if category_id in category_id_map:
            raise ValueError(f"Duplicate COCO category id: {category_id}.")
        category_id_map[category_id] = start + index
    return category_id_map


def coco_annotations_to_detection(
    annotations: Sequence[Mapping[str, Any]],
    *,
    category_id_map: CategoryIdMap | None = None,
    default_score: float = 1.0,
    include_scores: bool = True,
) -> DetectionInput:
    """将同一张图片的 COCO annotations 转换为统一格式。

    COCO 标注框格式为 ``[x, y, width, height]``，使用绝对像素坐标；
    返回的框格式为 ``[x1, y1, x2, y2]``。
    """

    default_score = validate_score(default_score, "default_score")
    boxes: list[list[float]] = []
    labels: list[int] = []
    scores: list[float] = []

    for index, annotation in enumerate(annotations):
        if "bbox" not in annotation:
            raise ValueError(f"COCO annotation at index {index} is missing 'bbox'.")
        if "category_id" not in annotation:
            raise ValueError(f"COCO annotation at index {index} is missing 'category_id'.")

        bbox = annotation["bbox"]
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            raise ValueError(f"COCO bbox at index {index} must have four values.")

        x, y, box_width, box_height = _parse_bbox(bbox, index)
        boxes.append([x, y, x + box_width, y + box_height])
        labels.append(resolve_category_id(annotation["category_id"], category_id_map))

        score = annotation.get("score", default_score)
        scores.append(validate_score(score, f"score at annotation {index}"))

    return build_detection(
        boxes=boxes,
        labels=labels,
        scores=scores,
        include_scores=include_scores,
    )


def coco_to_detections(
    coco_data: CocoData,
    *,
    category_id_map: CategoryIdMap | None = None,
    default_score: float = 1.0,
    include_empty_images: bool = True,
    include_scores: bool = True,
) -> list[DetectionInput]:
    """将 COCO 标注文件或检测结果列表转换为按图片组织的结果。

    返回值是一个列表，每个元素对应一张图片，并额外包含 ``image_id``。
    对完整 COCO 标注文件，函数会根据 ``images`` 保留空标注图片；
    对 COCO 检测结果列表，则按结果中的 ``image_id`` 自动分组。
    """

    data = _load_coco_data(coco_data)
    if isinstance(data, Mapping):
        return _convert_coco_annotation_file(
            data,
            category_id_map=category_id_map,
            default_score=default_score,
            include_empty_images=include_empty_images,
            include_scores=include_scores,
        )

    return _convert_coco_result_list(
        data,
        category_id_map=category_id_map,
        default_score=default_score,
        include_scores=include_scores,
    )


def _convert_coco_annotation_file(
    data: Mapping[str, Any],
    *,
    category_id_map: CategoryIdMap | None,
    default_score: float,
    include_empty_images: bool,
    include_scores: bool,
) -> list[DetectionInput]:
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    if not isinstance(images, Sequence) or not isinstance(annotations, Sequence):
        raise ValueError("COCO 'images' and 'annotations' must be lists.")

    annotations_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            raise ValueError(f"COCO annotation at index {index} must be an object.")
        if "image_id" not in annotation:
            raise ValueError(f"COCO annotation at index {index} is missing 'image_id'.")
        image_id = int(annotation["image_id"])
        annotations_by_image[image_id].append(annotation)

    result: list[DetectionInput] = []
    image_ids: list[int] = []
    for index, image in enumerate(images):
        if not isinstance(image, Mapping) or "id" not in image:
            raise ValueError(f"COCO image at index {index} is missing 'id'.")
        image_ids.append(int(image["id"]))

    if not images:
        image_ids = sorted(annotations_by_image)

    for image_id in image_ids:
        image_annotations = annotations_by_image.get(image_id, [])
        if not image_annotations and not include_empty_images:
            continue
        detection = coco_annotations_to_detection(
            image_annotations,
            category_id_map=category_id_map,
            default_score=default_score,
            include_scores=include_scores,
        )
        detection["image_id"] = image_id
        result.append(detection)

    return result


def _convert_coco_result_list(
    records: Sequence[Mapping[str, Any]],
    *,
    category_id_map: CategoryIdMap | None,
    default_score: float,
    include_scores: bool,
) -> list[DetectionInput]:
    records_by_image: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"COCO detection result at index {index} must be an object.")
        if "image_id" not in record:
            raise ValueError(f"COCO detection result at index {index} is missing 'image_id'.")
        records_by_image[int(record["image_id"])].append(record)

    result: list[DetectionInput] = []
    for image_id in sorted(records_by_image):
        detection = coco_annotations_to_detection(
            records_by_image[image_id],
            category_id_map=category_id_map,
            default_score=default_score,
            include_scores=include_scores,
        )
        detection["image_id"] = image_id
        result.append(detection)
    return result


def _load_coco_data(coco_data: CocoData) -> Mapping[str, Any] | Sequence[Mapping[str, Any]]:
    if isinstance(coco_data, Path):
        return json.loads(coco_data.read_text(encoding="utf-8"))
    if isinstance(coco_data, str):
        path = Path(coco_data)
        if not path.exists():
            raise FileNotFoundError(f"COCO file does not exist: {coco_data}")
        return json.loads(path.read_text(encoding="utf-8"))
    if isinstance(coco_data, (Mapping, Sequence)) and not isinstance(coco_data, (str, bytes)):
        return coco_data
    raise TypeError("coco_data must be a COCO object, result list, or JSON file path.")


def _parse_bbox(bbox: Any, index: int) -> tuple[float, float, float, float]:
    try:
        values = [float(value) for value in bbox]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"COCO bbox at index {index} must contain numeric values.") from exc

    if not all(value >= 0.0 for value in values):
        raise ValueError(f"COCO bbox at index {index} cannot contain negative values.")
    if values[2] <= 0.0 or values[3] <= 0.0:
        raise ValueError(f"COCO bbox at index {index} must have positive width and height.")

    return values[0], values[1], values[2], values[3]
