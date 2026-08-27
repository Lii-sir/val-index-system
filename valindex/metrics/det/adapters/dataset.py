# -*- coding: utf-8 -*-
"""按图片文件夹加载目标检测数据集的适配器。"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .common import DetectionInput, build_detection
from .yolo import yolo_to_detection

ImageExtensions = tuple[str, ...]

_DEFAULT_IMAGE_EXTENSIONS: ImageExtensions = (".bmp", ".gif", ".jpeg", ".jpg", ".png")


def load_detection_dataset(
    image_dir: str | Path,
    gt_label_dir: str | Path,
    pred_label_dir: str | Path,
    *,
    label_suffix: str = ".txt",
    image_extensions: Iterable[str] | None = None,
    default_score: float = 1.0,
    clip_boxes: bool = False,
    missing_ok: bool = False,
) -> tuple[list[DetectionInput], list[DetectionInput]]:
    """从图片文件夹、GT 标签文件夹和预测文件夹加载检测数据。

    该接口面向工程使用，返回值直接用于目标检测指标计算：

    - ``predictions``：预测结果列表；
    - ``targets``：真实标注列表。

    每个元素都包含 ``boxes``、``labels``、``scores``，并额外带有
    ``image_id``，其中 ``image_id`` 使用相对图片路径去掉后缀后的字符串。
    """

    image_dir = Path(image_dir)
    gt_label_dir = Path(gt_label_dir)
    pred_label_dir = Path(pred_label_dir)

    _validate_directory(image_dir, "image_dir")
    _validate_directory(gt_label_dir, "gt_label_dir")
    _validate_directory(pred_label_dir, "pred_label_dir")
    _validate_label_suffix(label_suffix)

    extensions = _normalize_extensions(image_extensions)
    image_files = _collect_image_files(image_dir, extensions)

    predictions: list[DetectionInput] = []
    targets: list[DetectionInput] = []

    for image_path in image_files:
        image_size = read_image_size(image_path)
        relative_stem = image_path.relative_to(image_dir).with_suffix("")
        image_id = relative_stem.as_posix()

        gt_label_path = (gt_label_dir / relative_stem).with_suffix(label_suffix)
        pred_label_path = (pred_label_dir / relative_stem).with_suffix(label_suffix)

        gt_detection = _load_yolo_file(
            gt_label_path,
            image_size=image_size,
            default_score=1.0,
            clip_boxes=clip_boxes,
            missing_ok=missing_ok,
        )
        pred_detection = _load_yolo_file(
            pred_label_path,
            image_size=image_size,
            default_score=default_score,
            clip_boxes=clip_boxes,
            missing_ok=missing_ok,
        )

        gt_detection["scores"] = [1.0] * len(gt_detection["labels"])
        gt_detection["image_id"] = image_id
        pred_detection["image_id"] = image_id

        predictions.append(pred_detection)
        targets.append(gt_detection)

    return predictions, targets


def _load_yolo_file(
    label_path: Path,
    *,
    image_size: tuple[int, int],
    default_score: float,
    clip_boxes: bool,
    missing_ok: bool,
) -> DetectionInput:
    if not label_path.exists():
        if missing_ok:
            return build_detection(boxes=[], labels=[], scores=[])
        raise FileNotFoundError(f"Label file does not exist: {label_path}")
    if not label_path.is_file():
        raise ValueError(f"Label path is not a file: {label_path}")

    return yolo_to_detection(
        label_path,
        image_size=image_size,
        default_score=default_score,
        clip_boxes=clip_boxes,
        include_scores=True,
    )


def _collect_image_files(image_dir: Path, extensions: ImageExtensions) -> list[Path]:
    files = [
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    files.sort(key=lambda path: path.relative_to(image_dir).as_posix())
    return files


def _normalize_extensions(image_extensions: Iterable[str] | None) -> ImageExtensions:
    if image_extensions is None:
        return _DEFAULT_IMAGE_EXTENSIONS

    normalized: list[str] = []
    for extension in image_extensions:
        if not isinstance(extension, str):
            raise TypeError("image_extensions must contain strings.")
        extension = extension.lower().strip()
        if not extension.startswith("."):
            extension = "." + extension
        normalized.append(extension)

    if not normalized:
        raise ValueError("image_extensions must not be empty.")

    return tuple(sorted(set(normalized)))


def _validate_directory(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"{name} must be a directory: {path}")


def _validate_label_suffix(label_suffix: str) -> None:
    if not isinstance(label_suffix, str) or not label_suffix:
        raise ValueError("label_suffix must be a non-empty string.")
    if not label_suffix.startswith("."):
        raise ValueError("label_suffix must start with a dot, such as '.txt'.")


def read_image_size(image_path: str | Path) -> tuple[int, int]:
    """读取图片尺寸，返回 ``(height, width)``。"""

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")

    data = path.read_bytes()
    if len(data) < 10:
        raise ValueError(f"Image file is too small to read size: {path}")

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return _read_png_size(data, path)
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return _read_gif_size(data, path)
    if data.startswith(b"BM"):
        return _read_bmp_size(data, path)
    if data.startswith(b"\xff\xd8"):
        return _read_jpeg_size(data, path)

    raise ValueError(f"Unsupported image format for size reading: {path}")


def _read_png_size(data: bytes, path: Path) -> tuple[int, int]:
    if len(data) < 24:
        raise ValueError(f"PNG file is too small to read size: {path}")
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    return _validate_positive_size(height, width, path)


def _read_gif_size(data: bytes, path: Path) -> tuple[int, int]:
    if len(data) < 10:
        raise ValueError(f"GIF file is too small to read size: {path}")
    width = int.from_bytes(data[6:8], "little")
    height = int.from_bytes(data[8:10], "little")
    return _validate_positive_size(height, width, path)


def _read_bmp_size(data: bytes, path: Path) -> tuple[int, int]:
    if len(data) < 26:
        raise ValueError(f"BMP file is too small to read size: {path}")
    width = int.from_bytes(data[18:22], "little", signed=True)
    height = abs(int.from_bytes(data[22:26], "little", signed=True))
    return _validate_positive_size(height, width, path)


def _read_jpeg_size(data: bytes, path: Path) -> tuple[int, int]:
    index = 2
    length = len(data)
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }

    while index < length:
        while index < length and data[index] != 0xFF:
            index += 1
        while index < length and data[index] == 0xFF:
            index += 1
        if index >= length:
            break

        marker = data[index]
        index += 1

        if marker in (0xD8, 0xD9):
            continue
        if marker == 0xDA:
            break

        if index + 2 > length:
            break

        segment_length = int.from_bytes(data[index:index + 2], "big")
        if segment_length < 2:
            raise ValueError(f"Invalid JPEG segment length in {path}.")

        if marker in sof_markers:
            if index + 7 > length:
                raise ValueError(f"JPEG file is too small to read size: {path}")
            height = int.from_bytes(data[index + 3:index + 5], "big")
            width = int.from_bytes(data[index + 5:index + 7], "big")
            return _validate_positive_size(height, width, path)

        index += segment_length

    raise ValueError(f"Could not determine JPEG size: {path}")


def _validate_positive_size(height: int, width: int, path: Path) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size read from {path}: ({height}, {width})")
    return height, width
