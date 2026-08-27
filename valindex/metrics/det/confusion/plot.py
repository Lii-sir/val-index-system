# -*- coding: utf-8 -*-
"""目标检测混淆矩阵绘图。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def save_detection_confusion_matrix(
    matrix: Any,
    *,
    labels: list[str] | None = None,
    save_path: str | Path,
    title: str = "Detection Confusion Matrix",
    normalize: bool = False,
    show_values: bool = True,
    cmap: str = "Blues",
) -> Path:
    """绘制并保存混淆矩阵图片。"""

    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be a square 2D array.")

    if labels is None:
        labels = [str(index) for index in range(array.shape[0])]
    if len(labels) != array.shape[0]:
        raise ValueError("labels length must match matrix size.")

    if normalize:
        row_sum = array.sum(axis=1, keepdims=True)
        array = np.divide(array, row_sum, out=np.zeros_like(array), where=row_sum > 0)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig_width = max(6.0, 0.9 * array.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_width))
    image = ax.imshow(array, interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    if show_values:
        threshold = array.max() * 0.5 if array.size else 0.0
        for row in range(array.shape[0]):
            for col in range(array.shape[1]):
                value = array[row, col]
                text = f"{value:.2f}" if normalize else f"{int(round(value))}"
                ax.text(
                    col,
                    row,
                    text,
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                    fontsize=9,
                )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path
