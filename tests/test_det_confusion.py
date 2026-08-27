# -*- coding: utf-8 -*-
"""目标检测混淆矩阵与 TP/FP/FN/TN 的单元测试。"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from valindex import metrics


class DetectionConfusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = [
            {
                "image_id": "img_001",
                "boxes": [[10, 10, 20, 20], [30, 30, 40, 40]],
                "labels": [0, 1],
                "scores": [0.9, 0.8],
            },
            {
                "image_id": "img_002",
                "boxes": [],
                "labels": [],
                "scores": [],
            },
        ]
        self.targets = [
            {
                "image_id": "img_001",
                "boxes": [[10, 10, 20, 20], [30, 30, 40, 40]],
                "labels": [0, 2],
                "scores": [1.0, 1.0],
            },
            {
                "image_id": "img_002",
                "boxes": [],
                "labels": [],
                "scores": [],
            },
        ]

    def test_detection_confusion_matrix_multiclass(self):
        result = metrics.det.detection_confusion_matrix(
            self.predictions,
            self.targets,
            iou_threshold=0.5,
        )

        self.assertEqual(result["labels"], ["0", "1", "2", "background"])
        self.assertEqual(result["matrix"].shape, (4, 4))
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["tn"], 1)
        self.assertEqual(result["matrix"][0, 0], 1)
        self.assertEqual(result["matrix"][2, 1], 1)

    def test_detection_confusion_matrix_can_ignore_class(self):
        result = metrics.det.detection_confusion_matrix(
            self.predictions,
            self.targets,
            ignore_class_ids=[2],
        )

        self.assertEqual(result["labels"], ["0", "1", "background"])
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 0)

    def test_detection_confusion_matrix_can_merge_classes(self):
        result = metrics.det.detection_confusion_matrix(
            self.predictions,
            self.targets,
            merge_classes=True,
        )

        self.assertEqual(result["labels"], ["all", "background"])
        self.assertEqual(result["matrix"].shape, (2, 2))
        self.assertEqual(result["tp"], 2)

    def test_detection_tp_fp_fn_tn(self):
        result = metrics.det.detection_tp_fp_fn_tn(self.predictions, self.targets)

        self.assertEqual(result, {"tp": 1, "fp": 1, "fn": 1, "tn": 1})

    def test_save_detection_confusion_matrix(self):
        with TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "confusion.png"
            matrix = metrics.det.detection_confusion_matrix(
                self.predictions,
                self.targets,
            )["matrix"]
            result_path = metrics.det.save_detection_confusion_matrix(
                matrix,
                labels=["0", "1", "2", "background"],
                save_path=save_path,
            )

            self.assertTrue(result_path.exists())
            self.assertGreater(result_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
