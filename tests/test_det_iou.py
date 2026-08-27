# -*- coding: utf-8 -*-
"""目标检测 IoU 的单元测试。"""

import numpy as np
import unittest

from valindex import metrics


class DetectionIoUTest(unittest.TestCase):
    def test_single_box_iou_returns_float(self):
        result = metrics.det.iou([10, 10, 20, 20], [12, 12, 22, 22])

        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 64 / 136)

    def test_multi_box_iou_returns_pairwise_matrix(self):
        pred_boxes = [[10, 10, 20, 20], [0, 0, 10, 10]]
        target_boxes = [[12, 12, 22, 22], [0, 0, 10, 10]]

        result = metrics.det.iou(pred_boxes, target_boxes)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2))
        self.assertAlmostEqual(result[0, 0], 64 / 136)
        self.assertAlmostEqual(result[1, 1], 1.0)

    def test_box_iou_matrix_always_returns_matrix(self):
        result = metrics.det.box_iou_matrix([0, 0, 10, 10], [0, 0, 10, 10])

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (1, 1))
        self.assertAlmostEqual(result[0, 0], 1.0)

    def test_intersection_matrix_returns_pairwise_areas(self):
        result = metrics.det.box_intersection_matrix(
            [[10, 10, 20, 20], [0, 0, 10, 10]],
            [[12, 12, 22, 22], [0, 0, 10, 10]],
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2))
        self.assertAlmostEqual(result[0, 0], 64.0)
        self.assertAlmostEqual(result[0, 1], 0.0)
        self.assertAlmostEqual(result[1, 1], 100.0)

    def test_union_matrix_returns_pairwise_areas(self):
        result = metrics.det.box_union_matrix(
            [[10, 10, 20, 20], [0, 0, 10, 10]],
            [[12, 12, 22, 22], [0, 0, 10, 10]],
        )

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (2, 2))
        self.assertAlmostEqual(result[0, 0], 136.0)
        self.assertAlmostEqual(result[0, 1], 200.0)
        self.assertAlmostEqual(result[1, 1], 100.0)

    def test_empty_boxes_return_empty_matrix(self):
        result = metrics.det.iou([], [[0, 0, 10, 10]])

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (0, 1))

    def test_zero_area_boxes_return_zero_when_union_is_zero(self):
        result = metrics.det.iou([0, 0, 0, 0], [0, 0, 0, 0])

        self.assertEqual(result, 0.0)

    def test_invalid_box_shape_raises_error(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            metrics.det.iou([0, 0, 10], [0, 0, 10, 10])

    def test_invalid_box_coordinates_raise_error(self):
        with self.assertRaisesRegex(ValueError, "x2 >= x1"):
            metrics.det.iou([10, 0, 0, 10], [0, 0, 10, 10])


if __name__ == "__main__":
    unittest.main()
