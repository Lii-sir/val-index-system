# -*- coding: utf-8 -*-
"""目标检测格式适配器的单元测试。"""

import unittest

from valindex import metrics


class DetectionAdapterTest(unittest.TestCase):
    def test_yolo_target_is_converted_to_xyxy(self):
        result = metrics.det.yolo_to_detection(
            "1 0.5 0.5 0.2 0.4",
            image_size=(100, 200),
        )

        self.assertEqual(result["boxes"], [[80.0, 30.0, 120.0, 70.0]])
        self.assertEqual(result["labels"], [1])
        self.assertEqual(result["scores"], [1.0])

    def test_yolo_prediction_uses_score_column(self):
        result = metrics.det.yolo_to_detection(
            [[2, 0.5, 0.5, 0.2, 0.4, 0.85]],
            image_size=(100, 200),
        )

        self.assertEqual(result["labels"], [2])
        self.assertEqual(result["scores"], [0.85])

    def test_yolo_batch_is_converted_to_detections(self):
        result = metrics.det.yolo_to_detections(
            [
                {
                    "image_id": "img_001",
                    "labels": "0 0.5 0.5 0.2 0.4 0.90",
                    "image_size": (100, 200),
                },
                {
                    "image_id": "img_002",
                    "labels": [],
                    "image_size": (100, 200),
                },
            ]
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["image_id"], "img_001")
        self.assertEqual(result[0]["boxes"], [[80.0, 30.0, 120.0, 70.0]])
        self.assertEqual(result[0]["scores"], [0.9])
        self.assertEqual(result[1]["image_id"], "img_002")
        self.assertEqual(result[1]["boxes"], [])

    def test_yolo_can_clip_out_of_image_box(self):
        result = metrics.det.yolo_to_detection(
            [[0, 0.0, 0.5, 0.4, 0.2]],
            image_size=(100, 200),
            clip_boxes=True,
        )

        self.assertEqual(result["boxes"], [[0.0, 40.0, 40.0, 60.0]])

    def test_coco_annotation_is_converted_to_xyxy(self):
        result = metrics.det.coco_annotations_to_detection(
            [{"bbox": [10, 20, 30, 40], "category_id": 17, "score": 0.9}],
        )

        self.assertEqual(result["boxes"], [[10.0, 20.0, 40.0, 60.0]])
        self.assertEqual(result["labels"], [17])
        self.assertEqual(result["scores"], [0.9])

    def test_coco_category_ids_can_be_mapped(self):
        category_map = metrics.det.build_coco_category_id_map(
            [{"id": 17, "name": "cat"}, {"id": 42, "name": "dog"}],
        )
        result = metrics.det.coco_annotations_to_detection(
            [{"bbox": [0, 0, 10, 20], "category_id": 42}],
            category_id_map=category_map,
        )

        self.assertEqual(category_map, {17: 0, 42: 1})
        self.assertEqual(result["labels"], [1])
        self.assertEqual(result["scores"], [1.0])

    def test_coco_file_data_is_grouped_by_image(self):
        result = metrics.det.coco_to_detections(
            {
                "images": [{"id": 1}, {"id": 2}],
                "annotations": [
                    {"image_id": 1, "bbox": [0, 0, 10, 10], "category_id": 0},
                ],
            },
        )

        self.assertEqual([item["image_id"] for item in result], [1, 2])
        self.assertEqual(result[0]["labels"], [0])
        self.assertEqual(result[1]["boxes"], [])

    def test_coco_result_list_is_grouped_by_image(self):
        result = metrics.det.coco_to_detections(
            [
                {
                    "image_id": 2,
                    "bbox": [10, 10, 20, 20],
                    "category_id": 1,
                    "score": 0.7,
                },
                {
                    "image_id": 1,
                    "bbox": [0, 0, 5, 5],
                    "category_id": 0,
                    "score": 0.8,
                },
            ],
        )

        self.assertEqual([item["image_id"] for item in result], [1, 2])
        self.assertEqual(result[0]["scores"], [0.8])
        self.assertEqual(result[1]["scores"], [0.7])

    def test_yolo_invalid_label_raises_error(self):
        with self.assertRaises(ValueError):
            metrics.det.yolo_to_detection(
                "0 0.5 0.5 0.0 0.2",
                image_size=(100, 100),
            )


if __name__ == "__main__":
    unittest.main()
