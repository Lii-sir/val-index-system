# -*- coding: utf-8 -*-
"""目标检测数据集级适配器的单元测试。"""

from __future__ import annotations

import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from valindex import metrics


class DetectionDatasetAdapterTest(unittest.TestCase):
    def test_load_detection_dataset_from_folders(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_dir = root / "images" / "scene"
            gt_dir = root / "gt" / "scene"
            pred_dir = root / "pred" / "scene"
            image_dir.mkdir(parents=True)
            gt_dir.mkdir(parents=True)
            pred_dir.mkdir(parents=True)

            image_path = image_dir / "img_001.bmp"
            self._write_bmp(image_path, width=200, height=100)

            (gt_dir / "img_001.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            (pred_dir / "img_001.txt").write_text("0 0.5 0.5 0.2 0.4 0.9\n", encoding="utf-8")

            predictions, targets = metrics.det.load_detection_dataset(
                image_dir=root / "images",
                gt_label_dir=root / "gt",
                pred_label_dir=root / "pred",
            )

            self.assertEqual(len(predictions), 1)
            self.assertEqual(len(targets), 1)
            self.assertEqual(predictions[0]["image_id"], "scene/img_001")
            self.assertEqual(targets[0]["image_id"], "scene/img_001")
            self.assertEqual(predictions[0]["boxes"], [[80.0, 30.0, 120.0, 70.0]])
            self.assertEqual(targets[0]["boxes"], [[80.0, 30.0, 120.0, 70.0]])
            self.assertEqual(predictions[0]["scores"], [0.9])
            self.assertEqual(targets[0]["scores"], [1.0])

    def test_load_detection_dataset_treats_missing_as_error_by_default(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "images").mkdir()
            (root / "gt").mkdir()
            (root / "pred").mkdir()
            self._write_bmp(root / "images" / "img_001.bmp", width=100, height=100)

            with self.assertRaises(FileNotFoundError):
                metrics.det.load_detection_dataset(
                    image_dir=root / "images",
                    gt_label_dir=root / "gt",
                    pred_label_dir=root / "pred",
                )

    def _write_bmp(self, path: Path, *, width: int, height: int) -> None:
        row_bytes = ((width * 3 + 3) // 4) * 4
        pixel_array_size = row_bytes * height
        file_size = 54 + pixel_array_size

        header = bytearray()
        header.extend(b"BM")
        header.extend(struct.pack("<I", file_size))
        header.extend(b"\x00\x00\x00\x00")
        header.extend(struct.pack("<I", 54))

        dib = bytearray()
        dib.extend(struct.pack("<I", 40))
        dib.extend(struct.pack("<i", width))
        dib.extend(struct.pack("<i", height))
        dib.extend(struct.pack("<H", 1))
        dib.extend(struct.pack("<H", 24))
        dib.extend(struct.pack("<I", 0))
        dib.extend(struct.pack("<I", pixel_array_size))
        dib.extend(struct.pack("<i", 2835))
        dib.extend(struct.pack("<i", 2835))
        dib.extend(struct.pack("<I", 0))
        dib.extend(struct.pack("<I", 0))

        pixels = bytes(pixel_array_size)
        path.write_bytes(bytes(header + dib) + pixels)


if __name__ == "__main__":
    unittest.main()
