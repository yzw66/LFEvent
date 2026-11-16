"""防止公开增强表混入原始帖子字段，并核对全部分割的样本覆盖。"""
import csv
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PublishedMetadataTests(unittest.TestCase):
    def test_only_requested_augmentation_columns_are_populated(self):
        allowed = {"sample_id", "caption", "theme", "type"}
        expected_fields = ["sample_id", "text", "image_name", "time", "label", "caption", "theme", "type", "tags"]
        expected_counts = {"NED": 16955, "CrisisMMD": 18082}
        for dataset, count in expected_counts.items():
            with self.subTest(dataset=dataset):
                with (ROOT / f"data/{dataset}.csv").open(encoding="utf-8", newline="") as stream:
                    reader = csv.DictReader(stream)
                    self.assertEqual(reader.fieldnames, expected_fields)
                    rows = list(reader)
                self.assertEqual(len(rows), count)
                ids = {row["sample_id"] for row in rows}
                self.assertEqual(len(ids), count)
                self.assertTrue(all(row[field] == "" for row in rows for field in expected_fields if field not in allowed))
                for setting in ["S1", "S2"]:
                    split_ids = set()
                    for part in ["train", "val", "test"]:
                        with (ROOT / f"splits/{dataset}/{setting}/{part}.csv").open(encoding="utf-8", newline="") as stream:
                            split_ids.update(row["sample_id"] for row in csv.DictReader(stream))
                    self.assertEqual(ids, split_ids)

if __name__ == "__main__":
    unittest.main()
