"""公开清单在没有帖子全文和图像文件时也能检查顺序、互斥及覆盖。"""
import csv
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PublishedSplitsTests(unittest.TestCase):
    def test_all_public_manifests(self):
        counts = {"NED": {"S1": [9846, 3535, 3574], "S2": [7170, 4877, 4908]},
                  "CrisisMMD": {"S1": [12715, 2680, 2687], "S2": [9071, 4503, 4508]}}
        for dataset, settings in counts.items():
            unions = []
            for setting, sizes in settings.items():
                seen = set()
                for part, expected in zip(["train", "val", "test"], sizes):
                    with self.subTest(dataset=dataset, setting=setting, part=part):
                        path = ROOT / "splits" / dataset / setting / f"{part}.csv"
                        with path.open(newline="", encoding="utf-8") as stream:
                            reader = csv.DictReader(stream)
                            self.assertEqual(reader.fieldnames, ["order", "sample_id"])
                            rows = list(reader)
                        self.assertEqual(len(rows), expected)
                        self.assertEqual([row["order"] for row in rows], list(map(str, range(expected))))
                        keys = [row["sample_id"] for row in rows]
                        prefix = "ned" if dataset == "NED" else "crisis"
                        self.assertTrue(all(re.fullmatch(prefix+r"_[0-9a-f]{64}", key) for key in keys))
                        self.assertEqual(len(keys), len(set(keys)))
                        self.assertFalse(seen.intersection(keys))
                        seen.update(keys)
                unions.append(seen)
            self.assertEqual(unions[0], unions[1])


if __name__ == "__main__":
    unittest.main()
