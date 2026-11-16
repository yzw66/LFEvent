"""检查固定分割的样本身份、顺序和完整恢复，以及错误清单的拒绝行为。"""
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

from prepare_data import build_plan, csv_payload, read_csv, sample_id, write_plan


class PrepareDataTests(unittest.TestCase):
    def test_missing_authorized_metadata_has_clear_cli_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(__file__).resolve().parents[1] / "prepare_data.py"
            result = subprocess.run([sys.executable, str(script), "--metadata-dir", temporary, "--datasets", "NED"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("data/README.md", result.stderr)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fields = ["sample_id", "text", "image_name", "time", "label", "caption", "theme", "type", "tags"]
        self.rows = []
        for i in range(4):
            row = {"text": f'含逗号,引号"和换行\n样本{i}', "image_name": "shared.jpg", "time": "2011-01-01",
                   "label": str(i % 2), "caption": f"caption {i}", "theme": "theme", "type": "", "tags": "[]"}
            self.rows.append({"sample_id": sample_id("NED", row), **row})
        self.save(self.root / "data/NED.csv", self.fields, self.rows)
        self.orders = {"S1": {"train": [2, 0], "val": [3], "test": [1]},
                       "S2": {"train": [3], "val": [1], "test": [0, 2]}}
        for setting, parts in self.orders.items():
            for part, positions in parts.items():
                self.save(self.manifest(setting, part), ["order", "sample_id"],
                          [{"order": n, "sample_id": self.rows[i]["sample_id"]} for n, i in enumerate(positions)])

    def save(self, path, fields, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(csv_payload(fields, rows))

    def manifest(self, setting="S1", part="train"):
        return self.root / "splits/NED" / setting / f"{part}.csv"

    def plan(self):
        return build_plan(self.root / "data", self.root / "splits", self.root / "output", ["NED"])

    def test_identity_distinguishes_shared_images_and_ignores_annotations(self):
        self.assertEqual(len({row["sample_id"] for row in self.rows}), 4)
        changed = {**self.rows[0], "label": "100", "caption": "different caption"}
        self.assertEqual(sample_id("NED", changed), self.rows[0]["sample_id"])
        self.assertNotEqual(sample_id("CrisisMMD", changed), self.rows[0]["sample_id"])

    def test_round_trip_preserves_values_and_manifest_order(self):
        plan = self.plan()
        write_plan(plan)
        fields = self.fields[1:]
        base = self.root / "output/NED/text/Qwen2-VL-7B-Instruct"
        for setting, parts in self.orders.items():
            suffix = "" if setting == "S1" else "2"
            for part, positions in parts.items():
                actual_fields, actual = read_csv(base / f"{part}{suffix}.csv")
                self.assertEqual(actual_fields, fields)
                self.assertEqual(actual, [{key: self.rows[i][key] for key in fields} for i in positions])
        before = {path: path.read_bytes() for path, _, _ in plan}
        write_plan(plan)
        self.assertEqual(before, {path: path.read_bytes() for path, _, _ in plan})

    def test_invalid_manifests_are_rejected(self):
        path = self.manifest()
        fields, original = read_csv(path)
        cases = [([{**original[0], "sample_id": "unknown"}, original[1]], "未知"),
                 ([{**original[0], "order": "1"}, original[1]], "order"),
                 ([original[0], {**original[1], "sample_id": original[0]["sample_id"]}], "重叠"),
                 ([original[0]], "未覆盖")]
        for rows, message in cases:
            with self.subTest(message=message):
                self.save(path, fields, rows)
                with self.assertRaisesRegex(ValueError, message):
                    self.plan()
        self.save(path, fields, original)
        fields, val = read_csv(self.manifest(part="val"))
        self.save(self.manifest(part="val"), fields, [{**val[0], "sample_id": original[0]["sample_id"]}])
        with self.assertRaisesRegex(ValueError, "重叠"):
            self.plan()

    def test_corrupted_master_identity_is_rejected(self):
        path = self.root / "data/NED.csv"
        self.save(path, self.fields, [self.rows[0], *self.rows])
        with self.assertRaisesRegex(ValueError, "重复"):
            self.plan()
        changed = [{**self.rows[0], "text": "modified"}, *self.rows[1:]]
        self.save(path, self.fields, changed)
        with self.assertRaisesRegex(ValueError, "不一致"):
            self.plan()

    def test_existing_different_data_is_protected_before_any_write(self):
        plan = self.plan()
        protected = plan[-1][0]
        protected.parent.mkdir(parents=True)
        protected.write_text("do not overwrite")
        with self.assertRaises(FileExistsError):
            write_plan(plan)
        self.assertFalse(plan[0][0].exists())
        self.assertEqual(protected.read_text(), "do not overwrite")
        write_plan(plan, overwrite=True)
        self.assertEqual(protected.read_bytes(), csv_payload(plan[-1][1], plan[-1][2]))

    def public_metadata(self):
        public = [{key: row[key] if key in {"sample_id", "caption", "theme", "type"} else ""
                   for key in self.fields} for row in self.rows]
        self.save(self.root / "data/NED.csv", self.fields, public)
        return public

    def test_public_augmentations_merge_by_content_id_and_preserve_values(self):
        self.rows[0]["caption"] = self.rows[0]["text"]
        self.public_metadata()
        extra = {**self.rows[0], "text": "extra sample outside the published corpus"}
        self.save(self.root / "raw/NED.csv", ["text", "image_name", "time", "label"],
                  [extra, *reversed(self.rows)])
        public_before = (self.root / "data/NED.csv").read_bytes()
        plan = build_plan(self.root / "data", self.root / "splits", self.root / "output", ["NED"],
                          raw_dir=self.root / "raw")
        write_plan(plan)
        self.assertEqual(public_before, (self.root / "data/NED.csv").read_bytes())
        for setting, parts in self.orders.items():
            suffix = "" if setting == "S1" else "2"
            for part, positions in parts.items():
                path = self.root / f"output/NED/text/Qwen2-VL-7B-Instruct/{part}{suffix}.csv"
                self.assertEqual(path.read_bytes(), csv_payload(self.fields[1:], [self.rows[i] for i in positions]))

    def test_unfilled_public_metadata_has_actionable_error(self):
        self.public_metadata()
        with self.assertRaisesRegex(ValueError, "--raw-dir"):
            self.plan()
        self.assertFalse((self.root / "output").exists())

    def test_missing_or_duplicate_raw_identity_is_rejected(self):
        self.public_metadata()
        path = self.root / "raw/NED.csv"
        fields = ["text", "image_name", "time", "label"]
        cases = [(self.rows[:-1], "缺少匹配"), ([*self.rows, self.rows[0]], "重复"),
                 ([{**self.rows[0], "time": "different"}, *self.rows[1:]], "缺少匹配")]
        for rows, message in cases:
            with self.subTest(message=message):
                self.save(path, fields, rows)
                with self.assertRaisesRegex(ValueError, message):
                    build_plan(self.root / "data", self.root / "splits", self.root / "output", ["NED"],
                               raw_dir=self.root / "raw")
        self.assertFalse((self.root / "output").exists())

    def test_local_raw_id_or_existing_metadata_conflict_is_rejected(self):
        public = self.public_metadata()
        path = self.root / "raw/NED.csv"
        self.save(path, self.fields, [{**self.rows[0], "sample_id": "wrong"}, *self.rows[1:]])
        with self.assertRaisesRegex(ValueError, "与内容不一致"):
            build_plan(self.root / "data", self.root / "splits", self.root / "output", ["NED"],
                       raw_dir=self.root / "raw")
        self.save(path, self.fields, self.rows)
        public[0]["label"] = "99"
        self.save(self.root / "data/NED.csv", self.fields, public)
        with self.assertRaisesRegex(ValueError, "字段冲突"):
            build_plan(self.root / "data", self.root / "splits", self.root / "output", ["NED"],
                       raw_dir=self.root / "raw")

if __name__ == "__main__":
    unittest.main()
