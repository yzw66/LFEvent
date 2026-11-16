"""验证图像路径、输出去重和相对路径约束，不下载视觉模型。"""
import csv
from pathlib import Path
import tempfile
import unittest

from extract_image_features import collect_images, image_path


class FeaturePathTests(unittest.TestCase):
    def test_ned_event_path_and_first_image(self):
        result = image_path(Path("dataset"), "NED", {"label": "2", "image_name": "first.jpg,second.jpg"})
        self.assertEqual(result, Path("dataset/NED/Images/E3/first.jpg"))

    def test_crisis_tweet_image_relative_path(self):
        row = {"label": "0", "image_name": "data_image/event/date/123_2.jpg"}
        self.assertEqual(image_path(Path("dataset"), "CrisisMMD", row), Path("dataset/MMD/Images/event/date/123_2.jpg"))

    def test_reject_paths_outside_dataset(self):
        for value in ["../image.jpg", "/tmp/image.jpg", "data_image/../../image.jpg"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                image_path(Path("dataset"), "CrisisMMD", {"image_name": value})

    def test_collect_images_deduplicates_feature_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "NED/text/Qwen2-VL-7B-Instruct/data.csv"
            path.parent.mkdir(parents=True)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["label", "image_name"])
                writer.writeheader()
                writer.writerows([{"label": "0", "image_name": name} for name in ["same.jpg", "same.jpg", "other.jpg"]])
            paths = collect_images(root, "NED")
            self.assertEqual(len(paths), 2)
            self.assertEqual([target.name for _, target in paths], ["same.pt", "other.pt"])


if __name__ == "__main__":
    unittest.main()
