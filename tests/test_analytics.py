import tempfile
import unittest
import csv
from pathlib import Path

from src.analytics import SessionAnalytics


class SessionAnalyticsTests(unittest.TestCase):
    def test_summary_counts_percentages_and_mean_confidence(self):
        analytics = SessionAnalytics(Path(tempfile.gettempdir()))
        analytics.add_emotion("t1", "happy", 0.8, False)
        analytics.add_emotion("t2", "neutral", 0.6, True)
        analytics.add_emotion("t3", "happy", 0.9, False)

        summary = analytics.summary()

        self.assertEqual(summary["valid_samples"], 3)
        self.assertEqual(summary["dominant_emotion"], "happy")
        self.assertEqual(summary["counts"], {"happy": 2, "neutral": 1})
        self.assertEqual(summary["percentages"]["happy"], 66.67)
        self.assertAlmostEqual(summary["mean_confidence"], 0.76667, places=4)

    def test_export_writes_observations_and_aggregate_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            analytics = SessionAnalytics(Path(directory))
            analytics.add_emotion("t1", "happy", 0.8, False)

            paths = analytics.export()

            self.assertEqual(set(paths), {
                "observations", "aggregate_csv", "aggregate_json", "gestures", "signs"
            })
            self.assertTrue(all(path.is_file() for path in paths.values()))
            self.assertIn("happy", paths["aggregate_json"].read_text(encoding="utf-8"))

    def test_gesture_and_sign_exports_include_associated_emotion(self):
        with tempfile.TemporaryDirectory() as directory:
            analytics = SessionAnalytics(Path(directory))
            analytics.add_gesture("t1", "Left", "OPEN_PALM", 0.9, "happy", 0.82, False)
            analytics.add_sign("t2", "ASL", "HELLO", 0.88, "neutral", 0.73, False)

            paths = analytics.export()

            with paths["gestures"].open(encoding="utf-8", newline="") as source:
                gesture_row = next(csv.DictReader(source))
            with paths["signs"].open(encoding="utf-8", newline="") as source:
                sign_row = next(csv.DictReader(source))
            self.assertEqual(gesture_row["emotion"], "happy")
            self.assertEqual(gesture_row["emotion_confidence"], "0.82")
            self.assertEqual(sign_row["emotion"], "neutral")
            self.assertEqual(sign_row["emotion_uncertain"], "False")


if __name__ == "__main__":
    unittest.main()