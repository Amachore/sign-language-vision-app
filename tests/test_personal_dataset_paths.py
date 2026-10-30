import csv
import tempfile
import unittest
from pathlib import Path

from src.record_dataset import _append_manifest_rows, _labels_compatible, _recording_action
from src.train_sign_models import _load_recorded_clips


class PersonalDatasetPathTests(unittest.TestCase):
    def test_r_toggles_recording_and_only_q_quits(self):
        self.assertEqual(_recording_action(ord("R"), False), "start")
        self.assertEqual(_recording_action(ord("R"), True), "stop")
        self.assertEqual(_recording_action(ord("R"), False), "start")
        self.assertEqual(_recording_action(ord("Q"), False), "quit")
        self.assertIsNone(_recording_action(27, False))

    def test_manifest_row_records_saved_video_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "dataset_manifest.csv"
            rows = [{"timestamp": "2026-10-30T12:00:00", "language": "ASL",
                     "sign_class": "HELLO", "signer_id": "01", "split": "train",
                     "video_file": "ASL_HELLO_signer_01.mp4"}]

            self.assertTrue(_append_manifest_rows(manifest, rows))
            with manifest.open(newline="", encoding="utf-8") as source:
                saved_rows = list(csv.DictReader(source))

            self.assertEqual(saved_rows[0]["video_file"], "ASL_HELLO_signer_01.mp4")
            self.assertEqual(rows, [])

    def test_existing_labels_can_be_a_subset_when_resuming_recording(self):
        self.assertTrue(_labels_compatible(["HELLO"], ["Hello", "More", "Yes"]))

    def test_conflicting_existing_labels_are_rejected(self):
        self.assertFalse(_labels_compatible(["HELLO", "PLEASE"], ["Hello", "More", "Yes"]))

    def test_training_uses_manifest_under_models_personal_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            personal = root / "models" / "personal_dataset"
            personal.mkdir(parents=True)
            clip = personal / "ASL_HELLO_signer_01.mp4"
            clip.write_bytes(b"my recording")
            manifest = personal / "dataset_manifest.csv"
            fields = ("timestamp", "language", "sign_class", "signer_id", "split", "video_file")
            with manifest.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"language": "ASL", "sign_class": "HELLO", "split": "train",
                                 "video_file": clip.name})

            result = _load_recorded_clips(root, "ASL")

            self.assertEqual([(item.label, item.path) for item in result],
                             [("HELLO", clip.resolve())])

    def test_recordings_output_manifest_is_not_used_for_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "recordings"
            output_dir.mkdir()
            clip = output_dir / "final-output.mp4"
            clip.write_bytes(b"application output")
            manifest = output_dir / "dataset_manifest.csv"
            fields = ("timestamp", "language", "sign_class", "signer_id", "split", "video_file")
            with manifest.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"language": "ASL", "sign_class": "OUTPUT", "split": "train",
                                 "video_file": clip.name})

            self.assertEqual(_load_recorded_clips(root, "ASL"), [])


if __name__ == "__main__":
    unittest.main()