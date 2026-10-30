"""Session metrics and non-destructive timestamped CSV/JSON exports."""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class SignPerformanceDashboard:
    """Tracks examiner-controlled ground-truth labels and model performance."""

    def __init__(self, supported_signs: Optional[List[str]] = None, min_required_trials: int = 12) -> None:
        self.supported_signs = self._normalise_supported_signs(supported_signs)
        self.min_required_trials = max(12, int(min_required_trials))
        self.trials: List[Dict[str, str]] = []

    @staticmethod
    def _normalise_label(label: Optional[str]) -> str:
        if label is None:
            return "UNKNOWN"
        value = str(label).strip().upper()
        return value or "UNKNOWN"

    @classmethod
    def _normalise_supported_signs(cls, supported_signs: Optional[List[str]]) -> List[str]:
        if supported_signs is None:
            supported_signs = ["HELLO", "THANK_YOU", "GOODBYE", "PLEASE", "LOVE", "NAME"]
        labels: List[str] = []
        seen = set()
        for label in supported_signs:
            normalised = cls._normalise_label(label)
            if normalised not in seen:
                labels.append(normalised)
                seen.add(normalised)
        if len(labels) < 6:
            fallback = ["HELLO", "THANK_YOU", "GOODBYE", "PLEASE", "LOVE", "NAME"]
            for label in fallback:
                if label not in seen:
                    labels.append(label)
                    seen.add(label)
                if len(labels) >= 6:
                    break
        return labels

    @property
    def trial_count(self) -> int:
        return len(self.trials)

    def clear(self) -> None:
        self.trials.clear()

    def record_trial(self, predicted_label: str, true_label: str, timestamp: Optional[str] = None) -> Dict[str, object]:
        trial = {
            "timestamp": timestamp or datetime.now().isoformat(timespec="seconds"),
            "predicted_label": self._normalise_label(predicted_label),
            "true_label": self._normalise_label(true_label),
        }
        self.trials.append(trial)
        return self.summary()

    def confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        matrix: Dict[str, Dict[str, int]] = {
            label: {predicted: 0 for predicted in self.supported_signs}
            for label in self.supported_signs
        }
        for trial in self.trials:
            true_label = trial["true_label"]
            predicted_label = trial["predicted_label"]
            if true_label not in matrix:
                matrix[true_label] = {predicted: 0 for predicted in self.supported_signs}
            if predicted_label not in matrix[true_label]:
                matrix[true_label][predicted_label] = 0
            matrix[true_label][predicted_label] += 1
        return matrix

    def summary(self) -> Dict[str, object]:
        matrix = self.confusion_matrix()
        per_class: Dict[str, Dict[str, float]] = {}
        correct = 0
        for label in self.supported_signs:
            tp = int(matrix.get(label, {}).get(label, 0))
            fp = sum(matrix.get(true_label, {}).get(label, 0) for true_label in self.supported_signs if true_label != label)
            fn = sum(matrix.get(label, {}).get(predicted_label, 0) for predicted_label in self.supported_signs if predicted_label != label)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            per_class[label] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": float(precision),
                "recall": float(recall),
            }
        for trial in self.trials:
            if trial["true_label"] == trial["predicted_label"] and trial["true_label"] in self.supported_signs:
                correct += 1
        overall_accuracy = (correct / len(self.trials)) if self.trials else 0.0
        return {
            "trial_count": len(self.trials),
            "overall_accuracy": float(overall_accuracy),
            "per_class": per_class,
            "supported_signs": list(self.supported_signs),
            "min_required_trials": self.min_required_trials,
            "ready_for_report": len(self.trials) >= self.min_required_trials,
            "matrix": matrix,
        }

    def export_csv(self, export_dir: Path) -> Dict[str, Path]:
        export_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = self.summary()
        metrics_path = export_dir / f"sign_performance_metrics_{stamp}.csv"
        matrix_path = export_dir / f"sign_confusion_matrix_{stamp}.csv"
        trial_path = export_dir / f"sign_labeled_trials_{stamp}.csv"
        metric_rows = []
        for label in self.supported_signs:
            metrics = summary["per_class"][label]
            metric_rows.append({
                "sign": label,
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
            })
        self._write_csv(metrics_path, ["sign", "tp", "fp", "fn", "precision", "recall"], metric_rows)
        matrix_rows = []
        for true_label, values in self.confusion_matrix().items():
            row = {"true_label": true_label}
            for predicted_label in self.supported_signs:
                row[predicted_label] = values.get(predicted_label, 0)
            matrix_rows.append(row)
        self._write_csv(matrix_path, ["true_label", *self.supported_signs], matrix_rows)
        self._write_csv(trial_path, ["timestamp", "predicted_label", "true_label"], self.trials)
        return {"metrics": metrics_path, "matrix": matrix_path, "trials": trial_path}

    @staticmethod
    def _write_csv(path: Path, fields: List[str], rows: List[Dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


class SessionAnalytics:
    def __init__(self, export_dir: Path) -> None:
        self.export_dir = Path(export_dir)
        self.reset()

    def reset(self) -> None:
        self.started_at = time.monotonic()
        self.emotions: List[Dict[str, object]] = []
        self.gestures: List[Dict[str, object]] = []
        self.signs: List[Dict[str, object]] = []

    @property
    def duration_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    def add_emotion(self, timestamp: str, label: str, confidence: float, uncertain: bool) -> None:
        self.emotions.append({
            "timestamp": timestamp,
            "emotion": label,
            "confidence": round(float(confidence), 5),
            "uncertain": bool(uncertain),
        })

    def add_gesture(self, timestamp: str, hand: str, pose: str, confidence: float) -> None:
        self.gestures.append({"timestamp": timestamp, "hand": hand, "pose": pose,
                              "confidence": round(float(confidence), 5)})

    def add_sign(self, timestamp: str, language: str, label: str, confidence: float) -> None:
        self.signs.append({"timestamp": timestamp, "language": language, "sign": label,
                           "confidence": round(float(confidence), 5)})

    def summary(self) -> Dict[str, object]:
        counts = Counter(str(item["emotion"]) for item in self.emotions)
        sample_count = len(self.emotions)
        dominant = counts.most_common(1)[0][0] if counts else "N/A"
        mean_confidence = (statistics.fmean(float(item["confidence"]) for item in self.emotions)
                           if sample_count else 0.0)
        return {
            "duration_seconds": round(self.duration_seconds, 2),
            "valid_samples": sample_count,
            "counts": dict(counts),
            "percentages": {name: round(count * 100.0 / sample_count, 2)
                            for name, count in counts.items()} if sample_count else {},
            "dominant_emotion": dominant,
            "mean_confidence": round(mean_confidence, 5),
        }

    def export(self) -> Dict[str, Path]:
        self.export_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = self.summary()
        counts = summary["counts"]
        percentages = summary["percentages"]
        common = {
            "valid_samples": summary["valid_samples"],
            "duration_seconds": summary["duration_seconds"],
            "dominant_emotion": summary["dominant_emotion"],
            "mean_confidence": summary["mean_confidence"],
        }
        paths = {
            "observations": self.export_dir / f"emotion_observations_{stamp}.csv",
            "aggregate_csv": self.export_dir / f"emotion_summary_{stamp}.csv",
            "aggregate_json": self.export_dir / f"emotion_summary_{stamp}.json",
            "gestures": self.export_dir / f"gesture_history_{stamp}.csv",
            "signs": self.export_dir / f"sign_history_{stamp}.csv",
        }
        self._write_csv(paths["observations"], ["timestamp", "emotion", "confidence", "uncertain"], self.emotions)
        aggregate_rows = [dict(emotion=name, count=count, percentage=percentages[name], **common)
                          for name, count in counts.items()]
        if not aggregate_rows:
            aggregate_rows = [dict(emotion="N/A", count=0, percentage=0.0, **common)]
        self._write_csv(paths["aggregate_csv"],
                        ["emotion", "count", "percentage", "valid_samples", "duration_seconds",
                         "dominant_emotion", "mean_confidence"], aggregate_rows)
        paths["aggregate_json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self._write_csv(paths["gestures"], ["timestamp", "hand", "pose", "confidence"], self.gestures)
        self._write_csv(paths["signs"], ["timestamp", "language", "sign", "confidence"], self.signs)
        return paths

    @staticmethod
    def _write_csv(path: Path, fields: List[str], rows: List[Dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
