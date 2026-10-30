"""Train ASL/FSL models only from personally recorded, labeled clips."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
from .gesture_recognition import HandTracker
from .sign_recognition import FRAME_COUNT, frame_feature, sequence_feature


@dataclass(frozen=True)
class Clip:
    label: str
    split: str
    path: Path


def _load_recorded_clips(root: Path, language: str) -> List[Clip]:
    dataset_dir = (root / "models" / "personal_dataset").resolve()
    manifest = dataset_dir / "dataset_manifest.csv"
    if not manifest.is_file():
        return []
    clips = []
    with manifest.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            if row.get("language", "").strip().upper() != language:
                continue
            split = row.get("split", "").strip().lower()
            filename = row.get("video_file", "").strip()
            if split not in ("train", "validation") or not filename:
                continue
            relative_path = Path(filename)
            if relative_path.name != filename or relative_path.suffix.lower() != ".mp4":
                continue
            video = (dataset_dir / relative_path).resolve()
            if video.parent == dataset_dir and video.is_file():
                label = row.get("sign_class", "").strip().upper()
                if label:
                    clips.append(Clip(label, split, video))
    return clips


def _extract_clip(path: Path, tracker: HandTracker):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return None
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        capture.release()
        return None
    indices = [round(index * (total_frames - 1) / (FRAME_COUNT - 1))
               for index in range(FRAME_COUNT)]
    features = []
    both_hands = 0
    detected_frames = 0
    try:
        for frame_index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            hands, _ = tracker.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarks = {side: details[0] for side, details in hands.items()}
            features.append(frame_feature(landmarks))
            if landmarks:
                detected_frames += 1
            if landmarks.get("Left") and landmarks.get("Right"):
                both_hands += 1
    finally:
        capture.release()
    if detected_frames < 3 or len(features) < 3:
        return None
    ratio = both_hands / max(1, detected_frames)
    return sequence_feature(features), ratio


def _train_language(root: Path, language: str) -> None:
    clips = _load_recorded_clips(root, language)
    vocabulary = sorted({clip.label for clip in clips})
    if len(vocabulary) != 3:
        raise RuntimeError(
            f"{language}: expected exactly 3 personally recorded labels in "
            f"models/personal_dataset/dataset_manifest.csv; "
            f"found {len(vocabulary)} ({', '.join(vocabulary) or 'none'})"
        )
    counts = Counter((clip.label, clip.split) for clip in clips)
    for label in vocabulary:
        if counts[(label, "train")] < 3 or counts[(label, "validation")] < 1:
            raise RuntimeError(
                f"{language} {label}: need at least 3 train clips and 1 validation clip; "
                f"found {counts[(label, 'train')]} train and {counts[(label, 'validation')]} validation"
            )
    features = {"train": [], "validation": []}
    labels = {"train": [], "validation": []}
    two_hand_ratios: Dict[str, List[float]] = defaultdict(list)
    tracker = HandTracker(max_hands=2)
    try:
        total = len(clips)
        for index, clip in enumerate(clips, start=1):
            extracted = _extract_clip(clip.path, tracker)
            if extracted is None:
                continue
            sequence, two_hand_ratio = extracted
            features[clip.split].append(sequence)
            labels[clip.split].append(clip.label)
            if clip.split == "train":
                two_hand_ratios[clip.label].append(two_hand_ratio)
            print(f"[{language}] {index}/{total} {clip.label}: {clip.split}", flush=True)
    finally:
        tracker.close()

    if (set(labels["train"]) != set(vocabulary)
            or set(labels["validation"]) != set(vocabulary)):
        raise RuntimeError(
            f"{language}: all three labels need usable hand landmarks in both train and validation clips"
        )

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    model = RandomForestClassifier(
        n_estimators=240,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features["train"], labels["train"])
    accuracy = float(accuracy_score(labels["validation"], model.predict(features["validation"])))
    actual_vocabulary = [str(label) for label in model.classes_]
    requires_two_hands = {
        label: (sum(two_hand_ratios[label]) / len(two_hand_ratios[label]) >= 0.6)
        if two_hand_ratios[label] else False
        for label in actual_vocabulary
    }
    bundle = {
        "model": model,
        "language": language,
        "vocabulary": actual_vocabulary,
        "requires_two_hands": requires_two_hands,
        "validation_accuracy": accuracy,
        "train_examples": len(labels["train"]),
        "validation_examples": len(labels["validation"]),
        "frame_count": FRAME_COUNT,
        "feature_version": 1,
    }
    import joblib

    output = root / "outputs" / f"{language.lower()}_sign_model.joblib"
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    print(f"[{language}] saved {output}")
    print(f"[{language}] validation accuracy: {accuracy:.1%} "
          f"({len(labels['validation'])} clips, {len(actual_vocabulary)} labels)")
    print(f"[{language}] vocabulary: {', '.join(actual_vocabulary)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("all", "asl", "fsl"), default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    languages = ("ASL", "FSL") if args.language == "all" else (args.language.upper(),)
    for language in languages:
        _train_language(root, language)


if __name__ == "__main__":
    main()