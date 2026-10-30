"""Record and label personal ASL/FSL clips under models/personal_dataset/."""

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2

from .camera_processing import CameraManager

FIELDS = ("timestamp", "language", "sign_class", "signer_id", "split", "video_file")
MANIFEST_NAME = "dataset_manifest.csv"
MINIMUM_FRAMES = 12


def _normalize_key(key: int) -> int:
    return key + 32 if ord("A") <= key <= ord("Z") else key


def _recording_action(key: int, is_recording: bool) -> str | None:
    key = _normalize_key(key)
    if key == ord("q"):
        return "quit"
    if key == ord("r"):
        return "stop" if is_recording else "start"
    return None


def _read_words(language: str) -> List[str]:
    while True:
        raw = input(f"Enter exactly 3 {language} sign labels, separated by commas: ")
        words = [word.strip().upper() for word in raw.split(",") if word.strip()]
        if len(words) == 3 and len(set(words)) == 3:
            return words
        print("Please enter three different labels, for example: HELLO, THANK YOU, YES")


def _read_manifest(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def _append_manifest_rows(path: Path, rows: List[Dict[str, str]]) -> bool:
    if not rows:
        return True
    try:
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as output:
            csv_writer = csv.DictWriter(output, fieldnames=FIELDS)
            if write_header:
                csv_writer.writeheader()
            csv_writer.writerows(rows)
    except PermissionError:
        return False
    rows.clear()
    return True


def _labels_compatible(existing: List[str], selected: List[str]) -> bool:
    existing_labels = {label.strip().upper() for label in existing if label.strip()}
    selected_labels = {label.strip().upper() for label in selected if label.strip()}
    return existing_labels.issubset(selected_labels)


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
    return safe or "unknown"


def _start_writer(directory: Path, language: str, sign_class: str, signer_id: str,
                  width: int, height: int, fps: float):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = (f"{language}_{_safe_name(sign_class)}_signer_{_safe_name(signer_id)}_"
                f"{stamp}.mp4")
    path = directory / filename
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        writer.release()
        return None, None
    return writer, path


def run() -> int:
    root = Path(__file__).resolve().parents[1]
    dataset_dir = root / "models" / "personal_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dataset_dir / MANIFEST_NAME
    existing = _read_manifest(manifest_path)

    words = {"ASL": _read_words("ASL"), "FSL": _read_words("FSL")}
    existing_words = {
        language: sorted({row["sign_class"].strip().upper() for row in existing
                          if row.get("language", "").strip().upper() == language})
        for language in words
    }
    for language, labels in existing_words.items():
        if not _labels_compatible(labels, words[language]):
            print(f"Existing {language} labels are {', '.join(labels)}. Restart with the same three labels.")
            return 2

    signer_id = input("Signer ID (for example 01): ").strip()
    if not signer_id:
        print("Signer ID cannot be empty.")
        return 2

    camera = CameraManager()
    if not camera.is_open:
        print("Camera unavailable. Checked indexes 0, 1, and 2.")
        camera.close()
        return 1

    language = "ASL"
    selected_index = 0
    writer = None
    current_path = None
    current_class = ""
    frame_count = 0
    status = "Ready"
    pending_rows: List[Dict[str, str]] = []
    counts: Dict[tuple, int] = {}
    for row in existing:
        key = (row["language"].strip().upper(), row["sign_class"].strip().upper())
        counts[key] = counts.get(key, 0) + 1

    capture = camera.capture
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if capture is not None else 0.0
    if not 5.0 <= fps <= 60.0:
        fps = 20.0
    window_name = "Record and Label Your Signs"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def finish_clip() -> None:
        nonlocal writer, current_path, frame_count, status
        if writer is None:
            return
        writer.release()
        writer = None
        if frame_count >= MINIMUM_FRAMES and current_path is not None:
            previous = counts.get((language, current_class), 0)
            split = "validation" if (previous + 1) % 5 == 0 else "train"
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "language": language,
                "sign_class": current_class,
                "signer_id": signer_id,
                "split": split,
                "video_file": current_path.name,
            }
            pending_rows.append(row)
            counts[(language, current_class)] = previous + 1
            if _append_manifest_rows(manifest_path, pending_rows):
                status = f"Saved {current_class} as {split} ({frame_count} frames)"
            else:
                status = "Video saved; close dataset_manifest.csv to finish saving its row"
        else:
            status = f"Clip kept on disk but not added: record at least {MINIMUM_FRAMES} frames"
        current_path = None
        frame_count = 0

    try:
        while camera.is_open:
            if pending_rows and _append_manifest_rows(manifest_path, pending_rows):
                status = "Pending video rows saved to dataset_manifest.csv"
            ok, frame = camera.read()
            if not ok or frame is None:
                status = "Camera read failed"
                break
            height, width = frame.shape[:2]
            if writer is not None:
                writer.write(frame)
                frame_count += 1

            display = cv2.flip(frame, 1)
            chosen = words[language][selected_index]
            cv2.putText(display, f"DATASET MODE: {language} | LABEL: {chosen}", (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (70, 235, 190), 2, cv2.LINE_AA)
            state = f"RECORDING {frame_count} frames" if writer is not None else status
            cv2.putText(display, state, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (100, 190, 255) if writer is not None else (240, 240, 240), 2,
                        cv2.LINE_AA)
            controls = "A ASL | F FSL | 1-3 label | R start/stop | Q quit"
            cv2.putText(display, controls, (16, height - 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.imshow(window_name, display)
            key = _normalize_key(cv2.waitKey(1) & 0xFF)
            action = _recording_action(key, writer is not None)
            if action == "quit":
                finish_clip()
                if pending_rows and not _append_manifest_rows(manifest_path, pending_rows):
                      filenames = ", ".join(row["video_file"] for row in pending_rows)
                      print("Could not update dataset_manifest.csv. The video files are saved, "
                          f"but add their rows after closing the manifest: {filenames}")
                break
            if action == "stop":
                finish_clip()
                continue
            if action == "start":
                current_class = words[language][selected_index]
                writer, current_path = _start_writer(
                    dataset_dir, language, current_class, signer_id, width, height, fps
                )
                frame_count = 0
                status = "Video writer failed" if writer is None else "Recording"
                continue
            if writer is not None:
                continue
            if key == ord("a"):
                language = "ASL"
                selected_index = min(selected_index, len(words[language]) - 1)
                status = "Ready"
            elif key == ord("f"):
                language = "FSL"
                selected_index = min(selected_index, len(words[language]) - 1)
                status = "Ready"
            elif key in (ord("1"), ord("2"), ord("3")):
                selected_index = int(chr(key)) - 1
                status = "Ready"
    finally:
        finish_clip()
        camera.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())