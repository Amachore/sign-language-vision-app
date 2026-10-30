"""Hand geometry and stable rule-based pose recognition."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence, Tuple

Point = Tuple[float, float, float]


@dataclass(frozen=True)
class HandMetrics:
	normalized_landmarks: Tuple[Point, ...]
	thumb_index_distance: float
	index_middle_angle: Optional[float]


def euclidean_distance(first: Point, second: Point) -> float:
	return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def angle_degrees(first: Point, vertex: Point, last: Point) -> Optional[float]:
	"""Return the angle at vertex, or None when either vector is degenerate."""
	left = tuple(a - b for a, b in zip(first, vertex))
	right = tuple(a - b for a, b in zip(last, vertex))
	left_norm = math.sqrt(sum(value * value for value in left))
	right_norm = math.sqrt(sum(value * value for value in right))
	if left_norm == 0.0 or right_norm == 0.0:
		return None
	cosine = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
	return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def measure_hand(landmarks: Sequence[Point]) -> Optional[HandMetrics]:
	"""Normalize 21 MediaPipe points to the wrist and palm scale."""
	if len(landmarks) != 21:
		return None
	wrist = landmarks[0]
	palm_scale = euclidean_distance(wrist, landmarks[9])
	if palm_scale < 1e-6:
		return None
	normalized = tuple(
		tuple((coordinate - origin) / palm_scale for coordinate, origin in zip(point, wrist))
		for point in landmarks
	)
	return HandMetrics(
		normalized_landmarks=normalized,
		thumb_index_distance=euclidean_distance(normalized[4], normalized[8]),
		index_middle_angle=angle_degrees(normalized[0], normalized[5], normalized[8]),
	)


def classify_pose(landmarks: Sequence[Point]) -> Tuple[Optional[str], float]:
	"""Classify five explicit hand poses from normalized landmark geometry."""
	metrics = measure_hand(landmarks)
	if metrics is None:
		return None, 0.0
	points = metrics.normalized_landmarks
	wrist = points[0]
	ratios = []
	for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
		pip_distance = euclidean_distance(points[pip], wrist)
		ratios.append(euclidean_distance(points[tip], wrist) / max(pip_distance, 1e-6))
	thumb_ip_distance = euclidean_distance(points[3], wrist)
	thumb_ratio = euclidean_distance(points[4], wrist) / max(thumb_ip_distance, 1e-6)
	extended = [ratio > 1.12 for ratio in ratios]
	thumb_extended = thumb_ratio > 1.08

	if all(extended) and thumb_extended:
		pose = "OPEN_PALM"
	elif not any(extended) and not thumb_extended:
		pose = "FIST"
	elif extended[0] and extended[1] and not extended[2] and not extended[3]:
		pose = "PEACE"
	elif extended[0] and not any(extended[1:]):
		pose = "POINT"
	elif thumb_extended and not any(extended):
		pose = "THUMBS_UP"
	else:
		return None, 0.0

	margins = [abs(ratio - 1.12) for ratio in ratios]
	margins.append(abs(thumb_ratio - 1.08))
	confidence = min(1.0, 0.7 + min(margins) * 0.5)
	return pose, confidence


class StabilityFilter:
	def __init__(self, window_size: int = 5, confidence_threshold: float = 0.72) -> None:
		if window_size < 1:
			raise ValueError("window_size must be at least one")
		self.window_size = window_size
		self.confidence_threshold = confidence_threshold
		self._window: Deque[Tuple[str, float]] = deque(maxlen=window_size)
		self._last_emitted: Optional[str] = None

	def reset(self) -> None:
		self._window.clear()
		self._last_emitted = None

	def update(self, label: Optional[str], confidence: float) -> Optional[str]:
		if label is None or confidence < self.confidence_threshold:
			self.reset()
			return None
		self._window.append((label, confidence))
		if len(self._window) != self.window_size:
			return None
		if any(item[0] != label for item in self._window):
			return None
		if label == self._last_emitted:
			return None
		self._last_emitted = label
		return label