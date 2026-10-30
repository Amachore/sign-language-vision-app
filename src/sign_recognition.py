"""Landmark-sequence features and inference for compact ASL/FSL models."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .hand_tracking import Point, StabilityFilter, measure_hand

FRAME_COUNT = 12
FEATURES_PER_HAND = 64


def frame_feature(hands: Dict[str, Sequence[Point]]) -> Tuple[float, ...]:
	values: List[float] = []
	for side in ("Left", "Right"):
		metrics = measure_hand(hands.get(side, ()))
		if metrics is None:
			values.extend([0.0] * 63)
			values.append(0.0)
		else:
			for point in metrics.normalized_landmarks:
				values.extend(point)
			values.append(1.0)
	return tuple(values)


def sequence_feature(frames: Sequence[Sequence[float]], frame_count: int = FRAME_COUNT) -> Tuple[float, ...]:
	if not frames:
		return tuple([0.0] * (frame_count * FEATURES_PER_HAND * 2))
	if len(frames) == 1:
		sampled = [frames[0]] * frame_count
	else:
		sampled = [frames[round(index * (len(frames) - 1) / (frame_count - 1))]
				   for index in range(frame_count)]
	return tuple(value for frame in sampled for value in frame)


@dataclass(frozen=True)
class SignPrediction:
	label: str
	confidence: float
	accepted: bool
	status: str


class SignRecognizer:
	def __init__(self, model_path: Path, confidence_threshold: float = 0.45,
				 cooldown_seconds: float = 1.5) -> None:
		self.model_path = Path(model_path)
		self.confidence_threshold = confidence_threshold
		self.cooldown_seconds = cooldown_seconds
		self.frames = deque(maxlen=FRAME_COUNT)
		self.stability = StabilityFilter(window_size=5, confidence_threshold=confidence_threshold)
		self.last_accepted_at = 0.0
		self.bundle = None
		if self.model_path.exists():
			try:
				import joblib

				self.bundle = joblib.load(self.model_path)
			except Exception:
				self.bundle = None

	@property
	def is_ready(self) -> bool:
		return self.bundle is not None and "model" in self.bundle

	@property
	def vocabulary(self) -> List[str]:
		return list(self.bundle.get("vocabulary", [])) if self.is_ready else []

	def reset(self) -> None:
		self.frames.clear()
		self.stability.reset()
		self.last_accepted_at = 0.0

	def update(self, hands: Dict[str, Sequence[Point]]) -> SignPrediction:
		if not self.is_ready:
			return SignPrediction("Unknown", 0.0, False, "Train the sign model first")
		self.frames.append(frame_feature(hands))
		if len(self.frames) < FRAME_COUNT:
			self.stability.update(None, 0.0)
			return SignPrediction("Unknown", 0.0, False, "Collecting hand sequence")

		features = [sequence_feature(list(self.frames))]
		probabilities = self.bundle["model"].predict_proba(features)[0]
		order = probabilities.argsort()[::-1]
		best_index = int(order[0])
		confidence = float(probabilities[best_index])
		margin = confidence - float(probabilities[int(order[1])]) if len(order) > 1 else confidence
		label = str(self.bundle["model"].classes_[best_index])
		if confidence < self.confidence_threshold or margin < 0.05:
			self.stability.update(None, 0.0)
			return SignPrediction("Unknown", confidence, False, "Low-confidence / unsupported")

		if self.bundle.get("requires_two_hands", {}).get(label, False):
			if not hands.get("Left") or not hands.get("Right"):
				self.stability.update(None, 0.0)
				return SignPrediction("Unknown", confidence, False, "Both hands required")

		accepted = self.stability.update(label, confidence)
		now = time.monotonic()
		if accepted is not None:
			if now - self.last_accepted_at < self.cooldown_seconds:
				return SignPrediction(label, confidence, False, "Cooldown")
			self.last_accepted_at = now
			return SignPrediction(label, confidence, True, "Accepted")
		return SignPrediction(label, confidence, False, "Stabilizing")
