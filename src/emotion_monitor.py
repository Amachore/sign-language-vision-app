"""Face detection and asynchronous DeepFace emotion inference."""

from __future__ import annotations

import queue
import threading
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp


class FaceDetector:
	def __init__(self, min_confidence: float = 0.55) -> None:
		self._detector = mp.solutions.face_detection.FaceDetection(
			model_selection=0, min_detection_confidence=min_confidence
		)

	def detect(self, rgb_frame, frame_width: int, frame_height: int) -> List[Tuple[int, int, int, int]]:
		result = self._detector.process(rgb_frame)
		boxes = []
		for detection in result.detections or []:
			box = detection.location_data.relative_bounding_box
			left = max(0, int(box.xmin * frame_width))
			top = max(0, int(box.ymin * frame_height))
			right = min(frame_width, int((box.xmin + box.width) * frame_width))
			bottom = min(frame_height, int((box.ymin + box.height) * frame_height))
			if right > left and bottom > top:
				boxes.append((left, top, right, bottom))
		return boxes[:2]

	def close(self) -> None:
		self._detector.close()


class EmotionMonitor:
	def __init__(self, confidence_threshold: float = 0.45, smoothing_window: int = 5) -> None:
		self.confidence_threshold = confidence_threshold
		self._queue: queue.Queue = queue.Queue(maxsize=1)
		self._stop = threading.Event()
		self._lock = threading.Lock()
		self._generation = 0
		self._score_history: Deque[Dict[str, float]] = deque(maxlen=smoothing_window)
		self._observations: Deque[Dict[str, object]] = deque()
		self._latest: Dict[str, object] = {
			"label": "Waiting for face",
			"confidence": 0.0,
			"uncertain": True,
			"status": "loading model",
		}
		self._thread = threading.Thread(target=self._run, name="emotion-inference", daemon=True)
		self._thread.start()

	def submit(self, face_crop) -> bool:
		if face_crop is None or face_crop.size == 0 or self._stop.is_set():
			return False
		if not self._queue.empty():
			return False
		try:
			self._queue.put_nowait((face_crop.copy(), self._generation))
			return True
		except queue.Full:
			return False

	def latest(self) -> Dict[str, object]:
		with self._lock:
			return dict(self._latest)

	def drain_observations(self) -> List[Dict[str, object]]:
		with self._lock:
			drained = list(self._observations)
			self._observations.clear()
			return drained

	def reset_session(self) -> None:
		with self._lock:
			self._generation += 1
			self._score_history.clear()
			self._observations.clear()
			self._latest = {"label": "Waiting for face", "confidence": 0.0,
							"uncertain": True, "status": "ready"}
		while True:
			try:
				self._queue.get_nowait()
			except queue.Empty:
				break

	def close(self) -> None:
		self._stop.set()
		try:
			self._queue.put_nowait(None)
		except queue.Full:
			pass
		self._thread.join(timeout=3.0)

	def _run(self) -> None:
		try:
			from deepface import DeepFace
		except Exception as error:
			with self._lock:
				self._latest = {"label": "Emotion model unavailable", "confidence": 0.0,
								"uncertain": True, "status": str(error)}
			return

		while not self._stop.is_set():
			try:
				item = self._queue.get(timeout=0.25)
			except queue.Empty:
				continue
			if item is None:
				break
			crop, generation = item
			try:
				prediction = DeepFace.analyze(
					img_path=crop,
					actions=["emotion"],
					enforce_detection=False,
					detector_backend="skip",
					silent=True,
				)
				if isinstance(prediction, list):
					prediction = prediction[0]
				scores = {str(name): float(value) / 100.0
						  for name, value in prediction["emotion"].items()}
				self._score_history.append(scores)
				labels = set().union(*(item.keys() for item in self._score_history))
				smoothed = {name: sum(item.get(name, 0.0) for item in self._score_history)
							/ len(self._score_history) for name in labels}
				label = max(smoothed, key=smoothed.get)
				confidence = smoothed[label]
				observation = {
					"timestamp": datetime.now().isoformat(timespec="seconds"),
					"label": label,
					"confidence": confidence,
					"uncertain": confidence < self.confidence_threshold,
				}
				with self._lock:
					if generation == self._generation:
						self._latest = {**observation, "status": "ready"}
						self._observations.append(observation)
			except Exception as error:
				with self._lock:
					if generation == self._generation:
						self._latest = {"label": "Emotion inference error", "confidence": 0.0,
										"uncertain": True, "status": str(error)}