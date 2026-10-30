"""Camera selection, frame-rate measurement, and live-frame overlays."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Deque, Optional, Tuple

import cv2


class CameraManager:
	def __init__(self, camera_indices: Tuple[int, ...] = (0, 1, 2)) -> None:
		self.capture: Optional[cv2.VideoCapture] = None
		self.camera_index: Optional[int] = None
		for index in camera_indices:
			capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
			if capture.isOpened():
				capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
				capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
				ok, _ = capture.read()
				if ok:
					self.capture = capture
					self.camera_index = index
					break
			capture.release()

	@property
	def is_open(self) -> bool:
		return self.capture is not None and self.capture.isOpened()

	def read(self):
		if not self.is_open:
			return False, None
		return self.capture.read()

	def close(self) -> None:
		if self.capture is not None:
			self.capture.release()
			self.capture = None


class MovingAverageFPS:
	def __init__(self, frame_window: int = 30, minimum_frames: int = 15) -> None:
		self.timestamps: Deque[float] = deque(maxlen=frame_window + 1)
		self.minimum_frames = minimum_frames

	def reset(self) -> None:
		self.timestamps.clear()

	def update(self, now: Optional[float] = None) -> Optional[float]:
		import time

		self.timestamps.append(time.perf_counter() if now is None else now)
		if len(self.timestamps) <= self.minimum_frames:
			return None
		elapsed = self.timestamps[-1] - self.timestamps[0]
		intervals = len(self.timestamps) - 1
		return intervals / elapsed if elapsed > 0 else None

	@property
	def sample_frames(self) -> int:
		return len(self.timestamps)


def draw_hud(frame, mode: str, fps: Optional[float], frame_count: int):
	height, width = frame.shape[:2]
	cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (12, 26),
				cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA)
	cv2.putText(frame, f"{width}x{height}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX,
				0.58, (240, 240, 240), 2, cv2.LINE_AA)
	fps_text = f"FPS avg: {fps:.1f}" if fps is not None else f"FPS avg: warming up ({frame_count}/16)"
	cv2.putText(frame, fps_text, (12, 78), cv2.FONT_HERSHEY_SIMPLEX,
				0.58, (240, 240, 240), 2, cv2.LINE_AA)
	cv2.putText(frame, f"MODE: {mode}", (12, 106), cv2.FONT_HERSHEY_SIMPLEX,
				0.68, (70, 235, 190), 2, cv2.LINE_AA)
	return frame
