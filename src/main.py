"""Realtime camera application for hand poses, emotion, ASL, and FSL."""

from __future__ import annotations

import queue
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List 

import cv2

from .analytics import SessionAnalytics, SignPerformanceDashboard
from .camera_processing import CameraManager, MovingAverageFPS, draw_hud
from .emotion_monitor import EmotionMonitor, FaceDetector
from .gesture_recognition import HandTracker, mp
from .hand_tracking import StabilityFilter, classify_pose, measure_hand
from .sign_recognition import SignRecognizer


MODES = ("POSES", "EMOTION", "ASL", "FSL")
SUPPORTED_SIGNS = ["HELLO", "THANK_YOU", "GOODBYE", "PLEASE", "LOVE", "NAME"]
GROUND_TRUTH_KEYS = {ord("a"): "HELLO", ord("s"): "THANK_YOU", ord("d"): "GOODBYE",
					ord("f"): "PLEASE", ord("g"): "LOVE", ord("h"): "NAME"}


class SpeechWorker:
	def __init__(self) -> None:
		self._queue: queue.Queue = queue.Queue()
		self._stop = threading.Event()
		self.status = "Starting local speech"
		self._thread = threading.Thread(target=self._run, name="speech-output", daemon=True)
		self._thread.start()

	def speak(self, text: str) -> None:
		if not self._stop.is_set():
			self._queue.put(text)

	def close(self) -> None:
		self._stop.set()
		self._queue.put(None)
		self._thread.join(timeout=3.0)

	def _run(self) -> None:
		try:
			import pyttsx3

			engine = pyttsx3.init()
			self.status = "Local speech ready"
		except Exception as error:
			self.status = f"Speech unavailable: {error}"
			return
		while not self._stop.is_set():
			text = self._queue.get()
			if text is None:
				break
			try:
				engine.say(text)
				engine.runAndWait()
			except Exception as error:
				self.status = f"Speech error: {error}"
				break


def _put_text(frame, text: str, origin, color=(240, 240, 240), scale=0.52, thickness=1) -> None:
	cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
				thickness, cv2.LINE_AA)


def _draw_emotion_panel(frame, analytics: SessionAnalytics) -> None:
	summary = analytics.summary()
	height, width = frame.shape[:2]
	panel_width = min(330, max(240, width // 3))
	left = max(0, width - panel_width - 12)
	top = 120
	bottom = min(height - 8, top + 255)
	if bottom <= top:
		return
	panel = frame[top:bottom, left:width - 8].copy()
	cv2.rectangle(panel, (0, 0), (panel.shape[1], panel.shape[0]), (22, 30, 34), -1)
	cv2.addWeighted(panel, 0.82, frame[top:bottom, left:width - 8], 0.18, 0,
					frame[top:bottom, left:width - 8])
	x = left + 10
	y = top + 22
	_put_text(frame, f"SESSION {summary['duration_seconds']:.0f}s", (x, y), (80, 225, 190), 0.5, 2)
	y += 24
	_put_text(frame, f"Valid emotion samples: {summary['valid_samples']}", (x, y))
	y += 22
	_put_text(frame, f"Dominant: {summary['dominant_emotion']}", (x, y))
	y += 22
	_put_text(frame, f"Mean confidence: {summary['mean_confidence']:.0%}", (x, y))
	y += 24
	for label, count in sorted(summary["counts"].items(), key=lambda item: (-item[1], item[0]))[:7]:
		percentage = summary["percentages"][label]
		_put_text(frame, f"{label}: {count} ({percentage:.1f}%)", (x, y))
		y += 20


def _draw_sign_performance_panel(frame, dashboard: SignPerformanceDashboard, predicted_label: str, status_text: str) -> None:
	if dashboard is None:
		return
	summary = dashboard.summary()
	height, width = frame.shape[:2]
	panel_width = min(430, max(320, width // 2))
	left = max(0, width - panel_width - 12)
	top = 10
	bottom = min(height - 10, top + 300)
	if bottom <= top:
		return
	panel = frame[top:bottom, left:width - 8].copy()
	cv2.rectangle(panel, (0, 0), (panel.shape[1], panel.shape[0]), (18, 18, 18), -1)
	cv2.addWeighted(panel, 0.72, frame[top:bottom, left:width - 8], 0.28, 0,
					frame[top:bottom, left:width - 8])
	x = left + 10
	y = top + 18
	_put_text(frame, "SIGN PERFORMANCE", (x, y), (80, 225, 190), 0.56, 2)
	y += 18
	_put_text(frame, f"Trials: {summary['trial_count']} | Acc: {summary['overall_accuracy']:.0%}", (x, y), (240, 240, 240), 0.45)
	y += 18
	_put_text(frame, f"Predicted: {predicted_label}", (x, y), (255, 220, 100), 0.45)
	y += 18
	_put_text(frame, f"Status: {status_text}", (x, y), (240, 240, 240), 0.43)
	y += 18
	for label in dashboard.supported_signs:
		metrics = summary["per_class"][label]
		_put_text(frame, f"{label}: TP{metrics['tp']} FP{metrics['fp']} FN{metrics['fn']} P{metrics['precision']:.2f} R{metrics['recall']:.2f}",
			  (x, y), (210, 210, 210), 0.38)
		y += 16
	matrix = summary["matrix"]
	base_y = y + 6
	_put_text(frame, "Confusion matrix (true -> predicted):", (x, base_y), (80, 225, 190), 0.38)
	row_y = base_y + 18
	for true_label in dashboard.supported_signs:
		row = [str(matrix.get(true_label, {}).get(predicted_label, 0)) for predicted_label in dashboard.supported_signs]
		_put_text(frame, f"{true_label}: {' '.join(row)}", (x, row_y), (220, 220, 220), 0.34)
		row_y += 14


def _export(analytics: SessionAnalytics, sign_text: List[Dict[str, str]], root: Path) -> None:
	paths = analytics.export()
	stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	history_path = root / "recordings" / f"recognized_sign_text_{stamp}.txt"
	history_path.parent.mkdir(parents=True, exist_ok=True)
	lines = [f"{item['timestamp']}\t{item['language']}\t{item['sign']}" for item in sign_text]
	history_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
	print("Exported:")
	for path in paths.values():
		print(f"  {path}")
	print(f"  {history_path}")


def run() -> int:
	root = Path(__file__).resolve().parents[1]
	camera = CameraManager()
	if not camera.is_open:
		print("Camera unavailable. Checked camera indexes 0, 1, and 2; connect a camera and retry.")
		camera.close()
		return 1

	hand_tracker = HandTracker(max_hands=2)
	face_detector = FaceDetector()
	emotion_monitor = EmotionMonitor()
	recognizers = {
		"ASL": SignRecognizer(root / "outputs" / "asl_sign_model.joblib"),
		"FSL": SignRecognizer(root / "outputs" / "fsl_sign_model.joblib"),
	}
	analytics = SessionAnalytics(root / "reports")
	performance_dashboard = SignPerformanceDashboard(SUPPORTED_SIGNS)
	fps_meter = MovingAverageFPS(frame_window=30, minimum_frames=15)
	pose_filters = {side: StabilityFilter(window_size=5) for side in ("Left", "Right")}
	speech = SpeechWorker()
	sign_text: List[Dict[str, str]] = []
	mode_index = 0
	frame_number = 0
	export_status = ""
	performance_status = "Awaiting 12 labeled trials"
	last_predicted_sign = "UNKNOWN"
	last_gesture_emotion = ""
	window_name = "Realtime ASL / FSL Vision"
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

	try:
		while camera.is_open:
			ok, frame = camera.read()
			if not ok or frame is None:
				print("Camera frame could not be read; stopping cleanly.")
				break
			frame_number += 1
			frame = cv2.flip(frame, 1)
			height, width = frame.shape[:2]
			rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			hands, hand_result = hand_tracker.process(rgb_frame)
			faces = face_detector.detect(rgb_frame, width, height)
			mode = MODES[mode_index]
			now_text = datetime.now().isoformat(timespec="seconds")

			if frame_number % 12 == 0 and faces:
				left, top, right, bottom = max(faces, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
				pad_x = max(8, (right - left) // 5)
				pad_y = max(8, (bottom - top) // 5)
				crop = frame[max(0, top - pad_y):min(height, bottom + pad_y),
							 max(0, left - pad_x):min(width, right + pad_x)]
				emotion_monitor.submit(crop)
			for observation in emotion_monitor.drain_observations():
				analytics.add_emotion(observation["timestamp"], observation["label"],
									  observation["confidence"], observation["uncertain"])
			emotion = emotion_monitor.latest()
			if emotion.get("status") == "ready" and emotion.get("label") != "Waiting for face":
				gesture_emotion = str(emotion["label"])
				emotion_confidence = float(emotion.get("confidence", 0.0))
				emotion_uncertain = bool(emotion.get("uncertain", True))
			else:
				gesture_emotion = "UNKNOWN"
				emotion_confidence = 0.0
				emotion_uncertain = True

			for side, (landmarks, handedness_confidence, raw_hand) in hands.items():
				mp.solutions.drawing_utils.draw_landmarks(
					frame, raw_hand, mp.solutions.hands.HAND_CONNECTIONS
				)
				metrics = measure_hand(landmarks)
				pose, confidence = classify_pose(landmarks)
				accepted_pose = None
				if mode == "POSES":
					accepted_pose = pose_filters[side].update(pose, confidence)
				else:
					pose_filters[side].reset()
				if accepted_pose:
					analytics.add_gesture(now_text, side, accepted_pose, confidence,
										  gesture_emotion, emotion_confidence, emotion_uncertain)
						last_gesture_emotion = f"{accepted_pose} | emotion: {gesture_emotion}"
				y = 142 + (0 if side == "Left" else 66)
				_put_text(frame, f"{side} hand ({handedness_confidence:.0%})", (12, y), (80, 225, 190), 0.55, 2)
				if metrics:
					angle = "N/A" if metrics.index_middle_angle is None else f"{metrics.index_middle_angle:.0f} deg"
					_put_text(frame, f"Thumb-index: {metrics.thumb_index_distance:.2f} palm | index angle: {angle}",
							  (12, y + 22))
				pose_text = accepted_pose or (pose if pose else "uncertain")
				_put_text(frame, f"Pose: {pose_text} ({confidence:.0%})", (12, y + 43))
			if not hands:
				for pose_filter in pose_filters.values():
					pose_filter.reset()
				_put_text(frame, "Hands: NONE DETECTED", (12, 142), (100, 190, 255), 0.58, 2)

			for index, (left, top, right, bottom) in enumerate(faces, start=1):
				cv2.rectangle(frame, (left, top), (right, bottom), (80, 225, 190), 2)
				_put_text(frame, f"Face {index}", (left, max(20, top - 8)), (80, 225, 190), 0.5, 2)
			if not faces:
				_put_text(frame, "Faces: NONE DETECTED", (12, 222 if hands else 166), (100, 190, 255), 0.58, 2)

			if emotion.get("status") == "ready" and emotion.get("label") != "Waiting for face":
				emotion_label = str(emotion["label"])
				if emotion.get("uncertain"):
					emotion_label = f"UNCERTAIN: {emotion_label}"
				_put_text(frame, f"Emotion: {emotion_label} ({float(emotion['confidence']):.0%})",
						  (12, 252 if hands else 196), (255, 220, 100), 0.56, 2)
			elif emotion.get("status") not in ("ready", "loading model"):
				_put_text(frame, "Emotion: model unavailable", (12, 252 if hands else 196),
						  (100, 190, 255), 0.52, 2)

			if mode in recognizers:
				sign = recognizers[mode].update({side: item[0] for side, item in hands.items()})
				last_predicted_sign = sign.label if sign.label else "UNKNOWN"
				label_color = (80, 225, 190) if sign.accepted else (235, 220, 150)
				sign_text_line = f"{mode}: {sign.label} ({sign.confidence:.0%}) - {sign.status}"
				_put_text(frame, sign_text_line, (12, height - 90), label_color, 0.58, 2)
				if sign.accepted:
					timestamp = datetime.now().isoformat(timespec="seconds")
					analytics.add_sign(timestamp, mode, sign.label, sign.confidence,
								   gesture_emotion, emotion_confidence, emotion_uncertain)
					sign_text.append({"timestamp": timestamp, "language": mode, "sign": sign.label,
								  "emotion": gesture_emotion})
					speech.speak(sign.label)
			elif mode == "POSES":
				_put_text(frame, "Rule-based poses: OPEN_PALM / FIST / PEACE / POINT / THUMBS_UP",
						  (12, height - 90), (235, 220, 150), 0.48)
			else:
				summary = analytics.summary()
				_draw_emotion_panel(frame, analytics)
				_put_text(frame, f"Emotion status: {emotion.get('status', 'waiting')}",
						  (12, height - 90), (235, 220, 150), 0.48)
				if summary["valid_samples"] == 0:
					_put_text(frame, "Waiting for valid classifier observations", (12, height - 68),
							  (235, 220, 150), 0.48)

			recent = " | ".join(f"{item['language']}:{item['sign']}" for item in sign_text[-3:])
			_put_text(frame, f"Text: {recent or '(empty)'}", (12, height - 48), (240, 240, 240), 0.48)
			if last_gesture_emotion:
				_put_text(frame, f"Last gesture: {last_gesture_emotion}", (12, height - 68),
						  (255, 220, 100), 0.45)
			_put_text(frame, "M mode | 1 poses | 2 emotion | 3 ASL | 4 FSL | A/S/D/F/G/H label | R reset | E export | Q quit",
					  (12, height - 18), (210, 210, 210), 0.43)
			if export_status:
				_put_text(frame, export_status, (max(12, width - 260), height - 18), (80, 225, 190), 0.45, 2)
			_draw_sign_performance_panel(frame, performance_dashboard, last_predicted_sign, performance_status)

			fps = fps_meter.update()
			draw_hud(frame, mode, fps, fps_meter.sample_frames)
			cv2.imshow(window_name, frame)
			key = cv2.waitKey(1) & 0xFF
			if key in (ord("q"), 27):
				break
			if key == ord("m"):
				mode_index = (mode_index + 1) % len(MODES)
			elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
				mode_index = int(chr(key)) - 1
			elif key in GROUND_TRUTH_KEYS:
				selected_label = GROUND_TRUTH_KEYS[key]
				trial_summary = performance_dashboard.record_trial(last_predicted_sign, selected_label)
				performance_dashboard.export_csv(root / "reports")
				performance_status = (
					f"Trial {performance_dashboard.trial_count}: pred={last_predicted_sign} true={selected_label} "
					f"acc={trial_summary['overall_accuracy']:.0%}"
				)
				if performance_dashboard.trial_count >= performance_dashboard.min_required_trials:
					performance_status += " | ready"
			elif key == ord("r"):
				analytics.reset()
				sign_text.clear()
				fps_meter.reset()
				for pose_filter in pose_filters.values():
					pose_filter.reset()
				for recognizer in recognizers.values():
					recognizer.reset()
				emotion_monitor.reset_session()
				export_status = "Session reset"
			elif key == ord("e"):
				_export(analytics, sign_text, root)
				export_status = "Export complete"
			if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
				break
	finally:
		camera.close()
		hand_tracker.close()
		face_detector.close()
		emotion_monitor.close()
		speech.close()
		cv2.destroyAllWindows()
	return 0


if __name__ == "__main__":
	raise SystemExit(run())