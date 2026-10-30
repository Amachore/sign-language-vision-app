import cv2
import mediapipe as mp


class HandTracker:
	def __init__(self, max_hands: int = 2) -> None:
		self._tracker = mp.solutions.hands.Hands(
			static_image_mode=False,
			max_num_hands=max_hands,
			model_complexity=0,
			min_detection_confidence=0.55,
			min_tracking_confidence=0.55,
		)

	def process(self, rgb_frame):
		result = self._tracker.process(rgb_frame)
		hands = {}
		landmarks = result.multi_hand_landmarks or []
		handedness = result.multi_handedness or []
		for hand, side_result in zip(landmarks, handedness):
			classification = side_result.classification[0]
			side = classification.label
			points = tuple((float(point.x), float(point.y), float(point.z))
						   for point in hand.landmark)
			current = hands.get(side)
			if current is None or classification.score > current[1]:
				hands[side] = (points, float(classification.score), hand)
		return hands, result

	def close(self) -> None:
		self._tracker.close()

