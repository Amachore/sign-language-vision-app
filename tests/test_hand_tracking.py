import unittest

from src.hand_tracking import StabilityFilter, angle_degrees, measure_hand


class HandTrackingTests(unittest.TestCase):
    def test_angle_uses_dot_product_geometry(self):
        self.assertAlmostEqual(angle_degrees((1, 0, 0), (0, 0, 0), (0, 1, 0)), 90.0)
        self.assertIsNone(angle_degrees((0, 0, 0), (0, 0, 0), (1, 0, 0)))

    def test_measure_hand_rejects_degenerate_palm(self):
        self.assertIsNone(measure_hand([(0.0, 0.0, 0.0)] * 21))

    def test_stability_requires_five_confident_frames_and_rearms(self):
        stable = StabilityFilter(window_size=5, confidence_threshold=0.7)
        for _ in range(4):
            self.assertIsNone(stable.update("OPEN_PALM", 0.9))
        self.assertEqual(stable.update("OPEN_PALM", 0.9), "OPEN_PALM")
        self.assertIsNone(stable.update("OPEN_PALM", 0.9))
        self.assertIsNone(stable.update("FIST", 0.4))
        for _ in range(4):
            self.assertIsNone(stable.update("FIST", 0.9))
        self.assertEqual(stable.update("FIST", 0.9), "FIST")


if __name__ == "__main__":
    unittest.main()