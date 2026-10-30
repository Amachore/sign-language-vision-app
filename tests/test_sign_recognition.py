import unittest

from src.sign_recognition import frame_feature, sequence_feature


class SignRecognitionFeatureTests(unittest.TestCase):
    def test_frame_feature_preserves_left_right_presence(self):
        points = [(0.1, 0.1, 0.0)] * 21
        points[9] = (0.1, 0.4, 0.0)
        feature = frame_feature({"Left": points})

        self.assertEqual(len(feature), 128)
        self.assertEqual(feature[63], 1.0)
        self.assertEqual(feature[127], 0.0)

    def test_empty_frame_is_zero_filled(self):
        feature = frame_feature({})

        self.assertEqual(len(feature), 128)
        self.assertFalse(any(feature))

    def test_sequence_is_resampled_to_twelve_frames(self):
        first = [1.0] * 128
        last = [2.0] * 128
        sequence = sequence_feature([first, last])

        self.assertEqual(len(sequence), 12 * 128)
        self.assertEqual(sequence[:128], tuple(first))
        self.assertEqual(sequence[-128:], tuple(last))


if __name__ == "__main__":
    unittest.main()