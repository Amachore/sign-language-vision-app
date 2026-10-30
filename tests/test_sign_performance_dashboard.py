import unittest

from src.analytics import SignPerformanceDashboard


class SignPerformanceDashboardTests(unittest.TestCase):
    def test_per_class_metrics_handle_zero_denominators(self):
        dashboard = SignPerformanceDashboard(
            supported_signs=["HELLO", "THANK_YOU", "GOODBYE", "PLEASE", "LOVE", "NAME"]
        )

        dashboard.record_trial("HELLO", "HELLO")
        dashboard.record_trial("HELLO", "GOODBYE")
        dashboard.record_trial("THANK_YOU", "THANK_YOU")

        summary = dashboard.summary()

        self.assertAlmostEqual(summary["overall_accuracy"], 2 / 3)
        self.assertEqual(summary["per_class"]["GOODBYE"]["tp"], 0)
        self.assertEqual(summary["per_class"]["GOODBYE"]["fp"], 0)
        self.assertEqual(summary["per_class"]["GOODBYE"]["fn"], 1)
        self.assertEqual(summary["per_class"]["GOODBYE"]["precision"], 0.0)
        self.assertEqual(summary["per_class"]["GOODBYE"]["recall"], 0.0)

    def test_confusion_matrix_refreshes_after_each_trial(self):
        dashboard = SignPerformanceDashboard(
            supported_signs=["HELLO", "THANK_YOU", "GOODBYE", "PLEASE", "LOVE", "NAME"]
        )

        dashboard.record_trial("HELLO", "HELLO")
        dashboard.record_trial("HELLO", "GOODBYE")

        matrix = dashboard.confusion_matrix()
        self.assertEqual(matrix["HELLO"]["HELLO"], 1)
        self.assertEqual(matrix["GOODBYE"]["HELLO"], 1)
        self.assertEqual(dashboard.trial_count, 2)


if __name__ == "__main__":
    unittest.main()
