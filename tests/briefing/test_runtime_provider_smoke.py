from __future__ import annotations

import unittest

from briefing_training.smoke_test_runtime_provider import (
    DEFAULT_SAMPLE_DIR,
    load_json_array,
)


class RuntimeProviderSmokeTests(unittest.TestCase):
    def test_default_fixtures_are_json_object_arrays(self) -> None:
        notifications = load_json_array(DEFAULT_SAMPLE_DIR / "raw_notifications.json")
        results = load_json_array(DEFAULT_SAMPLE_DIR / "filter_results.json")

        self.assertGreater(len(notifications), 0)
        self.assertGreater(len(results), 0)
        self.assertIn("id", notifications[0])
        self.assertIn("notification_id", results[0])


if __name__ == "__main__":
    unittest.main()
