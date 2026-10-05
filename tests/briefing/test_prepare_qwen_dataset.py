from datetime import datetime
import json
import unittest

from briefing_training.prepare_dataset import (
    SCENARIOS,
    TARGETED_SCENARIO_NAMES,
    build_record,
    generate_augmented_records,
    generate_balanced_records,
    generate_records,
    normalized_body_fingerprint,
    training_messages,
)
from src.briefing.schema import FILTER_CATEGORIES


class PrepareQwenDatasetTests(unittest.TestCase):
    def test_all_official_categories_are_covered(self) -> None:
        self.assertEqual(
            {scenario.category for scenario in SCENARIOS},
            set(FILTER_CATEGORIES),
        )

    def test_default_scale_adds_targeted_train_and_validation_cases(self) -> None:
        train_records = generate_augmented_records(
            "train",
            base_count=1000,
            targeted_count=200,
        )
        validation_records = generate_augmented_records(
            "validation",
            base_count=120,
            targeted_count=40,
        )

        self.assertEqual(len(train_records), 1200)
        self.assertEqual(len(validation_records), 160)
        self.assertEqual(
            len({normalized_body_fingerprint(record) for record in train_records}),
            1200,
        )
        self.assertEqual(
            len(
                {
                    normalized_body_fingerprint(record)
                    for record in validation_records
                }
            ),
            160,
        )
        self.assertTrue(
            {record["case_id"] for record in train_records}.isdisjoint(
                record["case_id"] for record in validation_records
            )
        )
        self.assertEqual(
            sum(
                record["metadata"].get("augmentation") == "failure_targeted"
                for record in train_records
            ),
            200,
        )
        self.assertEqual(
            sum(
                record["metadata"].get("augmentation") == "failure_targeted"
                for record in validation_records
            ),
            40,
        )

    def test_targeted_augmentation_covers_observed_failure_families(self) -> None:
        records = generate_augmented_records(
            "train",
            base_count=80,
            targeted_count=len(TARGETED_SCENARIO_NAMES),
        )
        targeted_scenarios = {
            record["metadata"]["scenario"]
            for record in records
            if record["metadata"].get("augmentation") == "failure_targeted"
        }

        self.assertEqual(targeted_scenarios, set(TARGETED_SCENARIO_NAMES))

    def test_expanded_scenarios_balance_all_categories(self) -> None:
        scenario_counts = {
            category: sum(
                scenario.category == category for scenario in SCENARIOS
            )
            for category in FILTER_CATEGORIES
        }

        self.assertEqual(len(SCENARIOS), 32)
        self.assertEqual(set(scenario_counts.values()), {4})

        records = generate_balanced_records("train", 1000)
        category_counts = {
            category: sum(
                record["input"]["category"] == category for record in records
            )
            for category in FILTER_CATEGORIES
        }
        self.assertEqual(set(category_counts.values()), {125})

    def test_large_dataset_timestamps_are_valid(self) -> None:
        records = generate_balanced_records("train", 1000)

        for record in records:
            for notification in record["input"]["notifications"]:
                datetime.fromisoformat(
                    notification["timestamp"].replace("Z", "+00:00")
                )

    def test_training_messages_append_strict_json_answer(self) -> None:
        record = build_record(SCENARIOS[0], "train", 0)
        messages = training_messages(record)

        self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant"])
        self.assertEqual(
            json.loads(messages[-1]["content"]),
            record["target"],
        )
        self.assertNotIn("synthetic_train", messages[1]["content"])

    def test_fragmented_chat_scenarios_use_adjacent_ids_and_timestamps(self) -> None:
        fragmented_scenarios = {
            scenario.name: scenario
            for scenario in SCENARIOS
            if scenario.name.startswith("fragmented_")
        }

        self.assertEqual(len(fragmented_scenarios), 16)
        self.assertEqual(
            {scenario.category for scenario in fragmented_scenarios.values()},
            set(FILTER_CATEGORIES),
        )
        self.assertEqual(
            {
                category: sum(
                    scenario.category == category
                    for scenario in fragmented_scenarios.values()
                )
                for category in FILTER_CATEGORIES
            },
            {category: 2 for category in FILTER_CATEGORIES},
        )
        for scenario in fragmented_scenarios.values():
            record = build_record(scenario, "train", 0)
            group = record["input"]
            notifications = group["notifications"]

            self.assertGreaterEqual(len(notifications), 3)
            self.assertTrue(group["app_name"])
            self.assertTrue(group["sender"])
            self.assertEqual(
                {notification["title"] for notification in notifications},
                {group["sender"]},
            )
            self.assertEqual(
                [notification["id"].rsplit("_", 1)[-1] for notification in notifications],
                [str(index) for index in range(1, len(notifications) + 1)],
            )

            timestamps = [
                datetime.fromisoformat(notification["timestamp"].replace("Z", "+00:00"))
                for notification in notifications
            ]
            self.assertTrue(
                all(
                    (later - earlier).total_seconds() == 60
                    for earlier, later in zip(timestamps, timestamps[1:])
                )
            )

    def test_validation_uses_held_out_variant(self) -> None:
        scenario = SCENARIOS[0]
        train_record = build_record(scenario, "train", 0)
        validation_record = build_record(scenario, "validation", 0)

        self.assertNotEqual(
            train_record["input"]["sender"],
            validation_record["input"]["sender"],
        )
        self.assertNotEqual(
            train_record["input"]["notifications"][0]["body"],
            validation_record["input"]["notifications"][0]["body"],
        )

    def test_invalid_split_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "split must be"):
            build_record(SCENARIOS[0], "test", 0)


if __name__ == "__main__":
    unittest.main()
