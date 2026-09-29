import json
import unittest

from briefing_training.prepare_dataset import (
    SCENARIOS,
    build_record,
    generate_records,
    training_messages,
)
from src.briefing.schema import FILTER_CATEGORIES


class PrepareQwenDatasetTests(unittest.TestCase):
    def test_all_official_categories_are_covered(self) -> None:
        self.assertEqual(
            {scenario.category for scenario in SCENARIOS},
            set(FILTER_CATEGORIES),
        )

    def test_default_scale_can_produce_200_train_and_40_validation_cases(self) -> None:
        train_records = generate_records("train", 20)
        validation_records = generate_records("validation", 4)

        self.assertEqual(len(train_records), 200)
        self.assertEqual(len(validation_records), 40)
        self.assertTrue(
            {record["case_id"] for record in train_records}.isdisjoint(
                record["case_id"] for record in validation_records
            )
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
