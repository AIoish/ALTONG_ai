import unittest

from briefing_training.prepare_dataset import SCENARIOS, build_record
from briefing_training.train_lora import (
    model_input_fingerprint,
    to_prompt_completion,
    validate_split_separation,
)


class TrainQwenLoraTests(unittest.TestCase):
    def test_prompt_completion_trains_only_on_assistant_json(self) -> None:
        record = build_record(SCENARIOS[0], "train", 0)
        example = to_prompt_completion(record)

        self.assertEqual(example["prompt"][-1]["role"], "user")
        self.assertTrue(example["prompt"][-1]["content"].endswith("/no_think"))
        self.assertEqual(example["completion"][0]["role"], "assistant")
        self.assertIn('"summary_lines"', example["completion"][0]["content"])

    def test_model_fingerprint_ignores_notification_ids(self) -> None:
        first = build_record(SCENARIOS[0], "train", 0)
        second = build_record(SCENARIOS[0], "train", 0)
        second["input"]["notifications"][0]["id"] = "another-id"

        self.assertEqual(
            model_input_fingerprint(first),
            model_input_fingerprint(second),
        )

    def test_duplicate_visible_inputs_across_splits_are_rejected(self) -> None:
        train_record = build_record(SCENARIOS[0], "train", 0)
        validation_record = build_record(SCENARIOS[0], "validation", 0)
        validation_record["input"] = train_record["input"]

        with self.assertRaisesRegex(ValueError, "duplicate inputs"):
            validate_split_separation([train_record], [validation_record])

    def test_distinct_splits_are_accepted(self) -> None:
        train_record = build_record(SCENARIOS[0], "train", 0)
        validation_record = build_record(SCENARIOS[0], "validation", 0)

        validate_split_separation([train_record], [validation_record])


if __name__ == "__main__":
    unittest.main()
