
from filtering_training.common.paths import TRAINING_ROOT
import json
from pathlib import Path

from src.filtering.schema import FilteringSample


DATASET_PATH = TRAINING_ROOT / "data" / "sample_notifications.jsonl"


def validate_dataset():
    valid_count = 0
    error_count = 0

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)
                FilteringSample.model_validate(data)

                valid_count += 1

            except Exception as error:
                error_count += 1

                print(f"[ERROR] line {line_number}")
                print(error)
                print()

    print("Dataset validation finished")
    print(f"Valid samples : {valid_count}")
    print(f"Invalid samples : {error_count}")


if __name__ == "__main__":
    validate_dataset()
