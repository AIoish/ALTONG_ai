import json
import tempfile
import unittest
from pathlib import Path

from filtering_training.external.select_external_candidates import (
    FOLDERS, eligible_record, select_candidates,
)


class ExternalCandidateSelectionTests(unittest.TestCase):
    def test_rejects_non_english_text_and_obvious_contact_patterns(self):
        base = {
            "id": "one",
            "notification": {"app_display_name": "Example", "title": "Status update",
                             "body": "The review is ready for you."},
            "classification": {"folder": "Work", "priority": 2},
        }
        self.assertEqual(eligible_record(base)[1], "eligible")
        changed = json.loads(json.dumps(base))
        changed["notification"]["body"] = "담당자가 검토했습니다."
        self.assertEqual(eligible_record(changed)[1], "non_english_script")
        changed["notification"]["body"] = "Contact me at person@example.com"
        self.assertEqual(eligible_record(changed)[1], "sensitive_pattern")

    def test_selection_balances_source_folders_without_creating_model_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            output = root / "selected.jsonl"
            rows = []
            for folder in FOLDERS:
                for index in range(3):
                    rows.append({
                        "id": f"{folder}-{index}",
                        "notification": {"app_display_name": "App A" if index == 0 else f"{folder} App B",
                                         "title": f"{folder} update {index}",
                                         "body": f"A distinct notice about {folder} item {index}."},
                        "classification": {"folder": folder, "priority": index + 1},
                    })
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            report = select_candidates(source, output, total=8, max_per_app=4)
            selected = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(report["selected_folder_counts"], {folder: 2 for folder in FOLDERS})
            self.assertEqual(len(selected), 8)
            self.assertLessEqual(report["max_app_count"], 4)
            self.assertTrue(all("label" not in row and "context" not in row for row in selected))
            first_output = output.read_bytes()
            select_candidates(source, output, total=8, max_per_app=4)
            self.assertEqual(output.read_bytes(), first_output)


if __name__ == "__main__":
    unittest.main()
