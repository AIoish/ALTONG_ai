import json
import tempfile
import unittest
from pathlib import Path

from filtering_training.external.audit_external_translations import audit


class ExternalTranslationAuditTests(unittest.TestCase):
    def test_flags_missing_information_without_approving_other_rows(self):
        rows = [
            {"source_id": "one", "source_title": "Meeting in 15 minutes",
             "source_body": "Please join the meeting.",
             "title": "회의 15분 전", "body": "회의에 참여해 주세요.", "source_folder": "Work"},
            {"source_id": "two", "source_title": "Budget review",
             "source_body": "Please check 3 budget items.",
             "title": "예산 검토", "body": "", "source_folder": "Work"},
            {"source_id": "three", "source_title": "Status update",
             "source_body": "Everything is ready.",
             "title": "예산 검토", "body": "", "source_folder": "Alerts"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draft.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                            encoding="utf-8")
            report = audit(path)
        self.assertEqual(report["count"], 3)
        self.assertEqual(report["flagged_count"], 2)
        self.assertNotIn("one", report["flagged_source_ids"])
        self.assertIn("empty_field", report["flagged_source_ids"]["two"])
        self.assertIn("source_number_missing", report["flagged_source_ids"]["two"])
        self.assertIn("duplicate_translation", report["flagged_source_ids"]["three"])


if __name__ == "__main__":
    unittest.main()