from collections import Counter
import unittest

from briefing_training.evaluate import (
    _render_review_markdown,
    validate_evaluation_cases,
)
from briefing_training.smoke_test_model import load_cases
from src.briefing.schema import FILTER_CATEGORIES


EXPECTED_CATEGORY_COUNTS = {
    "긴급 업무": 3,
    "일반 업무": 3,
    "일정/회의": 3,
    "시스템/보안": 3,
    "개인 중요": 2,
    "개인 일반": 2,
    "광고/홍보": 2,
    "기타": 2,
}


class EvaluationDatasetTests(unittest.TestCase):
    def test_default_evaluation_set_has_twenty_reviewed_cases(self) -> None:
        cases = load_cases()

        validate_evaluation_cases(cases)

        self.assertEqual(len(cases), 20)
        self.assertEqual(
            Counter(case["input"]["category"] for case in cases),
            EXPECTED_CATEGORY_COUNTS,
        )
        self.assertEqual(
            {case["input"]["category"] for case in cases},
            set(FILTER_CATEGORIES),
        )

    def test_evaluation_cases_have_unique_ids_and_human_references(self) -> None:
        cases = load_cases()

        case_ids = [case["case_id"] for case in cases]
        self.assertEqual(len(case_ids), len(set(case_ids)))
        for case in cases:
            self.assertGreaterEqual(len(case["reference_summary_lines"]), 1)
            self.assertLessEqual(
                len(case["reference_summary_lines"]),
                case["max_summary_lines"],
            )

    def test_review_report_shows_source_reference_and_model_output(self) -> None:
        report = {
            "model": "Qwen/Qwen3-0.6B",
            "case_count": 1,
            "structured_output_rate": 1.0,
            "case_pass_rate": 1.0,
            "fact_coverage": 1.0,
            "results": [
                {
                    "case_id": "review_case",
                    "category": "기타",
                    "passed": True,
                    "fact_hits": 1,
                    "fact_total": 1,
                    "missing_expected_facts": [],
                    "line_limit_passed": True,
                    "forbidden_phrase_hits": [],
                    "notifications": [
                        {
                            "timestamp": "2026-09-30T00:00:00Z",
                            "title": "장소 정정",
                            "body": "장소가 B강의실로 변경되었습니다.",
                        }
                    ],
                    "reference_summary_lines": ["장소가 B강의실로 변경되었습니다."],
                    "summary_lines": ["장소는 B강의실입니다."],
                    "error": None,
                }
            ],
        }

        rendered = _render_review_markdown(report)

        self.assertIn("review_case — PASS", rendered)
        self.assertIn("### 입력 알림", rendered)
        self.assertIn("### 사람 기준 요약", rendered)
        self.assertIn("### 모델 요약", rendered)
        self.assertIn("B강의실", rendered)


if __name__ == "__main__":
    unittest.main()
