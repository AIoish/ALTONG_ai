from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import patch

from briefing_training import evaluate
from briefing_training.prepare_conversation_repairs import repair_cases
from src.briefing.category_correction import correct_category, correction_manifest
from src.briefing.qwen_provider import QwenBriefingProvider
from src.briefing.pipeline import SessionBriefingService


def group(*bodies, title="동아리방", app="KakaoTalk.exe"):
    return {"app_name": app, "sender": "참가자", "notifications": [
        {"id": f"n_{i}", "body": body, "title": title, "timestamp": f"2026-10-10T09:{i:02d}:00Z"}
        for i, body in enumerate(bodies)]}


class CategoryCorrectionTests(unittest.TestCase):
    def test_multiple_source_signals_correct_category(self):
        cases = (
            (("사진 계정 로그인이 감지됐어요", "제 로그인은 아니에요", "기기 차단하고 세션 종료했어요"), "시스템/보안"),
            (("추천 야간 배치", "멈춰서 중단됐어요", "지금 재시작 중이에요"), "긴급 업무"),
            (("출판 행사 안내문", "검토하고 올려 주세요"), "일반 업무"),
            (("언어 상담 예약", "다음 주로 변경됐어요"), "개인 중요"),
            (("배송 완료됐어요", "택배함에 뒀어요"), "개인 중요"),
            (("물리치료 예약이 변경됐어요",), "개인 중요"),
            (("회의가 내일 오후 3시에 진행돼요",), "일정/회의"),
            (("엘리베이터 점검 일정이 변경됐어요",), "시스템/보안"),
            (("결제한 카드로 12000원이 환불되었습니다",), "개인 중요"),
            (("할인 쿠폰은 내일 만료됩니다",), "광고/홍보"),
        )
        for bodies, expected in cases:
            with self.subTest(bodies=bodies):
                result = correct_category(group(*bodies), "기타")
                self.assertEqual(result.category, expected)
                self.assertTrue(result.changed)

    def test_single_keywords_and_social_messages_abstain(self):
        for body in ("긴급 긴급!!!", "예약", "배송", "할인", "오늘 안내문 읽어봤어 재밌더라", "회의", "로그인"):
            with self.subTest(body=body):
                self.assertEqual(correct_category(group(body), "개인 일반").category, "개인 일반")

    def test_chat_room_title_sender_and_upstream_category_are_not_evidence(self):
        source = group("점심 먹었어", title="서버 장애 복구 회의")
        source.update(sender="비밀번호 재설정", category="긴급 업무", expected_category="시스템/보안")
        self.assertEqual(correct_category(source, "개인 일반").category, "개인 일반")

    def test_nonchat_notification_title_is_source_evidence(self):
        result = correct_category(group("모든 기기에서 로그아웃했습니다", app="AccountApp",
                                        title="비밀번호 재설정 완료"), "기타")
        self.assertEqual(result.category, "시스템/보안")

    def test_document_and_meeting_distinction(self):
        for body, expected in (("행사 안내문 검토하고 올려 주세요", "일반 업무"),
                               ("회의자료를 검토하고 올려 주세요", "일반 업무"),
                               ("안내문 검토 회의 내일 할게요", "일정/회의")):
            # Conflicting document + event cues retain a plausible model category.
            self.assertEqual(correct_category(group(body), expected).category, expected)

    def test_conflicting_topics_keep_model_prediction(self):
        result = correct_category(group("서버 장애 복구 완료", "치과 예약을 변경했어요"), "일정/회의")
        self.assertEqual(result.category, "일정/회의")
        self.assertTrue(result.conflicting)
        self.assertFalse(result.changed)

    def test_login_api_incident_is_not_account_security(self):
        result = correct_category(group("로그인 API에서 500 오류가 발생했습니다", "배포 후 정상화되었습니다"), "기타")
        self.assertEqual(result.category, "긴급 업무")

    def test_negated_and_hypothetical_incident_abstain(self):
        for text in ("서버 오류는 발생하지 않았어요", "서버 장애가 나면 연락해 주세요", "배포 실패는 아니에요"):
            with self.subTest(text=text):
                result = correct_category(group(text), "일반 업무")
                self.assertFalse(result.changed)

    def test_provider_corrects_before_dashboard_and_preserves_summary(self):
        summary = "원본 모델 요약은 바꾸지 않습니다."
        class Backend:
            def generate(self, messages):
                return json.dumps({"primary_category": "개인 일반", "summary_lines": [summary]})
        source = group("상담 예약이 내일 오후 2시로 변경됐어요")
        notifications = [{**n, "app_name": source["app_name"], "sender": source["sender"]}
                         for n in source["notifications"]]
        for enabled, expected in ((True, "개인 중요"), (False, "개인 일반")):
            provider = QwenBriefingProvider(backend=Backend(), allow_fallback=False, category_correction=enabled)
            report = SessionBriefingService(provider=provider).build(
                session_id="correction_test", notifications=notifications,
                filter_results=[{"notification_id": n["id"], "is_passed": False} for n in notifications])
            self.assertEqual(report.groups[0].primary_category, expected)
            self.assertEqual(report.groups[0].summary_lines, (summary,))
            self.assertEqual(set(report.to_dict()), {"session_id", "groups"})

    def test_evaluation_exposes_raw_and_corrected_scores(self):
        case = repair_cases("evaluation")[-1]
        raw = json.dumps({"primary_category": "개인 일반", "summary_lines": case["reference_summary_lines"]})
        for extra, expected in (([], 1.0), (["--no-category-correction"], 0.0)):
            output = io.StringIO()
            with patch("sys.argv", ["evaluate", "--task", "briefing", "--greedy"] + extra), \
                 patch.object(evaluate, "load_cases", return_value=[case]), \
                 patch.object(evaluate, "load_model", return_value=(None, None)), \
                 patch.object(evaluate, "generate_summary", return_value=(raw, 0.1)), \
                 redirect_stdout(output):
                evaluate.main()
            report = json.loads(output.getvalue())
            self.assertEqual(report["raw_category_accuracy"], 0.0)
            self.assertEqual(report["category_accuracy"], expected)
            self.assertEqual(report["raw_case_pass_rate"], 0.0)
            self.assertEqual(report["case_pass_rate"], expected)
            self.assertEqual(report["results"][0]["raw_response"], raw)
            self.assertEqual(report["results"][0]["summary_lines"], case["reference_summary_lines"])

    def test_invalid_json_is_not_repaired_using_category_rules(self):
        case = repair_cases("evaluation")[-1]
        output = io.StringIO()
        with patch("sys.argv", ["evaluate", "--task", "briefing"]), \
             patch.object(evaluate, "load_cases", return_value=[case]), \
             patch.object(evaluate, "load_model", return_value=(None, None)), \
             patch.object(evaluate, "generate_summary", return_value=("not JSON", 0.1)), \
             redirect_stdout(output):
            evaluate.main()
        report = json.loads(output.getvalue())
        self.assertEqual(report["structured_output_rate"], 0.0)
        self.assertEqual(report["category_correction_count"], 0)

    def test_manifest_is_json_serializable_and_has_no_evaluation_ids(self):
        manifest = correction_manifest()
        text = json.dumps(manifest, ensure_ascii=False)
        self.assertIn("category-correction-v1", text)
        self.assertNotIn("evaluation_repair", text)


if __name__ == "__main__":
    unittest.main()
