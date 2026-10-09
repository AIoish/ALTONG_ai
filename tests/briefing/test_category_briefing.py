from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

from briefing_training import evaluate
from briefing_training import train_lora
from briefing_training.prepare_category_dataset import (
    TRAIN_PATH, VALIDATION_PATH, EVALUATION_PATH, migrate_record, validate_records,
    short_request_records, short_evaluation_records,
)
from briefing_training.prepare_dataset import SCENARIOS, build_record
from briefing_training.smoke_test_model import render_generation_prompt, load_cases
from briefing_training.smoke_test_runtime_provider import DEFAULT_SAMPLE_DIR, load_json_array
from briefing_training.train_lora import load_records, render_prompt_completion, validate_split_separation
from src.briefing import QwenBriefingProvider, SessionBriefingService
from src.briefing.category_prompt import build_messages, parse_response
from src.briefing.reports import category_report, render_category_markdown
from src.briefing.schema import FilterResult, FILTER_CATEGORIES


def notification(id, *, room="프로젝트방", sender="팀장", app="KakaoTalk.exe", body="회의 내일 할게요.", minute=0):
    return {"id": id, "app_name": app, "sender": sender, "title": room,
            "body": body, "timestamp": f"2026-10-08T09:{minute:02d}:00Z"}


def build(notifications, *, provider=None, filtering_category=None):
    results = [{"notification_id": n["id"], "is_passed": False} for n in notifications]
    if filtering_category:
        for result in results:
            result["category"] = filtering_category
    return SessionBriefingService(provider=provider).build(
        session_id="category_test", notifications=notifications, filter_results=results,
    )


class FakeBackend:
    def __init__(self, category="일정/회의"):
        self.category = category
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        return json.dumps({"primary_category": self.category, "summary_lines": ["내일 오후 1시에 회의합니다."]}, ensure_ascii=False)


class FakeTokenizer:
    eos_token = "<eos>"

    def apply_chat_template(self, messages, **kwargs):
        return json.dumps([messages, kwargs], ensure_ascii=False)


class CategoryBriefingTests(unittest.TestCase):
    def test_filter_category_is_optional(self):
        result = FilterResult.from_mapping({"notification_id": "n", "is_passed": False})
        self.assertIsNone(result.category)

    def test_same_sender_different_rooms_never_merge(self):
        report = build([notification("a"), notification("b", room="동아리방", minute=1)])
        self.assertEqual(len(report.groups), 2)
        self.assertEqual([g.room_name for g in report.groups], ["프로젝트방", "동아리방"])

    def test_same_room_different_senders_never_merge(self):
        self.assertEqual(len(build([notification("a"), notification("b", sender="팀원", minute=1)]).groups), 2)

    def test_uuid_fragments_merge_for_known_room_and_extract_times(self):
        items = [
            notification("d2b735b2-9b7d-497a-b4c8-4bcde667a4c1"),
            notification("7a53996d-644d-46f5-9966-2d326d02e1bd", body="오후 1시부터 2시까지요.", minute=1),
            notification("another-random-id", body="장소는 창의관 402호입니다.", minute=2),
        ]
        report = build(items)
        self.assertEqual(len(report.groups), 1)
        schedule = report.groups[0].schedule_summaries[0].to_dict()
        self.assertEqual(schedule["when"], "2026-10-09T04:00:00Z")
        self.assertEqual(schedule["end_at"], "2026-10-09T05:00:00Z")
        self.assertEqual(schedule["where"], "창의관 402호")

    def test_exe_alias_uses_same_app_identity(self):
        report = build([notification("a"), notification("b", app="KakaoTalk", minute=1)])
        self.assertEqual(len(report.groups), 1)

    def test_room_name_alone_does_not_merge_far_unrelated_topics(self):
        report = build([notification("a", body="요리 준비"), notification("b", body="자동차 수리", minute=10)])
        self.assertEqual(len(report.groups), 2)

    def test_over_thirty_minutes_splits_even_with_same_topic(self):
        self.assertEqual(len(build([notification("a"), notification("b", minute=31)]).groups), 2)

    def test_non_kakao_titles_are_not_room_ids(self):
        report = build([notification("a", app="Slack", room="회의 공지"), notification("b", app="Slack", room="회의 수정", minute=1)])
        self.assertEqual(len(report.groups), 1)
        self.assertIsNone(report.groups[0].room_name)

    def test_fallback_room_does_not_group_random_ids_by_time_alone(self):
        report = build([notification("random-a", room="카카오톡", body="요리 준비"), notification("random-b", room="카카오톡", body="자동차 수리", minute=1)])
        self.assertEqual(len(report.groups), 2)

    def test_model_owns_category_and_is_called_once(self):
        backend = FakeBackend()
        provider = QwenBriefingProvider(backend=backend, allow_fallback=False)
        report = build([notification("a")], provider=provider, filtering_category="광고/홍보")
        self.assertEqual(report.groups[0].primary_category, "일정/회의")
        self.assertEqual(len(backend.calls), 1)
        self.assertNotIn('"category"', backend.calls[0][-1]["content"])
        self.assertNotIn('"urgency_score"', backend.calls[0][-1]["content"])

    def test_rule_fallback_does_not_use_filter_labels(self):
        notifications = [notification("a", body="비밀번호 재설정이 완료됐어요.")]
        first = build(notifications, filtering_category="광고/홍보").to_dict()
        second = build(notifications, filtering_category="개인 일반").to_dict()
        self.assertEqual(first, second)
        self.assertEqual(first["groups"][0]["primary_category"], "시스템/보안")

    def test_classifier_category_can_trigger_schedule_without_keywords(self):
        backend = FakeBackend()
        report = build([notification("a", body="내일 오후 1시, 장소는 B강의실입니다.")],
                       provider=QwenBriefingProvider(backend=backend, allow_fallback=False))
        self.assertEqual(len(report.groups[0].schedule_summaries), 1)

    def test_schedule_is_not_limited_to_meeting_category(self):
        report = build([notification("a", body="내일 오후 1시 치과 예약입니다.")],
                       provider=QwenBriefingProvider(backend=FakeBackend("개인 중요"), allow_fallback=False))
        self.assertEqual(report.groups[0].primary_category, "개인 중요")
        self.assertEqual(len(report.groups[0].schedule_summaries), 1)

    def test_invalid_model_contract_does_not_silently_use_filter_labels(self):
        class OldBackend:
            def generate(self, messages):
                return '{"summary_lines":["기존 요약"]}'
        with self.assertRaisesRegex(ValueError, "primary_category"):
            build([notification("a")], provider=QwenBriefingProvider(backend=OldBackend(), allow_fallback=False))

    def test_category_parser_rejects_invalid_fields_or_labels(self):
        for payload in (
            {"primary_category": "업무", "summary_lines": ["요약"]},
            {"primary_category": "기타", "summary_lines": ["요약"], "extra": 1},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_response(json.dumps(payload))

    def test_category_report_preserves_cards_and_order_without_extra_inference(self):
        notifications = load_json_array(DEFAULT_SAMPLE_DIR / "category_notifications.json")
        report = build(notifications)
        grouped = category_report(report)
        self.assertEqual(len(report.groups), 4)
        sections = [section["category"] for section in grouped["categories"]]
        self.assertEqual(sections, [category for category in FILTER_CATEGORIES if category in sections])
        cards = [card for section in grouped["categories"] for card in section["groups"]]
        self.assertEqual(len(cards), 4)
        self.assertEqual(sum(len(card["schedule_summaries"]) for card in cards), 1)
        self.assertEqual(set(report.to_dict()), {"session_id", "groups"})
        markdown = render_category_markdown(report)
        self.assertIn("## 일반 업무", markdown)
        self.assertIn("## 일정/회의", markdown)
        self.assertIn("## 시스템/보안", markdown)
        self.assertIn("프로젝트방", markdown)
        self.assertIn("동아리방", markdown)

    def test_empty_category_report(self):
        report = build([])
        self.assertEqual(category_report(report), {"session_id": "category_test", "categories": []})
        self.assertIn("없습니다", render_category_markdown(report))

    def test_room_name_is_not_a_schedule_fact(self):
        report = build([notification("a", room="회의 취소 10월 12일 오후 3시방", body="자료 업로드 완료했습니다.")])
        self.assertEqual(report.groups[0].primary_category, "일반 업무")
        self.assertEqual(report.groups[0].schedule_summaries, ())

    def test_room_date_does_not_override_message_start(self):
        report = build([notification("a", room="10월 12일 오후 3시방", body="내일 오후 1시 회의입니다.")])
        self.assertEqual(report.groups[0].schedule_summaries[0].to_dict()["when"], "2026-10-09T04:00:00Z")

    def test_upload_and_notice_review_without_dates_are_not_schedules(self):
        report = build([notification("a", body="발표 자료 업로드 완료했습니다."),
                        notification("b", room="다른방", body="행사 안내문 검토해 주세요.")])
        self.assertTrue(all(not group.schedule_summaries for group in report.groups))

    def test_category_migration_preserves_original_and_moves_label_to_target(self):
        source = build_record(SCENARIOS[0], "train", 0)
        before = deepcopy(source)
        migrated = migrate_record(source)
        self.assertEqual(source, before)
        self.assertNotIn("category", migrated["input"])
        self.assertEqual(migrated["target"]["primary_category"], before["input"]["category"])
        validate_records([migrated], expected_split="train")

    def test_train_evaluate_and_runtime_have_identical_visible_prompt(self):
        record = migrate_record(build_record(SCENARIOS[0], "train", 0))
        tokenizer = FakeTokenizer()
        training = render_prompt_completion(record, tokenizer)["prompt"]
        for style in ("training", "runtime"):
            evaluated = render_generation_prompt(tokenizer, record["input"], task="briefing", prompt_style=style, max_summary_lines=record["max_summary_lines"])
            self.assertEqual(training, evaluated)
        self.assertNotIn('"category":', training)
        self.assertNotIn("/no_think", training)

    def test_upstream_category_is_not_exposed_even_if_supplied(self):
        record = migrate_record(build_record(SCENARIOS[0], "train", 0))
        record["input"]["category"] = "LABEL_MUST_NOT_LEAK"
        messages = build_messages(record["input"])
        self.assertNotIn("LABEL_MUST_NOT_LEAK", str(messages))

    def test_prepared_dataset_counts_and_split_separation(self):
        train = load_records(TRAIN_PATH, expected_split="train")
        validation = load_records(VALIDATION_PATH, expected_split="validation")
        evaluation = load_cases(EVALUATION_PATH)
        self.assertEqual((len(train), len(validation), len(evaluation)), (1280, 176, 30))
        validate_split_separation(train, validation)
        self.assertEqual({record["target"]["primary_category"] for record in train}, set(FILTER_CATEGORIES))
        self.assertTrue(all("category" not in record["input"] for record in evaluation))
        evaluate.validate_evaluation_cases(evaluation)

    def test_reviewed_labels_apply_to_all_splits(self):
        for path in (TRAIN_PATH, VALIDATION_PATH):
            records = load_cases(path)
            deadlines = [r for r in records if r["metadata"]["scenario"] == "deadline_extended"]
            self.assertTrue(deadlines)
            self.assertTrue(all(r["target"]["primary_category"] == "일반 업무" for r in deadlines))
        cases = {r["case_id"]: r for r in load_cases(EVALUATION_PATH)}
        for name, category in (("assignment_deadline", "일반 업무"),
                               ("orientation_venue_corrected", "일정/회의"),
                               ("maintenance_schedule_corrected", "시스템/보안"),
                               ("refund_completed", "개인 중요")):
            self.assertEqual(cases[name]["expected_category"], category)
        evaluate.validate_evaluation_cases([cases["password_reset_completed"]])

    def test_short_requests_are_one_line_and_split_disjoint(self):
        train, validation = short_request_records("train"), short_request_records("validation")
        validate_split_separation(train, validation)
        self.assertTrue(all(len(r["target"]["summary_lines"]) == 1 and r["max_summary_lines"] == 3
                            for r in train + validation))
        for r in train + validation:
            self.assertNotIn("자료 모임방", " ".join(r["target"]["summary_lines"]))

    def test_short_benchmark_is_held_out_and_references_pass(self):
        cases = short_evaluation_records()
        evaluate.validate_evaluation_cases(cases)
        self.assertEqual(len(cases), 8)
        evaluation_as_train = [
            {**r, "target": {"primary_category": r["expected_category"],
                              "summary_lines": r["reference_summary_lines"]}} for r in cases
        ]
        validate_split_separation(short_request_records("train"), evaluation_as_train)
        validate_split_separation(short_request_records("validation"), evaluation_as_train)

    def test_short_benchmark_scores_concision_without_prompt_leak(self):
        case = short_evaluation_records()[0]
        self.assertEqual(case["max_summary_lines"], 3)
        for extra, expected in (([], True), (["확인 부탁드립니다."], False)):
            raw = json.dumps({"primary_category": case["expected_category"],
                              "summary_lines": case["reference_summary_lines"] + extra}, ensure_ascii=False)
            output = io.StringIO()
            with patch("sys.argv", ["evaluate", "--task", "briefing", "--greedy"]), \
                 patch.object(evaluate, "load_cases", return_value=[case]), \
                 patch.object(evaluate, "load_model", return_value=(None, None)), \
                 patch.object(evaluate, "generate_summary", return_value=(raw, 0.1)), \
                 redirect_stdout(output):
                evaluate.main()
            result = json.loads(output.getvalue())["results"][0]
            self.assertEqual(result["passed"], expected)

    def test_offline_fallback_matches_reviewed_category_policy(self):
        for body, expected in (("보고서 제출 마감은 내일입니다.", "일반 업무"),
                               ("엘리베이터 점검 일정은 내일입니다.", "시스템/보안"),
                               ("치과 예약 일정은 내일입니다.", "개인 중요"),
                               ("신규 버전 배포에 실패했습니다.", "긴급 업무")):
            self.assertEqual(build([notification("a", body=body)]).groups[0].primary_category, expected)

    def test_room_subject_not_repeated_in_migrated_body(self):
        source = load_cases()[20]
        migrated = migrate_record(source, evaluation=True)
        self.assertTrue(all("캡스톤 팀 채팅" not in n["body"] for n in migrated["input"]["notifications"]))

    def test_schedule_what_is_concise_not_attendee_sentence(self):
        report = build([
            notification("a", body="참석자는 개발팀입니다. 회의 내일 할게요."),
            notification("b", body="오후 1시부터 2시까지요.", minute=1),
            notification("c", body="장소는 창의관 402호입니다.", minute=2),
        ])
        value = report.groups[0].schedule_summaries[0].to_dict()
        self.assertEqual(value["what"], "개발팀 회의")
        self.assertEqual(value["who"], "개발팀")
        self.assertEqual(value["end_at"], "2026-10-09T05:00:00Z")

    def test_short_request_prompt_forbids_room_and_timestamp_invention(self):
        prompt = build_messages(short_request_records("train")[0]["input"])[0]["content"]
        self.assertIn("줄 수는 상한이지 목표", prompt)
        self.assertIn("방 이름을 장소", prompt)
        self.assertIn("timestamp는 수신 시각", prompt)

    def test_old_checkpoint_cannot_resume_new_task_before_model_load(self):
        with patch("sys.argv", ["train_lora", "--task", "briefing", "--resume-from-checkpoint", "old-v6-checkpoint"]), \
             patch.object(train_lora, "train") as training, redirect_stdout(io.StringIO()), \
             self.assertRaisesRegex(ValueError, "cannot resume a v5/v6"):
            train_lora.main()
        training.assert_not_called()

    def test_evaluation_scores_category_and_facts_separately(self):
        case = migrate_record(load_cases()[0], evaluation=True)
        correct = json.dumps({"primary_category": case["expected_category"], "summary_lines": case["reference_summary_lines"]}, ensure_ascii=False)
        wrong = json.dumps({"primary_category": "기타", "summary_lines": case["reference_summary_lines"]}, ensure_ascii=False)
        for raw, accuracy in ((correct, 1.0), (wrong, 0.0)):
            output = io.StringIO()
            with patch("sys.argv", ["evaluate", "--task", "briefing", "--greedy"]), \
                 patch.object(evaluate, "load_cases", return_value=[case]), \
                 patch.object(evaluate, "load_model", return_value=(None, None)), \
                 patch.object(evaluate, "generate_summary", return_value=(raw, 0.1)), \
                 redirect_stdout(output):
                evaluate.main()
            result = json.loads(output.getvalue())
            self.assertEqual(result["category_accuracy"], accuracy)
            self.assertEqual(result["fact_coverage"], 1.0)
            self.assertEqual(result["case_pass_rate"], accuracy)


if __name__ == "__main__":
    unittest.main()
