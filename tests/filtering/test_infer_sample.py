import io
import json
import unittest
from unittest.mock import patch

from filtering_training.infer_sample import parse_input, run_stream


class InferenceInputTests(unittest.TestCase):
    def test_pascal_case_client_records_are_parsed_without_gold_label(self):
        payload = {
            "notification": {"Id": "client-1", "AppName": "TestApp", "Sender": None,
                             "Title": "Test alert", "Body": "Check status",
                             "Timestamp": "2026-09-30T05:00:00Z"},
            "context": {"ActiveProcess": "Code.exe", "WindowTitle": "test.py - editor",
                        "LastUpdated": "2026-09-30T05:00:00Z", "DurationSeconds": 45,
                        "RecentProcesses": ["Code.exe", "chrome.exe", "KakaoTalk.exe"]},
        }
        notification, context = parse_input(payload)
        self.assertEqual(notification.id, "client-1")
        self.assertEqual(notification.sender, "")
        self.assertEqual(context.duration_seconds, 45)
        self.assertEqual(len(context.recent_processes), 3)

    def test_stream_returns_one_json_result_per_input_and_survives_bad_input(self):
        payload = {"notification": {"id": "case-1", "app_name": "TestApp", "sender": "",
                                    "title": "Check", "body": "Status",
                                    "timestamp": "2026-09-30T05:00:00Z"},
                   "context": {"active_process": "", "window_title": "",
                               "last_updated": "2026-09-30T05:00:00Z"}}
        incoming = io.StringIO(json.dumps(payload) + "\n{bad json}\n")
        outgoing = io.StringIO()
        with patch('sys.stdin', incoming), patch('sys.stdout', outgoing), patch(
                'filtering_training.infer_sample.predict', return_value={'notification_id': 'case-1', 'is_passed': False}) as model:
            run_stream(object(), object())
        results = [json.loads(line) for line in outgoing.getvalue().splitlines()]
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['notification_id'], 'case-1')
        self.assertEqual(results[1], {'error': 'invalid_input_or_model_output'})
        self.assertEqual(model.call_count, 1)
