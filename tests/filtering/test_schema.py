import unittest

from pydantic import ValidationError

from src.filtering.schema import CurrentContext, RawNotification


class FilteringSchemaTests(unittest.TestCase):
    def test_utc_z_datetimes_are_accepted(self) -> None:
        notification = RawNotification(
            id="noti_test",
            app_name="TestApp",
            sender="가상 발신자",
            title="테스트 알림",
            body="테스트 알림 본문입니다.",
            timestamp="2026-09-21T09:00:00Z",
        )
        context = CurrentContext(
            active_process="Code.exe",
            window_title="test.py - Visual Studio Code",
            last_updated="2026-09-21T08:59:55Z",
        )

        self.assertEqual(notification.timestamp.utcoffset().total_seconds(), 0)
        self.assertEqual(context.last_updated.utcoffset().total_seconds(), 0)

    def test_null_sender_is_normalized_for_real_notifications(self) -> None:
        notification = RawNotification.model_validate({
            "id": "realistic_test",
            "app_name": "집중",
            "sender": None,
            "title": "집중 시간이 끝났습니다",
            "body": "",
            "timestamp": "2026-09-27T09:00:00Z",
        })
        self.assertEqual(notification.sender, "")
        self.assertEqual(notification.body, "")
    def test_timezone_less_datetimes_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RawNotification(
                id="noti_test",
                app_name="TestApp",
                sender="가상 발신자",
                title="테스트 알림",
                body="테스트 알림 본문입니다.",
                timestamp="2026-09-21T09:00:00",
            )

        with self.assertRaises(ValidationError):
            CurrentContext(
                active_process="Code.exe",
                window_title="test.py - Visual Studio Code",
                last_updated="2026-09-21T08:59:55",
            )

    def test_non_z_utc_offset_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RawNotification(
                id="noti_test",
                app_name="TestApp",
                sender="가상 발신자",
                title="테스트 알림",
                body="테스트 알림 본문입니다.",
                timestamp="2026-09-21T09:00:00+00:00",
            )


if __name__ == "__main__":
    unittest.main()
