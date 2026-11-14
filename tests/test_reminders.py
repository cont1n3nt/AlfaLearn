import unittest
from datetime import datetime

from alfa_ai import AIRequest
from alfa_ai.reminders import plan_reminder

NOW = datetime.fromisoformat("2026-11-12T16:00:00+03:00")


def plan(query, *, timezone="Europe/Moscow", previous_query=None, now=NOW):
    return plan_reminder(
        AIRequest(query, previous_query=previous_query, now=now, timezone=timezone)
    )[0]


class ReminderTests(unittest.TestCase):
    def test_relative_create_uses_explicit_clock_and_confirmation(self):
        result = plan("Напомни завтра в 18:00 пройти курс")
        self.assertEqual(
            (result.action, result.subject, result.repeat),
            ("create_reminder", "пройти курс", "once"),
        )
        self.assertEqual(result.local_datetime, "2026-11-13T18:00")
        self.assertEqual(result.timezone, "Europe/Moscow")
        self.assertEqual(result.missing_fields, ())
        self.assertTrue(result.needs_confirmation)

    def test_tax_request_does_not_create_an_inferred_deadline(self):
        result = plan("Напомни заплатить налог", timezone=None)
        self.assertEqual(set(result.missing_fields), {"date", "time", "timezone"})
        self.assertIsNone(result.local_datetime)

    def test_recurrence_next_occurrence_and_update(self):
        result = plan("Напоминай каждый понедельник в 09:00 учиться")
        self.assertEqual(
            (result.repeat, result.weekday, result.local_datetime),
            ("weekly", 0, "2026-11-16T09:00"),
        )
        result = plan("Напоминай ежедневно в 09:00 учиться")
        self.assertEqual(result.local_datetime, "2026-11-13T09:00")
        result = plan("Измени напоминание #abc на ежедневное")
        self.assertIn("change", result.missing_fields)
        result = plan("Измени напоминание #abc, повторяй ежедневно")
        self.assertEqual(
            (result.action, result.reminder_id, result.repeat), ("update_reminder", "abc", "daily")
        )
        self.assertFalse(result.missing_fields)

    def test_invalid_or_ambiguous_times_are_not_proposed(self):
        self.assertIn("time", plan("Напомни сегодня в 09:00 пройти курс").missing_fields)
        self.assertIn("date", plan("Напомни 2026-02-30 в 18:00 пройти курс").missing_fields)
        self.assertIn("repeat", plan("Напоминай по будням в 18:00 пройти курс").missing_fields)
        self.assertIn("time", plan("Напомни завтра до 18:00 пройти курс").missing_fields)
        self.assertIn(
            "time",
            plan(
                "Напомни 2026-11-01 в 01:30 пройти курс",
                timezone="America/New_York",
                now=datetime.fromisoformat("2026-10-30T10:00:00+00:00"),
            ).missing_fields,
        )

    def test_cancel_needs_an_exact_identifier(self):
        self.assertIn("reminder_id", plan("Отмени напоминание").missing_fields)
        result = plan("Отмени напоминание #r42")
        self.assertEqual((result.action, result.reminder_id), ("cancel_reminder", "r42"))
        self.assertEqual(result.missing_fields, ())

    def test_multiple_weekdays_require_clarification(self):
        for text in (
            "Напоминай каждый понедельник и каждую среду в 09:00 учиться",
            "Напоминай по понедельникам и средам в 09:00 учиться",
        ):
            with self.subTest(text=text):
                result = plan(text)
                self.assertIn("repeat", result.missing_fields)
                self.assertIsNone(result.repeat)
                self.assertIsNone(result.local_datetime)

    def test_short_slot_reply_uses_previous_reminder(self):
        result = plan("18:00", previous_query="Напомни завтра пройти курс")
        self.assertEqual(result.local_datetime, "2026-11-13T18:00")
        self.assertEqual(result.subject, "пройти курс")
        self.assertIsNone(
            plan_reminder(
                AIRequest("Как выбрать принтер?", previous_query="Напомни завтра пройти курс")
            )
        )


if __name__ == "__main__":
    unittest.main()
