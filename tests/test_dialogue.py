import unittest

from alfa_ai import AIRequest, AIResponse, LearningProfile, recommend_materials
from alfa_ai.dialogue import resolve_dialogue


class DialogueTests(unittest.TestCase):
    def test_owner_declaration_returns_patch_without_guessing_task(self):
        _, patch, clarification = resolve_dialogue(AIRequest("У меня салон красоты."))
        self.assertEqual(patch.industry, "salon")
        self.assertIsNone(patch.stage)
        self.assertIn("задачу", clarification)
        response = AIResponse("clarify", clarification, "test", profile_patch=patch)
        self.assertEqual(response.to_dict()["profile_patch"], {"industry": "salon"})

    def test_follow_up_keeps_previous_topic_and_user_constraints(self):
        request = AIRequest(
            "А для моего салона?",
            LearningProfile(free_only=True),
            previous_query="Как запустить программу лояльности?",
        )
        effective, patch, clarification = resolve_dialogue(request)
        self.assertIsNone(clarification)
        self.assertIn("программу лояльности", effective.query)
        self.assertIn("салон красоты", effective.query)
        self.assertTrue(effective.profile.free_only)
        self.assertEqual(patch.industry, "salon")
        self.assertEqual(recommend_materials(effective)[0].kind, "article")

    def test_missing_previous_query_requires_clarification(self):
        for previous in (None, "А для меня?"):
            with self.subTest(previous=previous):
                _, _, clarification = resolve_dialogue(
                    AIRequest("А для моего салона?", previous_query=previous)
                )
                self.assertIsNotNone(clarification)

    def test_third_person_and_one_shot_budget_do_not_change_profile(self):
        _, patch, _ = resolve_dialogue(AIRequest("У знакомого салон. Найди бесплатную статью."))
        self.assertIsNone(patch)
        _, patch, _ = resolve_dialogue(AIRequest("Запускаю программу лояльности. У меня салон."))
        self.assertIsNone(patch.stage)

    def test_independent_requests_do_not_share_profile(self):
        resolve_dialogue(AIRequest("Открываю свою кофейню."))
        effective, patch, _ = resolve_dialogue(AIRequest("Как выбрать принтер этикеток?"))
        self.assertIsNone(patch)
        self.assertIsNone(effective.profile.industry)

    def test_business_declaration_updates_profile_and_asks_for_task(self):
        _, patch, clarification = resolve_dialogue(AIRequest("Управляю своей кофейней."))
        self.assertEqual((patch.industry, patch.stage), ("cafe", "running"))
        self.assertIn("задачу", clarification)


if __name__ == "__main__":
    unittest.main()
