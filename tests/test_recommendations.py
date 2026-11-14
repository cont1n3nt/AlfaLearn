import unittest
from dataclasses import asdict

from alfa_ai import AIRequest, LearningProfile, recommend_materials
from alfa_ai.reviewed import LOYALTY


class RecommendationTests(unittest.TestCase):
    def test_budget_filter_removes_subscription_and_paid_cards(self):
        for query, access in (
            ("кассовый разрыв", "subscription"),
            ("УСН на маркетплейсах", "paid"),
        ):
            with self.subTest(query=query):
                self.assertEqual(recommend_materials(AIRequest(query))[0].access, access)
                self.assertEqual(
                    recommend_materials(AIRequest(query, LearningProfile(free_only=True))), ()
                )

    def test_source_is_not_repeated_in_next_step(self):
        request = AIRequest("Как начислять бонусы в салоне?")
        self.assertEqual(recommend_materials(request)[0].material_id, LOYALTY)
        self.assertEqual(recommend_materials(request, excluded_ids={LOYALTY}), ())

    def test_format_preference_is_used_only_after_topic_match(self):
        profile = LearningProfile(preferred_format="podcast")
        cards = recommend_materials(AIRequest("Отзывы и бонусы в салоне", profile))
        self.assertEqual(cards[0].kind, "podcast")
        self.assertEqual(
            recommend_materials(AIRequest("Нужен принтер этикеток", profile))[0].kind, "article"
        )

    def test_paid_output_exposes_only_public_card_fields(self):
        [card] = recommend_materials(AIRequest("Какой курс есть про УСН на маркетплейсах?"))
        self.assertEqual(
            set(asdict(card)), {"material_id", "title", "url", "kind", "access", "reason"}
        )
        self.assertEqual(card.access, "paid")
        self.assertTrue(card.url.startswith("https://kurs.alfabank.ru/courses/"))

    def test_no_subscription_allows_a_one_time_purchase(self):
        self.assertEqual(recommend_materials(AIRequest("Кассовый разрыв, без подписки")), ())
        cards = recommend_materials(AIRequest("Курс по УСН на маркетплейсах без подписки"))
        self.assertEqual(cards[0].access, "paid")


if __name__ == "__main__":
    unittest.main()
