"""Map consultation responses to text and links for the bot."""

from alfa_ai import AIResponse

TOO_LONG = "Сократите вопрос до 1000 символов."


def format_answer(response: AIResponse) -> str:
    sections = [response.text]
    if response.proposed_action is not None:
        sections = ["Настройка и доставка напоминаний пока недоступны в боте."]
    if response.sources:
        sources = []
        for source in response.sources:
            interval = ""
            if source.start_sec is not None and source.end_sec is not None:
                interval = f" ({format_timestamp(source.start_sec)}–{format_timestamp(source.end_sec)})"
            sources.append(f"[{source.source_id}] {source.title}{interval}\n{source.url}")
        sections.append("Источники\n" + "\n".join(sources))
    if response.recommendations:
        access_labels = {"free": "бесплатно", "paid": "платно", "subscription": "по подписке"}
        cards = [
            f"{card.title} ({access_labels[card.access]})\n{card.reason}\n{card.url}"
            for card in response.recommendations
        ]
        sections.append("Материалы\n" + "\n\n".join(cards))
    return "\n\n".join(sections)


def format_timestamp(seconds: float) -> str:
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes:02d}:{remainder:02d}"
