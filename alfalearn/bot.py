from aiogram import Dispatcher, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from alfa_ai import AIService
from alfalearn.accounts.bot import create_accounts_router
from alfalearn.analytics import EventStore
from alfalearn.accounts.service import AccountService
from alfalearn.calculator_menu import create_calculator_router
from alfalearn.grouping import AnswerBuffer
from alfalearn.llm import OllamaClient
from alfalearn.memory import ConversationStore
from alfalearn.reminder_runtime import ReminderDialogue
from alfalearn.vat_bot import create_vat_router

# HTTP GET /hello keeps the connectivity check answer.
HELLO_MESSAGE = "Hello, world!"

RESET_MESSAGE = (
    "История диалога очищена — я больше не помню, о чём мы говорили. "
    "Сообщения в чате остаются; удалить их можно через меню чата → «Очистить историю»."
)

ACCOUNT_COMMANDS = (
    "/login — войти в демо-аккаунт (anna, boris, vera)\n"
    "/profile — мой доступ к платным курсам\n"
    "/logout — выйти из аккаунта\n"
)


def greeting_message(*, llm: bool, accounts: bool) -> str:
    """Answer to /start and /hello."""
    skills = ["• найти ответ на вопрос о бизнесе, налогах и финансах;"]
    if accounts:
        skills.append("• разобраться, какие курсы Альфа-Курса вам уже доступны, а какие можно открыть;")
    skills.append("• запомнить контекст разговора, чтобы не повторять одно и то же.")
    footer = (
        "Просто напишите свой вопрос в чат 👇" if llm
        else "Ответы на вопросы пока в разработке — сейчас работают команды из /help."
    )
    return (
        "Привет! Я АльфаНавигатор — помощник по обучающим материалам Альфа-Курса "
        "для предпринимателей.\n\n"
        "Помогу:\n" + "\n".join(skills) + "\n\n" + footer
    )


def help_message(*, accounts: bool, reminders: bool = False) -> str:
    """Answer to /help: available commands except /start and /hello."""
    return (
        "Доступные команды:\n"
        + (ACCOUNT_COMMANDS if accounts else "")
        + "/calc — калькуляторы Альфа-Курса\n"
        + ("/reminders — мои активные напоминания\n" if reminders else "")
        + "/reset — очистить историю диалога\n"
        "/help — показать этот список"
    )


def create_dispatcher(
    group_delay_seconds: float = 1.0,
    *, llm: OllamaClient | None = None, classification_timeout_seconds: float = 0.8,
    memory: ConversationStore | None = None, history_context_chars: int = 12_000,
    accounts: AccountService | None = None,
    ai: AIService | None = None,
    reminders: ReminderDialogue | None = None,
    analytics: EventStore | None = None,
) -> Dispatcher:
    router = Router(name="answers")
    buffer = AnswerBuffer(
        group_delay_seconds, llm=llm,
        classification_timeout_seconds=classification_timeout_seconds,
        memory=memory, history_context_chars=history_context_chars, ai=ai,
        reminders=reminders, analytics=analytics,
    )
    greeting = greeting_message(llm=ai is not None or llm is not None, accounts=accounts is not None)
    help_text = help_message(accounts=accounts is not None, reminders=reminders is not None)

    @router.callback_query(lambda query: bool(query.data and query.data.startswith("feedback:")))
    async def feedback(query: CallbackQuery) -> None:
        _, vote, trace_id = query.data.split(":", 2)
        if vote == "clarify":
            await query.answer("Напишите, что уточнить — я попробую ответить точнее.")
            return
        if analytics is None:
            await query.answer("Спасибо за оценку!")
            return
        event = "feedback_positive" if vote == "+" else "feedback_negative"
        fresh = analytics.record_once(f"feedback:{trace_id}", event)
        if vote == "-" and fresh and query.message is not None:
            previous = query.message.reply_markup
            rows = [row for row in (previous.inline_keyboard if previous else [])
                    if not any(button.callback_data and button.callback_data.startswith("feedback:") for button in row)]
            rows.append([InlineKeyboardButton(
                text="Уточнить вопрос", callback_data=f"feedback:clarify:{trace_id}"
            )])
            await query.message.edit_reply_markup(reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))
        await query.answer("Спасибо за оценку!" if fresh else "Оценка уже учтена.")

    @router.message(CommandStart())
    @router.message(Command("hello"))
    async def start(message: Message) -> None:
        buffer.add(message, command_answer=greeting)

    @router.message(Command("help"))
    async def help_command(message: Message) -> None:
        buffer.add(message, command_answer=help_text)

    @router.message(Command("reset"))
    async def reset(message: Message) -> None:
        buffer.reset(message)
        await message.answer(f"{RESET_MESSAGE}\n\n{greeting}")

    @router.message(Command("reminders"))
    async def list_reminders(message: Message) -> None:
        if reminders is None:
            await message.answer("Напоминания сейчас недоступны.")
            return
        items = reminders.store.list_visible(buffer.conversation_key(message))
        if not items:
            await message.answer("Активных напоминаний нет.")
            return
        lines = ["Ваши напоминания:"]
        for item in items:
            if item["status"] == "failed":
                lines.append(
                    f"#{item['id']} — не доставлено после трёх попыток; "
                    "измените расписание или отключите напоминание."
                )
                continue
            cadence = {"once": "разово", "daily": "ежедневно", "weekly": "еженедельно"}
            lines.append(
                f"#{item['id']} — {item['subject']}; {item['local_at'].replace('T', ' ')} "
                f"({item['timezone']}), {cadence[item['repeat']]}"
            )
        await message.answer("\n".join(lines))

    # Register last so commands retain their dedicated replies.
    @router.message()
    async def answer(message: Message) -> None:
        buffer.add(message)

    dispatcher = Dispatcher()
    dispatcher["answer_buffer"] = buffer
    dispatcher.shutdown.register(buffer.close)
    if accounts is not None:
        # Before the catch-all: login steps and passwords must not reach the answers buffer.
        dispatcher.include_router(create_accounts_router(accounts))
    dispatcher.include_router(create_calculator_router())
    dispatcher.include_router(create_vat_router())
    dispatcher.include_router(router)
    return dispatcher
