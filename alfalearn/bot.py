from aiogram import Dispatcher, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from alfalearn.accounts.bot import create_accounts_router
from alfalearn.accounts.service import AccountService
from alfalearn.grouping import AnswerBuffer
from alfalearn.llm import OllamaClient
from alfalearn.memory import ConversationStore

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


def help_message(*, accounts: bool) -> str:
    """Answer to /help: available commands except /start and /hello."""
    return (
        "Доступные команды:\n"
        + (ACCOUNT_COMMANDS if accounts else "")
        + "/reset — очистить историю диалога\n"
        "/help — показать этот список"
    )


def create_dispatcher(
    group_delay_seconds: float = 1.0,
    *, llm: OllamaClient | None = None, classification_timeout_seconds: float = 0.8,
    memory: ConversationStore | None = None, history_context_chars: int = 12_000,
    accounts: AccountService | None = None,
) -> Dispatcher:
    router = Router(name="answers")
    buffer = AnswerBuffer(
        group_delay_seconds, llm=llm,
        classification_timeout_seconds=classification_timeout_seconds,
        memory=memory, history_context_chars=history_context_chars,
    )
    greeting = greeting_message(llm=llm is not None, accounts=accounts is not None)
    help_text = help_message(accounts=accounts is not None)

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
    dispatcher.include_router(router)
    return dispatcher
