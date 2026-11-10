# AlfaLearn — АльфаНавигатор

Минимальный Telegram-бот для кейса «Альфа-Курс»: будущий помощник по обучающим
материалам для предпринимателей. Сейчас реализована только проверка связи
Hello World; поиск, LLM, профиль и напоминания пока не подключены.

Бот: [@AlfaLearnBot](https://t.me/AlfaLearnBot).
Отображаемое имя: **АльфаНавигатор**.
Первый проверенный username `AlfaLearnBot` оказался свободен. Другие варианты
для названия: «КурсНавигатор», «АльфаМаршрут».

## Запуск на Windows

Нужен Python 3.11+ и доступ к Telegram API. Из корня проекта:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Укажи токен BotFather в `BOT_TOKEN` в `.env`. В текущем рабочем проекте `.env`
уже заполнен — не перезаписывай его примером.

```powershell
.\.venv\Scripts\python.exe -m alfalearn
```

Один процесс запускает FastAPI и Telegram long polling. Публичный домен,
HTTPS-туннель и webhook для локального запуска не нужны. Запускай только один
экземпляр бота и один worker. Остановка: `Ctrl+C`.

## Что проверить

- `GET http://127.0.0.1:8000/hello` → `{"message":"Hello, world!"}`.
- `GET http://127.0.0.1:8000/` → тот же ответ.
- `GET http://127.0.0.1:8000/health` → состояние процесса polling и username.
- [Swagger UI](http://127.0.0.1:8000/docs) — документация HTTP API.
- В Telegram: `/start` — приветствие, `/hello`, `/help` или обычный текст — `Hello, world!`.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/hello
Invoke-RestMethod http://127.0.0.1:8000/health
```

## Настройки

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `BOT_TOKEN` | — | Секретный токен из BotFather |
| `BOT_ENABLED` | `true` | `false` запускает только HTTP API без Telegram |
| `HOST` | `127.0.0.1` | Адрес HTTP-сервера |
| `PORT` | `8000` | Порт HTTP-сервера |

Переменные окружения имеют приоритет над `.env`. Сервер проверяет токен и
отсутствие webhook перед запуском. Если webhook уже настроен, запуск завершится
с ошибкой: конфигурацию Telegram нужно сначала переключить на polling.
Если задача polling остановится после запуска, `/health` вернёт HTTP 503.

`.env`, виртуальное окружение, настройки IDE, логи и локальные выгрузки базы
исключены из Git. В репозитории хранится только `.env.example` без ключа.

## Структура и проверки

```text
alfalearn/
  __main__.py   # python -m alfalearn
  config.py     # настройки из .env
  bot.py        # Telegram handlers
  api.py        # HTTP endpoints и lifecycle polling
scripts/
  configure_bot.py  # имя, описание и меню команд в Telegram
tests/
  test_app.py   # HTTP API, отсутствие токена, маршрутизация Telegram updates
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Тесты не обращаются в Telegram. Для повторной настройки имени, публичного описания и
команд бота (токен берётся из `.env`):

```powershell
.\.venv\Scripts\python.exe -m scripts.configure_bot
```

Основные зависимости зафиксированы в `requirements.txt`; полный снимок
проверенного окружения, включая тестовые зависимости, — `requirements.lock`.
Для его воспроизведения: `python -m pip install -r requirements.lock`.

Документация: [aiogram polling](https://docs.aiogram.dev/en/latest/dispatcher/long_polling.html),
[FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/).
