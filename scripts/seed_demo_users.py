"""Create or reset the three demo users. Run from the project root.

    python -m scripts.seed_demo_users            # add missing users (the bot also does this on start)
    python -m scripts.seed_demo_users --reset    # recreate users, drop their Telegram logins
"""

import argparse

from alfalearn.accounts.demo_data import DEMO_USERS
from alfalearn.accounts.service import AccountService
from alfalearn.accounts.store import AccountStore
from alfalearn.config import Settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="пересоздать демо-юзеров и сбросить входы")
    args = parser.parse_args(argv)

    path = Settings().accounts_path
    store = AccountStore(path)
    try:
        service = AccountService(store)
        created = service.reset_demo() if args.reset else service.seed_demo()
    finally:
        store.close()

    print(f"База: {path}")
    print(f"Создано юзеров: {len(created)}" + (" (сброс)" if args.reset else ""))
    for user in DEMO_USERS:
        print(f"  {user.login:<6} {user.password:<16} {user.name} — {user.access_note}")


if __name__ == "__main__":
    main()
