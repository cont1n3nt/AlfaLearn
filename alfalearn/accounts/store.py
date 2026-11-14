"""SQLite storage for demo accounts. The only module with SQL in it.

Written in the same style as `alfalearn.memory.ConversationStore`: plain `sqlite3`,
one connection guarded by a lock, short transactions. Table names do not clash with
the conversation history, so both stores may share one database file.
"""

import sqlite3
from collections.abc import Iterable
from pathlib import Path
from threading import RLock

from alfalearn.accounts.demo_data import Course, DemoUser, Product
from alfalearn.accounts.passwords import hash_password

SCHEMA = """
CREATE TABLE IF NOT EXISTS courses (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    description TEXT NOT NULL,
    lessons INTEGER NOT NULL,
    duration TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS products (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK(kind IN ('subscription', 'course')),
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    price_rub INTEGER NOT NULL,
    price_note TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS product_courses (
    product_id TEXT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    course_id TEXT NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    PRIMARY KEY (product_id, course_id)
);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    login TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    name TEXT NOT NULL,
    access_note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS user_products (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    product_id TEXT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    since TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, product_id)
);
CREATE TABLE IF NOT EXISTS telegram_links (
    telegram_id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    linked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class AccountStore:
    def __init__(self, path: str | Path = ":memory:"):
        database = str(path)
        if database != ":memory:":
            database_path = Path(path).expanduser()
            database_path.parent.mkdir(parents=True, exist_ok=True)
            database = str(database_path)
        self._lock = RLock()
        self._connection: sqlite3.Connection | None = sqlite3.connect(
            database, timeout=5, check_same_thread=False,
        )
        try:
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA busy_timeout = 5000")
            self._connection.execute("PRAGMA foreign_keys = ON")
            if database != ":memory:":
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.executescript(SCHEMA)
        except BaseException:
            self._connection.close()
            self._connection = None
            raise

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("Account store is closed")
        return self._connection

    # --- Demo data -------------------------------------------------------------

    def seed(
        self, courses: Iterable[Course], products: Iterable[Product], users: Iterable[DemoUser],
    ) -> list[str]:
        """Refresh the catalog and add missing demo users. Returns logins of new users.

        Existing users, their passwords and Telegram links are left untouched.
        """
        created = []
        with self._lock:
            connection = self._db()
            with connection:
                for course in courses:
                    connection.execute(
                        """INSERT INTO courses (id, title, url, description, lessons, duration)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ON CONFLICT(id) DO UPDATE SET title = excluded.title, url = excluded.url,
                               description = excluded.description, lessons = excluded.lessons,
                               duration = excluded.duration""",
                        (course.id, course.title, course.url, course.description,
                         course.lessons, course.duration),
                    )
                for product in products:
                    connection.execute(
                        """INSERT INTO products (id, kind, title, url, price_rub, price_note)
                           VALUES (?, ?, ?, ?, ?, ?)
                           ON CONFLICT(id) DO UPDATE SET kind = excluded.kind, title = excluded.title,
                               url = excluded.url, price_rub = excluded.price_rub,
                               price_note = excluded.price_note""",
                        (product.id, product.kind, product.title, product.url,
                         product.price_rub, product.price_note),
                    )
                    connection.execute("DELETE FROM product_courses WHERE product_id = ?", (product.id,))
                    connection.executemany(
                        "INSERT INTO product_courses (product_id, course_id) VALUES (?, ?)",
                        [(product.id, course_id) for course_id in product.course_ids],
                    )
                for user in users:
                    cursor = connection.execute(
                        """INSERT INTO users (login, password_hash, name, access_note)
                           VALUES (?, ?, ?, ?) ON CONFLICT(login) DO NOTHING""",
                        (user.login, hash_password(user.password), user.name, user.access_note),
                    )
                    if not cursor.rowcount:
                        continue
                    created.append(user.login)
                    connection.executemany(
                        "INSERT INTO user_products (user_id, product_id) VALUES (?, ?)",
                        [(cursor.lastrowid, product_id) for product_id in user.product_ids],
                    )
        return created

    def delete_users(self, logins: Iterable[str]) -> int:
        """Delete users with their products and Telegram links."""
        with self._lock:
            connection = self._db()
            with connection:
                return connection.executemany(
                    "DELETE FROM users WHERE login = ?", [(login,) for login in logins],
                ).rowcount

    # --- Users and Telegram links ----------------------------------------------

    def user_by_login(self, login: str) -> sqlite3.Row | None:
        with self._lock:
            return self._db().execute(
                "SELECT id, login, password_hash, name, access_note FROM users WHERE login = ?",
                (login,),
            ).fetchone()

    def user_by_telegram(self, telegram_id: int) -> sqlite3.Row | None:
        with self._lock:
            return self._db().execute(
                """SELECT users.id, users.login, users.name, users.access_note FROM telegram_links
                   JOIN users ON users.id = telegram_links.user_id
                   WHERE telegram_links.telegram_id = ?""",
                (telegram_id,),
            ).fetchone()

    def link_telegram(self, telegram_id: int, user_id: int) -> None:
        with self._lock:
            connection = self._db()
            with connection:
                connection.execute(
                    """INSERT INTO telegram_links (telegram_id, user_id) VALUES (?, ?)
                       ON CONFLICT(telegram_id) DO UPDATE SET user_id = excluded.user_id,
                           linked_at = CURRENT_TIMESTAMP""",
                    (telegram_id, user_id),
                )

    def unlink_telegram(self, telegram_id: int) -> bool:
        with self._lock:
            connection = self._db()
            with connection:
                cursor = connection.execute(
                    "DELETE FROM telegram_links WHERE telegram_id = ?", (telegram_id,),
                )
                return bool(cursor.rowcount)

    # --- Access --------------------------------------------------------------

    def products(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._db().execute(
                "SELECT id, kind, title, url, price_rub, price_note FROM products ORDER BY rowid",
            ).fetchall()

    def user_product_ids(self, user_id: int) -> set[str]:
        with self._lock:
            rows = self._db().execute(
                "SELECT product_id FROM user_products WHERE user_id = ?", (user_id,),
            ).fetchall()
        return {row["product_id"] for row in rows}

    def paid_courses(self) -> list[sqlite3.Row]:
        """Every paid course with the product that opens it."""
        with self._lock:
            return self._db().execute(
                """SELECT courses.id, courses.title, courses.url, courses.description,
                          courses.lessons, courses.duration, product_courses.product_id
                   FROM courses JOIN product_courses ON product_courses.course_id = courses.id
                   JOIN products ON products.id = product_courses.product_id
                   ORDER BY products.rowid, courses.rowid""",
            ).fetchall()

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
