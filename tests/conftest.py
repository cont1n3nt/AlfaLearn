import pytest


@pytest.fixture(autouse=True)
def isolated_accounts_db(tmp_path, monkeypatch):
    """Apps created in tests keep demo users in a temporary database, not in .run/."""
    monkeypatch.setenv("ACCOUNTS_PATH", str(tmp_path / "accounts.sqlite3"))
