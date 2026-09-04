"""Shared test fixtures.

The rate limiter in `app.security` is a module-level, in-process dictionary. It
is shared by every test that touches an endpoint, so without isolation the
counter accumulates across modules and unrelated tests start failing with 429s
purely because of how many tests ran before them.

Clearing it per test keeps each test asserting the thing it is actually about.
Rate limiting itself is covered deliberately in `test_rate_limiting.py`.
"""
from __future__ import annotations

import pytest

from app.db import init_db
from app.security import _BUCKETS


@pytest.fixture(scope="session", autouse=True)
def _schema():
    """Ensure the schema exists before any test queries the database."""
    init_db()


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    _BUCKETS.clear()
    yield
    _BUCKETS.clear()
