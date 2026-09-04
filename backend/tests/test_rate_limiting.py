"""Rate limiting on the conversational endpoints.

These are the only tests that intentionally exhaust a bucket. Every other test
gets a cleared limiter from conftest, so this behaviour is asserted in exactly
one place rather than accidentally everywhere.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.security import _BUCKETS

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear():
    _BUCKETS.clear()
    yield
    _BUCKETS.clear()


def test_message_endpoint_is_rate_limited():
    limit = settings.chat_rate_limit
    last = None
    for _ in range(limit + 3):
        last = client.post("/api/chat/message", json={"message": "hello"})
        if last.status_code == 429:
            break
    assert last.status_code == 429
    assert "Retry-After" in last.headers
    assert "Try again" in last.json()["detail"]


def test_upload_endpoint_is_rate_limited():
    limit = settings.upload_rate_limit
    last = None
    for i in range(limit + 3):
        last = client.post(
            "/api/chat/attachments",
            files={"file": (f"n{i}.txt", b"some text content")},
        )
        if last.status_code == 429:
            break
    assert last.status_code == 429


def test_rate_limit_response_is_not_a_stack_trace():
    for _ in range(settings.chat_rate_limit + 3):
        response = client.post("/api/chat/message", json={"message": "hello"})
        if response.status_code == 429:
            assert "Traceback" not in response.text
            assert "/home/" not in response.text
            return
    pytest.fail("the rate limit was never reached")
