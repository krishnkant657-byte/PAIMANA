"""Tests for the assistant's conversational scope guard."""
from app.services import assistant


def test_scope_json_parser_accepts_clean_json():
    assert assistant._extract_json('{"scope":"CASUAL"}') == {"scope": "CASUAL"}


def test_scope_json_parser_accepts_fenced_or_extra_text():
    assert assistant._extract_json('```json\n{"scope":"OFF_TOPIC"}\n```') == {"scope": "OFF_TOPIC"}


def test_scope_json_parser_rejects_unknown_scope():
    assert assistant._extract_json('{"scope":"EVERYTHING"}') is None


def test_casual_message_uses_general_chat(monkeypatch):
    monkeypatch.setattr(assistant, "plan_and_execute", lambda *args, **kwargs: {
        "intent": "UNKNOWN", "resolved": False, "reason": "not a PAIMANA query"
    })
    monkeypatch.setattr(assistant, "_classify_chat_scope", lambda q: "CASUAL")
    monkeypatch.setattr(assistant, "_call_general_chat", lambda q: "The capital of France is Paris.")

    body = assistant.answer(None, "What is the capital of France?")
    assert body["resolved"] is True
    assert body["intent"] == "CASUAL_CHAT"
    assert body["answer_source"] == "llm_general"
    assert "Paris" in body["answer"]
    assert "not sourced from PAIMANA" in body["grounding_note"]


def test_far_off_topic_message_is_redirected(monkeypatch):
    monkeypatch.setattr(assistant, "plan_and_execute", lambda *args, **kwargs: {
        "intent": "UNKNOWN", "resolved": False, "reason": "not a PAIMANA query"
    })
    monkeypatch.setattr(assistant, "_classify_chat_scope", lambda q: "OFF_TOPIC")

    body = assistant.answer(None, "Build me a complete multiplayer game unrelated to projects.")
    assert body["resolved"] is False
    assert body["intent"] == "OFF_TOPIC"
    assert body["answer_source"] == "scope_guard"
    assert "get back to projects" in body["answer"].lower()
