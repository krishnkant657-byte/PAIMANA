"""Tests for the hybrid agent: routing, context, and reference resolution.

Routing and reference resolution are tested without a database or a language
model, because both are deterministic by design. That is the point of the
architecture: the decision about *which* project an answer is about is made by
the platform, not guessed by a model, so it can be asserted on directly.
"""
from __future__ import annotations

import pytest

from app.services import agent
from app.services import conversation as convo


def ctx(**overrides) -> dict:
    base = convo.new_context()
    base.update(overrides)
    return base


LISTED = [
    {"project_code": "100532", "name": "New Siliguri Metro Corridor"},
    {"project_code": "100644", "name": "Augmentation of Jhansi Water Supply Scheme"},
    {"project_code": "100777", "name": "Four Laning of Hubli Highway Section"},
]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
class TestRouting:
    @pytest.mark.parametrize("message", [
        "hi", "hello", "hey there", "good morning", "thanks", "thank you",
        "ok", "how are you?", "bye",
    ])
    def test_small_talk(self, message):
        route, _ = agent.classify(message, ctx(), False)
        assert route == agent.ROUTE_SMALLTALK

    @pytest.mark.parametrize("message", [
        "what can you do?", "who are you?", "how do you work?",
    ])
    def test_capability_questions(self, message):
        route, _ = agent.classify(message, ctx(), False)
        assert route == agent.ROUTE_SMALLTALK

    @pytest.mark.parametrize("message", [
        "which projects are high risk?",
        "which projects are delayed?",
        "show me projects in Bihar",
        "which sector has the most cost overruns?",
        "what is the national situation?",
        "show open early warnings",
        "which interventions are pending?",
        "bhai konsi project high risk pe hai",
        "railway wale late projects dikha bro",
        "pehla wala kyu high risk hai",
        "sabse bekaar project konsa hai",
        "कौन से प्रोजेक्ट्स हाई रिस्क में हैं?",
    ])
    def test_paimana_questions(self, message):
        route, _ = agent.classify(message, ctx(), False)
        assert route == agent.ROUTE_PAIMANA

    @pytest.mark.parametrize("message", [
        "what is Python?", "what is GDP?", "what is machine learning?",
        "what is the capital of France?", "write a birthday message",
        "draft an email to my team",
    ])
    def test_general_questions_are_allowed_not_blocked(self, message):
        """A reasonable general question must never be treated as off-topic."""
        route, _ = agent.classify(message, ctx(), False)
        assert route == agent.ROUTE_GENERAL
        assert route != agent.ROUTE_OFF_TOPIC

    @pytest.mark.parametrize("message", [
        "give me a recipe for butter chicken",
        "tell me my horoscope",
        "write me a novel about pirates",
        "build me a multiplayer game",
        "give me dating advice",
    ])
    def test_far_off_topic(self, message):
        route, _ = agent.classify(message, ctx(), False)
        assert route == agent.ROUTE_OFF_TOPIC

    def test_attachment_routes_to_file_analysis(self):
        route, _ = agent.classify("what is this about?", ctx(), True)
        assert route == agent.ROUTE_FILE

    def test_attachment_plus_comparison_routes_to_cross_check(self):
        route, _ = agent.classify(
            "compare this with PAIMANA project data", ctx(), True
        )
        assert route == agent.ROUTE_CROSS

    def test_follow_up_with_context_stays_on_paimana(self):
        route, _ = agent.classify("why is it risky?", ctx(recent_projects=LISTED), False)
        assert route == agent.ROUTE_PAIMANA

    def test_off_topic_word_inside_a_project_question_is_not_off_topic(self):
        route, _ = agent.classify(
            "is there a recipe for reducing project cost overruns?", ctx(), False
        )
        assert route == agent.ROUTE_PAIMANA


# ---------------------------------------------------------------------------
# Off-topic escalation
# ---------------------------------------------------------------------------
class TestOffTopicEscalation:
    def test_three_levels_escalate(self):
        context = ctx()
        first = agent._off_topic("recipe", context)["answer"]
        second = agent._off_topic("horoscope", context)["answer"]
        third = agent._off_topic("novel", context)["answer"]
        assert context["off_topic_strikes"] == 3
        assert first != second != third

    def test_first_warning_is_gentle_not_a_refusal(self):
        answer = agent._off_topic("recipe", ctx())["answer"].lower()
        assert "happy to keep answering general questions" in answer
        for aggressive in ("error", "forbidden", "not allowed", "cannot"):
            assert aggressive not in answer

    def test_no_stack_trace_or_error_code_in_any_level(self):
        context = ctx()
        for _ in range(4):
            answer = agent._off_topic("x", context)["answer"]
            assert "Traceback" not in answer and "500" not in answer


# ---------------------------------------------------------------------------
# Reference resolution
# ---------------------------------------------------------------------------
class TestReferenceResolution:
    def test_ordinal_resolves_against_the_last_list(self):
        result = convo.resolve_reference(ctx(recent_projects=LISTED), "why is the first one risky?")
        assert result["resolved"] is True
        assert result["projects"][0]["project_code"] == "100532"

    def test_second_ordinal(self):
        result = convo.resolve_reference(ctx(recent_projects=LISTED), "tell me about the second one")
        assert result["projects"][0]["project_code"] == "100644"

    def test_pronoun_resolves_to_the_project_in_focus(self):
        context = ctx(current_project=LISTED[0], recent_projects=LISTED)
        result = convo.resolve_reference(context, "how delayed is it?")
        assert result["resolved"] is True
        assert result["projects"][0]["project_code"] == "100532"

    def test_out_of_range_ordinal_asks_rather_than_guessing(self):
        result = convo.resolve_reference(ctx(recent_projects=LISTED[:2]), "what about the fifth one?")
        assert result["resolved"] is False
        assert result["ambiguous"] is True
        assert result["projects"] == []

    def test_ambiguous_pronoun_asks_which(self):
        """With several projects listed and no focus, "it" must not be guessed."""
        result = convo.resolve_reference(ctx(recent_projects=LISTED), "how delayed is it?")
        assert result["resolved"] is False
        assert result["ambiguous"] is True
        assert result["options"]

    def test_no_prior_context_says_so(self):
        result = convo.resolve_reference(ctx(), "why is it risky?")
        assert result["resolved"] is False
        assert result["ambiguous"] is False

    def test_message_without_a_reference_is_left_alone(self):
        result = convo.resolve_reference(ctx(recent_projects=LISTED), "which projects are delayed?")
        assert result["resolved"] is False
        assert result["projects"] == []

    def test_rewrite_substitutes_the_resolved_name(self):
        rewritten = convo.rewrite_with_reference("why is the first one risky?", [LISTED[0]])
        assert "New Siliguri Metro Corridor" in rewritten
        assert "first one" not in rewritten


class TestContextMaintenance:
    def test_a_list_result_records_the_ordering(self):
        context = ctx()
        convo.update_from_result(context, {"intent": "HIGH_RISK_LIST", "projects": LISTED})
        assert [p["project_code"] for p in context["recent_projects"]] == \
            ["100532", "100644", "100777"]

    def test_single_project_answer_preserves_the_ordinal_list(self):
        """Regression: a follow-up used to wipe the list it was answering about."""
        context = ctx()
        convo.update_from_result(context, {"intent": "HIGH_RISK_LIST", "projects": LISTED})
        convo.update_from_result(context, {"intent": "WHY_RISK", "project": LISTED[0]})
        assert len(context["recent_projects"]) == 3
        assert context["current_project"]["project_code"] == "100532"

    def test_a_new_list_supersedes_the_old_ordering(self):
        context = ctx()
        convo.update_from_result(context, {"intent": "HIGH_RISK_LIST", "projects": LISTED})
        fresh = [{"project_code": "999", "name": "Other"}]
        convo.update_from_result(context, {"intent": "DELAYED_PROJECTS", "projects": fresh})
        assert [p["project_code"] for p in context["recent_projects"]] == ["999"]

    def test_filters_are_remembered(self):
        context = ctx()
        convo.update_from_result(context, {"intent": "STATE_PROJECTS", "state": "Bihar",
                                           "period": "2026-07", "projects": LISTED})
        assert context["state"] == "Bihar"
        assert context["period"] == "2026-07"

    def test_compact_summary_is_short_not_a_transcript(self):
        context = ctx(current_project=LISTED[0], recent_projects=LISTED,
                      state="Bihar", period="2026-07")
        summary = convo.compact_summary(context, [])
        assert len(summary) < 700
        assert "New Siliguri Metro Corridor" in summary
        assert "Bihar" in summary


class TestGroundingLabels:
    def test_every_source_has_a_grounding_note(self):
        for source in (agent.SRC_PAIMANA, agent.SRC_GENERAL, agent.SRC_FILE,
                       agent.SRC_MIXED, agent.SRC_DERIVED, agent.SRC_SYSTEM):
            assert agent._grounding_note(source)

    def test_general_answers_disclaim_paimana_sourcing(self):
        assert "not based on PAIMANA" in agent._grounding_note(agent.SRC_GENERAL)

    def test_file_answers_disclaim_paimana_sourcing(self):
        assert "not from PAIMANA" in agent._grounding_note(agent.SRC_FILE)

    def test_no_llm_general_answer_is_honest_not_fabricated(self):
        from app.config import settings

        if settings.llm_enabled:
            pytest.skip("A language model is configured on this deployment.")
        result = agent._general_answer("what is Python?", "")
        assert result["resolved"] is True
        assert "Python" in result["answer"]
