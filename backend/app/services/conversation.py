"""Conversation state and reference resolution.

The design constraint from the existing assistant carries over: the model is not
handed a growing transcript and asked to work it out. Instead the platform keeps
a small, explicit context object — which project we are on, which projects were
last listed, the last filter, which files are attached — and resolves references
like "it", "the first one" or "that project" *before* any query runs.

That matters for grounding. If "how delayed is it?" were resolved by the model,
the model would be choosing which project the answer is about, and a wrong guess
would produce a confidently-worded answer about the wrong project. Resolving it
here means the reference either resolves to a specific project code or the
assistant asks which one was meant.
"""
from __future__ import annotations

import datetime as dt
import re

from sqlalchemy.orm import Session

from ..models import ChatMessage, Conversation, Project

MAX_RECENT_PROJECTS = 10
MAX_HISTORY_TURNS = 12

ORDINALS = {
    "first": 0, "1st": 0, "one": 0, "pehla": 0, "pahla": 0, "pehle": 0, "1": 0, "पहला": 0,
    "second": 1, "2nd": 1, "two": 1, "dusra": 1, "dosra": 1, "doosra": 1, "2": 1, "दूसरा": 1,
    "third": 2, "3rd": 2, "three": 2, "tisra": 2, "teesra": 2, "3": 2, "तीसरा": 2,
    "fourth": 3, "4th": 3, "chautha": 3, "chotha": 3, "4": 3, "चौथा": 3,
    "fifth": 4, "5th": 4, "panchwa": 4, "5": 4, "पांचवा": 4,
    "sixth": 5, "6th": 5, "seventh": 6, "7th": 6,
    "eighth": 7, "8th": 7, "ninth": 8, "9th": 8, "tenth": 9, "10th": 9,
}

#: Words that refer back to something already mentioned rather than naming it.
ANAPHORA = re.compile(
    r"\b(it|its|it's|this|that|these|those|them|they|their|the same|"
    r"the project|this project|that project|the one|yeh|ye|woh|wo|iska|uska|ispe|uspe|"
    r"iswala|uswala|is\s+project|us\s+project|bhai|bro|dude|yaha|"
    r"(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th|pehla|pahla|dusra|dosra|doosra|tisra|teesra|chautha|panchwa|"
    r"पहला|दूसरा|तीसरा)(?:\s+(?:one|project|item|wala|wali))?|"
    r"the last|aakhri|akheri|aakhri wala|the latter|the former|above|previous)\b",
    re.IGNORECASE,
)

ORDINAL_REF = re.compile(
    r"\b(?:the\s+)?(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th|pehla|pahla|dusra|dosra|doosra|tisra|teesra|chautha|panchwa|"
    r"पहला|दूसरा|तीसरा)\b(?:\s+(?:one|project|item|wala|wali))?",
    re.IGNORECASE,
)

LAST_REF = re.compile(r"\b(?:the\s+)?(last|final|bottom|aakhri|akheri)\s+(?:one|project|item|wala|wali)?\b", re.IGNORECASE)



def new_context() -> dict:
    """The empty structured context stored on a conversation."""
    return {
        "current_project": None,        # {"project_code": ..., "name": ...}
        "recent_projects": [],          # ordered, most recently listed first
        "last_intent": None,
        "last_result_kind": None,
        "state": None,
        "sector": None,
        "period": None,
        "attachments": [],              # [{"id","filename","kind","status"}]
        "off_topic_strikes": 0,
        "turn_count": 0,
    }


def get_or_create(db: Session, conversation_id: str | None,
                  owner: str | None = None) -> Conversation:
    if conversation_id:
        existing = (
            db.query(Conversation).filter_by(public_id=conversation_id).one_or_none()
        )
        if existing is not None:
            if existing.context is None:
                existing.context = new_context()
            return existing

    import secrets

    convo = Conversation(
        public_id=conversation_id or secrets.token_urlsafe(16),
        owner=owner,
        context=new_context(),
    )
    db.add(convo)
    db.commit()
    db.refresh(convo)
    return convo


def history(db: Session, convo: Conversation, limit: int = MAX_HISTORY_TURNS) -> list[ChatMessage]:
    rows = (
        db.query(ChatMessage)
        .filter_by(conversation_id=convo.id)
        .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
        .limit(limit)
        .all()
    )
    return list(reversed(rows))


def record_message(db: Session, convo: Conversation, *, role: str, content: str,
                   source: str | None = None, intent: str | None = None,
                   project_codes: list[str] | None = None,
                   attachment_ids: list[int] | None = None,
                   analysis_meta: dict | None = None) -> ChatMessage:
    message = ChatMessage(
        conversation_id=convo.id,
        role=role,
        content=content,
        source=source,
        intent=intent,
        project_codes=project_codes or [],
        attachment_ids=attachment_ids or [],
        analysis_meta=analysis_meta or {},
    )
    db.add(message)
    convo.updated_at = dt.datetime.now(dt.timezone.utc)
    db.commit()
    db.refresh(message)
    return message


# ---------------------------------------------------------------------------
# Context maintenance
# ---------------------------------------------------------------------------
def remember_projects(context: dict, projects: list[dict], replace: bool = True) -> dict:
    """Record the projects an answer just listed, most recent first.

    ``replace=False`` merges instead. That matters: after "which projects are
    high risk?" lists ten, "why is the first one risky?" produces a
    single-project answer, and if that answer replaced the list then "compare it
    with the second one" would have nothing to count against. The ordinal
    ordering has to survive until a new list supersedes it.
    """
    if not projects:
        return context
    entries = [
        {"project_code": p.get("project_code"), "name": p.get("name")}
        for p in projects if p.get("project_code")
    ]
    if not entries:
        return context

    if replace:
        context["recent_projects"] = entries[:MAX_RECENT_PROJECTS]
    else:
        existing = context.get("recent_projects") or []
        known = {e.get("project_code") for e in existing}
        merged = existing + [e for e in entries if e["project_code"] not in known]
        context["recent_projects"] = merged[:MAX_RECENT_PROJECTS]

    # A single-project answer sets the conversational focus. A list does not:
    # "why is the first one risky" must still be explicit about which.
    if len(entries) == 1:
        context["current_project"] = entries[0]
    return context


def update_from_result(context: dict, result: dict) -> dict:
    """Fold a verified PAIMANA result back into the conversation context."""
    context["last_intent"] = result.get("intent")
    if result.get("period"):
        context["period"] = result["period"]
    if result.get("state"):
        context["state"] = result["state"]
    if result.get("sector"):
        context["sector"] = result["sector"]

    if result.get("project"):
        context["current_project"] = {
            "project_code": result["project"].get("project_code"),
            "name": result["project"].get("name"),
        }
        remember_projects(context, [result["project"]], replace=False)
        context["last_result_kind"] = "single_project"
    elif result.get("projects"):
        # A fresh list supersedes the previous ordering.
        remember_projects(context, result["projects"], replace=True)
        context["last_result_kind"] = "project_list"
    elif result.get("interventions") or result.get("alerts"):
        context["last_result_kind"] = "list"
    return context


# ---------------------------------------------------------------------------
# Reference resolution
# ---------------------------------------------------------------------------
def has_reference(question: str) -> bool:
    """Does this message lean on something said earlier?"""
    return bool(ANAPHORA.search(question or ""))


def names_a_project_explicitly(question: str) -> bool:
    """Cheap check for an explicit project code in the message."""
    return bool(re.search(r"\b\d{5,10}\b", question or ""))


def resolve_reference(context: dict, question: str) -> dict:
    """Resolve "it" / "the first one" / "compare it with the second".

    Returns ``{"projects": [...], "resolved": bool, "ambiguous": bool, "note": str}``.
    Never guesses: if the reference cannot be tied to a specific project the
    caller is told, so it can ask rather than answer about the wrong project.
    """
    q = (question or "").lower()
    recent = context.get("recent_projects") or []
    current = context.get("current_project")

    if not ANAPHORA.search(q):
        return {"projects": [], "resolved": False, "ambiguous": False, "note": None}

    picked: list[dict] = []

    # Ordinal references index into the last listed set, in the order shown.
    for match in ORDINAL_REF.finditer(q):
        word = match.group(1).lower()
        index = ORDINALS.get(word)
        if index is None:
            continue
        if index < len(recent):
            if recent[index] not in picked:
                picked.append(recent[index])
        else:
            return {
                "projects": [], "resolved": False, "ambiguous": True,
                "note": (f"You referred to the {word} project, but the previous answer "
                         f"listed {len(recent)}."),
            }

    if LAST_REF.search(q) and recent:
        if recent[-1] not in picked:
            picked.append(recent[-1])

    if picked:
        return {"projects": picked, "resolved": True, "ambiguous": False,
                "note": "Resolved from the previous answer."}

    # Bare pronouns resolve to the single project in focus.
    if current and current.get("project_code"):
        return {"projects": [current], "resolved": True, "ambiguous": False,
                "note": f"Interpreted as {current.get('name') or current['project_code']}."}

    if len(recent) == 1:
        return {"projects": [recent[0]], "resolved": True, "ambiguous": False,
                "note": "Resolved from the previous answer."}

    if recent:
        return {
            "projects": [], "resolved": False, "ambiguous": True,
            "note": ("The previous answer listed several projects, so I am not sure which "
                     "one you mean."),
            "options": recent[:5],
        }

    return {"projects": [], "resolved": False, "ambiguous": False,
            "note": "There is no earlier project in this conversation to refer back to."}


def rewrite_with_reference(question: str, projects: list[dict]) -> str:
    """Produce an explicit question for the existing planner.

    The planner in ``services/assistant.py`` resolves projects by name or code
    from the question text. Rather than change it, the resolved project name is
    substituted in, so the existing, tested lookup path does the work.
    """
    if not projects:
        return question
    names = " and ".join(
        f'"{p.get("name") or p.get("project_code")}"' for p in projects[:3]
    )
    cleaned = ORDINAL_REF.sub("", question)
    cleaned = re.sub(r"\b(it|its|it's|this project|that project|the project|them|they|"
                     r"the one|the same)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ?.,")
    return f"{cleaned} for {names}".strip()


def compact_summary(context: dict, recent: list[ChatMessage]) -> str:
    """A short natural-language recap for the model.

    Deliberately not a transcript. Roughly 400 characters of state beats 12
    turns of dialogue for both cost and reliability.
    """
    bits = []
    if context.get("current_project"):
        cp = context["current_project"]
        bits.append(f"Project in focus: {cp.get('name')} ({cp.get('project_code')}).")
    if context.get("recent_projects") and len(context["recent_projects"]) > 1:
        listed = ", ".join(
            f"{i + 1}. {p.get('name')}" for i, p in enumerate(context["recent_projects"][:5])
        )
        bits.append(f"Projects most recently listed, in order: {listed}.")
    if context.get("state"):
        bits.append(f"State filter in play: {context['state']}.")
    if context.get("sector"):
        bits.append(f"Sector filter in play: {context['sector']}.")
    if context.get("period"):
        bits.append(f"Reporting period: {context['period']}.")
    if context.get("attachments"):
        names = ", ".join(a["filename"] for a in context["attachments"][:5])
        bits.append(f"Files attached to this conversation: {names}.")

    if recent:
        last_user = next((m for m in reversed(recent) if m.role == "user"), None)
        if last_user:
            bits.append(f"The user's previous message was: \"{last_user.content[:180]}\"")
    return " ".join(bits)


def sync_attachments(context: dict, attachments: list) -> dict:
    context["attachments"] = [
        {
            "id": a.id,
            "filename": a.filename,
            "kind": a.detected_kind,
            "status": a.status,
        }
        for a in attachments
    ]
    return context


def resolve_project_records(db: Session, entries: list[dict]) -> list[Project]:
    codes = [e["project_code"] for e in entries if e.get("project_code")]
    if not codes:
        return []
    return db.query(Project).filter(Project.project_code.in_(codes)).all()
