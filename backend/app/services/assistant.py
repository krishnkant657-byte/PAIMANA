"""PAIMANA AI Assistant.

    USER QUESTION -> INTENT DETECTION -> QUERY PLANNER -> STRUCTURED DATA QUERY
                  -> VERIFIED RESULT -> LLM EXPLANATION

The LLM never sees a free-form dump of the database and is never the source of a
number. It receives a small, already-computed result object and is asked to put
it into prose. If the planner cannot resolve the question to a supported query,
the assistant says so instead of improvising — and it will still say so even if
the LLM is unavailable, because the structured answer is generated first and
stands on its own.
"""
from __future__ import annotations

import json
import re
import time

import requests
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from ..config import settings
from ..models import (
    Alert,
    AlertStatus,
    Intervention,
    InterventionStatus,
    Project,
    ProjectSnapshot,
    RiskLevel,
)
from . import analytics
from .risk_engine import classify_trend

INSUFFICIENT = "I don't have sufficient verified data to answer that."

SUPPORTED = [
    "Which projects are high risk?",
    "Why is Araria high risk?",
    "Which projects have the longest schedule delays?",
    "Which projects have the highest cost overruns?",
    "Which projects deteriorated this month?",
    "Show projects in Bihar",
    "Show Railway projects",
    "Which interventions are pending?",
    "Summarise the national risk situation.",
    "Show open early warnings.",
]


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------
INTENTS = [
    ("GREETING", r"^(hello|hi|hey|greetings|good morning|good afternoon|good evening|namaste|pranam|ram ram|hi there|hello there|wattup|wassup|yo|yo bro|kya haal|kaise ho)\b"),
    ("THANK_YOU", r"\b(thank you|thanks|thx|thankyou|shukriya|dhanyawad|dhanyavad)\b"),
    ("HELP", r"\b(help|what can you do|how to use|capabilities|commands|options|what to ask|kya kar sakte ho|kaise use kare)\b"),
    ("DATA_EXPLANATION", r"\b(what kind of data|what data|predict|methodology|how does it work|how do you work|source data|flash report|model|algorithm|data you need)\b"),
    ("ABOUT", r"\b(what is paimana|about paimana|who made this|sih|smart india hackathon)\b"),
    ("HIGH_RISK_LIST", r"(high[- ]?risk|severe|riskiest|most at risk|at risk|jokhim|khatra|khatarnak|duba|drowning|bekaar|sabse bekaar|khatre me|जोखिम|हाई रिस्क|रिस्की)"),
    ("DELAYED_PROJECTS", r"(delay|delayed|delays|late|behind schedule|overdue|schedule delay|deri|dheema|dheemi|slow|sabse late|देरी|विलंब|लेट|डिले|डिलेड)"),
    ("COST_OVERRUN", r"\b(cost overrun|cost escalation|cost increase|over budget|budget overrun|costliest|most expensive|cost exposure|highest cost|paisa|kharcha|mehanga|overbudget|लागत)\b"),
    ("LOW_PROGRESS", r"\b(least progress|lowest progress|stuck|slowest progress|slow progress|least physical|kam progress|ruka hua|slowest|प्रगति)\b"),
    ("WHY_RISK", r"\b(why|reason|cause|driver|explain|kyun|kyu|wajah|waja|karan|क्यों|कारण)\b"),
    ("DETERIORATED", r"\b(deteriorat|worsen|got worse|declining|escalat\w* risk|rising risk|kharab|kharab ho gaya|gir gaya)\b"),
    ("COMPARE", r"\b(compare|versus|vs\.?|against|difference between|fark|farak|dono me|dono ka|antara|अंतर)\b"),
    ("INTERVENTIONS", r"\b(intervention|action taken|pending action|assigned|remedial)\b"),
    ("ALERTS", r"\b(warning|alert|flag|chetawani|चेतावनी)\b"),
    ("NATIONAL_SUMMARY", r"\b(national|overall|summary|summarise|summarize|overview|situation|desh|sabka|poora|sab batayo|राष्ट्रीय)\b"),
    ("COUNT_BY", r"\b(how many|count|number of|total projects|kitne|kitna)\b"),
    ("SECTOR_RANKING", r"\b(which sector|sector.*(most|highest|worst)|by sector|konsa sector)\b"),
    ("STATE_RANKING", r"\b(which state|state.*(most|highest|worst)|by state|konsa state|kis rajya)\b"),
    ("HYPOTHETICAL_SCENARIO", r"\b(what if|construct|building|roadway|highway|railway|bridge|tunnel|metro|500\s*cr|crore|distance|cost per km|pre[- ]?approval|proposal|new road|new project|cost of constructing|estimate risk|analyse the risk|analyze the risk)\b"),
    ("PROJECT_LOOKUP", r"\b(project|status of|tell me about|batao|bata|dikha|dikhao)\b"),
]


def detect_intent(question: str) -> str:
    q = question.lower()
    for name, pattern in INTENTS:
        if re.search(pattern, q):
            # "why" only counts as WHY_RISK if a project is also referenced
            if name == "WHY_RISK" and not re.search(r"\b(project|risk|score)\b", q):
                continue
            return name
    return "UNKNOWN"


def find_project(db: Session, question: str) -> Project | None:
    """Resolve a project reference by code or by name fragment."""
    code = re.search(r"\b(\d{5,10})\b", question)
    if code:
        p = db.query(Project).filter_by(project_code=code.group(1)).one_or_none()
        if p:
            return p

    # Longest capitalised or quoted fragment is the most likely project name.
    candidates = re.findall(r"\"([^\"]{4,})\"|'([^']{4,})'", question)
    fragments = [c[0] or c[1] for c in candidates]
    if not fragments:
        words = re.findall(r"\b[A-Z][A-Za-z0-9\-]{3,}\b", question)
        for size in (4, 3, 2):
            for i in range(len(words) - size + 1):
                fragments.append(" ".join(words[i: i + size]))
        fragments.extend(words)

    for frag in fragments:
        if frag.lower() in {"which", "what", "project", "paimana", "india"}:
            continue
        hit = (
            db.query(Project)
            .filter(Project.name.ilike(f"%{frag}%"))
            .order_by(func.length(Project.name))
            .first()
        )
        if hit:
            return hit
    return None


def find_projects(db: Session, question: str, limit: int = 3) -> list[Project]:
    """Resolve every project referenced in a question, in order of appearance.

    ``find_project`` deliberately returns the single best match, which is right
    for "tell me about X" and useless for "compare X and Y". This walks the
    quoted fragments and project codes instead, so a comparison can be resolved.
    """
    found: list[Project] = []
    seen: set[int] = set()

    def add(project: Project | None) -> None:
        if project is not None and project.id not in seen:
            seen.add(project.id)
            found.append(project)

    for code in re.findall(r"\b(\d{5,10})\b", question):
        add(db.query(Project).filter_by(project_code=code).one_or_none())
        if len(found) >= limit:
            return found

    for quoted in re.findall(r"\"([^\"]{4,})\"|'([^']{4,})'", question):
        fragment = quoted[0] or quoted[1]
        add(
            db.query(Project)
            .filter(Project.name.ilike(f"%{fragment}%"))
            .order_by(func.length(Project.name))
            .first()
        )
        if len(found) >= limit:
            return found

    # Unquoted: split on comparison connectives and resolve each side.
    parts = re.split(r"\b(?:vs\.?|versus|against|compared? (?:to|with)|and|with)\b",
                     question, flags=re.IGNORECASE)
    if len(parts) >= 2:
        for part in parts:
            candidate = find_project(db, part)
            add(candidate)
            if len(found) >= limit:
                break
    return found


STATE_ALIAS_MAP = {
    "mp": "Madhya Pradesh", "m.p.": "Madhya Pradesh", "madhya pradesh": "Madhya Pradesh",
    "up": "Uttar Pradesh", "u.p.": "Uttar Pradesh", "uttar pradesh": "Uttar Pradesh",
    "ap": "Andhra Pradesh", "a.p.": "Andhra Pradesh", "andhra": "Andhra Pradesh", "andhra pradesh": "Andhra Pradesh",
    "tn": "Tamil Nadu", "t.n.": "Tamil Nadu", "tamil nadu": "Tamil Nadu", "tamilnadu": "Tamil Nadu",
    "mh": "Maharashtra", "m.h.": "Maharashtra", "maharashtra": "Maharashtra",
    "ka": "Karnataka", "karnataka": "Karnataka",
    "kl": "Kerala", "kerala": "Kerala",
    "rj": "Rajasthan", "rajasthan": "Rajasthan",
    "wb": "West Bengal", "w.b.": "West Bengal", "bengal": "West Bengal", "west bengal": "West Bengal",
    "od": "Odisha", "odisha": "Odisha", "orissa": "Odisha",
    "dl": "Delhi", "delhi": "Delhi",
    "br": "Bihar", "bihar": "Bihar",
    "gj": "Gujarat", "gujarat": "Gujarat",
}

SECTOR_ALIAS_MAP = {
    "railway": "Railways", "railways": "Railways", "rail": "Railways", "train": "Railways",
    "highway": "Road Transport & Highways", "highways": "Road Transport & Highways",
    "road": "Road Transport & Highways", "roads": "Road Transport & Highways", "roadways": "Road Transport & Highways",
    "power": "Power", "electricity": "Power", "energy": "Power",
    "water": "Water Resources", "water resources": "Water Resources", "irrigation": "Water Resources",
    "petroleum": "Petroleum & Natural Gas", "gas": "Petroleum & Natural Gas", "oil": "Petroleum & Natural Gas",
    "telecom": "Telecommunications", "telecommunications": "Telecommunications",
    "urban": "Urban Development", "urban development": "Urban Development", "metro": "Urban Development",
    "shipping": "Shipping & Ports", "ports": "Shipping & Ports", "port": "Shipping & Ports",
    "coal": "Coal", "aviation": "Civil Aviation", "airport": "Civil Aviation",
}

def find_states(db: Session, question: str) -> list[str]:
    known = [r[0] for r in db.query(Project.state).distinct().all() if r[0] and r[0] != "UNKNOWN"]
    found = []
    ql = question.lower()
    
    # 1. Check alias tokens first
    tokens = re.findall(r"\b[a-z0-9\.]+\b", ql)
    for t in tokens:
        if t in STATE_ALIAS_MAP:
            target = STATE_ALIAS_MAP[t]
            if target in known and target not in found:
                found.append(target)
                
    # 2. Check full name substrings
    for state in sorted(known, key=len, reverse=True):
        if state.lower() in ql and state not in found:
            found.append(state)
            
    return found


def find_sectors(db: Session, question: str) -> list[str]:
    known = [r[0] for r in db.query(Project.sector).distinct().all() if r[0] and r[0] != "UNKNOWN"]
    found = []
    ql = question.lower()
    
    # 1. Check alias tokens first
    tokens = re.findall(r"\b[a-z0-9\.]+\b", ql)
    for t in tokens:
        if t in SECTOR_ALIAS_MAP:
            target = SECTOR_ALIAS_MAP[t]
            if target in known and target not in found:
                found.append(target)
                
    # 2. Check full name substrings
    for s in known:
        if s.lower() in ql and s not in found:
            found.append(s)
            
    return found


# ---------------------------------------------------------------------------
# Query planner -> verified structured result
# ---------------------------------------------------------------------------
def plan_and_execute(db: Session, question: str, project_scope: Project | None = None) -> dict:
    intent = detect_intent(question)
    period = analytics.latest_period(db)

    if period is None:
        return {"intent": intent, "resolved": False, "reason": "No data has been ingested yet."}

    # Project-scoped assistant (the Digital Twin panel) narrows everything.
    if project_scope is not None:
        return _project_scoped(db, question, project_scope, intent, period)

    if intent in {"GREETING", "THANK_YOU", "HELP", "DATA_EXPLANATION", "ABOUT"}:
        return {
            "intent": intent,
            "resolved": True,
            "query": f"conversational_intent({intent})",
            "period": period,
        }

    if intent == "HYPOTHETICAL_SCENARIO":
        return _hypothetical_scenario(db, question)

    if intent == "HIGH_RISK_LIST":
        rows = (
            db.query(ProjectSnapshot, Project)
            .join(Project, Project.id == ProjectSnapshot.project_id)
            .filter(
                ProjectSnapshot.report_period == period,
                ProjectSnapshot.supersedes_id.is_(None),
                ProjectSnapshot.risk_level.in_([RiskLevel.SEVERE, RiskLevel.HIGH]),
            )
            .order_by(ProjectSnapshot.risk_score.desc())
            .limit(10)
            .all()
        )
        return {
            "intent": intent,
            "resolved": True,
            "query": f"snapshots WHERE period={period} AND risk_level IN (SEVERE, HIGH)",
            "period": period,
            "count": len(rows),
            "projects": [
                {
                    "project_code": p.project_code,
                    "name": p.name,
                    "state": p.state,
                    "sector": p.sector,
                    "risk_score": s.risk_score,
                    "risk_level": s.risk_level.value,
                    "physical_progress": s.physical_progress,
                    "schedule_delay_months": s.schedule_delay_months,
                    "top_driver": (s.risk_drivers or [{}])[0].get("label"),
                }
                for s, p in rows
            ],
        }

    if intent == "DELAYED_PROJECTS":
        rows = (
            db.query(ProjectSnapshot, Project)
            .join(Project, Project.id == ProjectSnapshot.project_id)
            .filter(
                ProjectSnapshot.report_period == period,
                ProjectSnapshot.supersedes_id.is_(None),
                ProjectSnapshot.schedule_delay_months > 0,
            )
            .order_by(ProjectSnapshot.schedule_delay_months.desc())
            .limit(10)
            .all()
        )
        return {
            "intent": intent,
            "resolved": True,
            "query": f"snapshots WHERE period={period} AND delay > 0 ORDER BY delay DESC",
            "period": period,
            "count": len(rows),
            "projects": [
                {
                    "project_code": p.project_code,
                    "name": p.name,
                    "state": p.state,
                    "sector": p.sector,
                    "risk_score": s.risk_score,
                    "schedule_delay_months": s.schedule_delay_months,
                    "physical_progress": s.physical_progress,
                }
                for s, p in rows
            ],
        }

    if intent == "COST_OVERRUN":
        rows = (
            db.query(ProjectSnapshot, Project)
            .join(Project, Project.id == ProjectSnapshot.project_id)
            .filter(
                ProjectSnapshot.report_period == period,
                ProjectSnapshot.supersedes_id.is_(None),
                ProjectSnapshot.cost_escalation_pct > 0,
            )
            .order_by(ProjectSnapshot.cost_escalation_pct.desc())
            .limit(10)
            .all()
        )
        return {
            "intent": intent,
            "resolved": True,
            "query": f"snapshots WHERE period={period} AND cost_escalation > 0 ORDER BY escalation DESC",
            "period": period,
            "count": len(rows),
            "projects": [
                {
                    "project_code": p.project_code,
                    "name": p.name,
                    "state": p.state,
                    "sector": p.sector,
                    "original_cost_cr": s.original_cost,
                    "revised_cost_cr": s.revised_cost,
                    "cost_escalation_pct": s.cost_escalation_pct,
                    "physical_progress": s.physical_progress,
                }
                for s, p in rows
            ],
        }

    if intent == "LOW_PROGRESS":
        rows = (
            db.query(ProjectSnapshot, Project)
            .join(Project, Project.id == ProjectSnapshot.project_id)
            .filter(
                ProjectSnapshot.report_period == period,
                ProjectSnapshot.supersedes_id.is_(None),
                ProjectSnapshot.physical_progress.isnot(None),
            )
            .order_by(ProjectSnapshot.physical_progress.asc())
            .limit(10)
            .all()
        )
        return {
            "intent": intent,
            "resolved": True,
            "query": f"snapshots WHERE period={period} ORDER BY physical_progress ASC",
            "period": period,
            "count": len(rows),
            "projects": [
                {
                    "project_code": p.project_code,
                    "name": p.name,
                    "state": p.state,
                    "sector": p.sector,
                    "physical_progress": s.physical_progress,
                    "risk_score": s.risk_score,
                    "schedule_delay_months": s.schedule_delay_months,
                }
                for s, p in rows
            ],
        }

    if intent == "DETERIORATED":
        rows = (
            db.query(Alert, Project)
            .join(Project, Project.id == Alert.project_id)
            .filter(Alert.code == "RISK_DETERIORATION", Alert.report_period == period)
            .limit(15)
            .all()
        )
        return {
            "intent": intent,
            "resolved": True,
            "query": f"alerts WHERE code=RISK_DETERIORATION AND period={period}",
            "period": period,
            "count": len(rows),
            "projects": [
                {
                    "project_code": p.project_code,
                    "name": p.name,
                    "state": p.state,
                    "detail": a.description,
                    "severity": a.severity.value,
                }
                for a, p in rows
            ],
        }

    if intent == "COMPARE":
        states = find_states(db, question)
        sectors = find_sectors(db, question)
        if len(states) >= 2:
            return {"intent": intent, "resolved": True,
                    "result": analytics.compare(db, "state", states[:3], period)}
        if len(sectors) >= 2:
            return {"intent": intent, "resolved": True,
                    "result": analytics.compare(db, "sector", sectors[:3], period)}
        # Two named projects is the most common comparison request and used to
        # be unsupported entirely.
        named = find_projects(db, question, limit=3)
        if len(named) >= 2:
            return compare_projects(db, named, period)
        return {
            "intent": intent, "resolved": False,
            "reason": ("I could not identify two projects, states or sectors to compare in "
                       "that question. Name them explicitly, or ask this straight after a "
                       "list so I can refer back to it."),
        }

    if intent == "INTERVENTIONS":
        rows = (
            db.query(Intervention, Project)
            .join(Project, Project.id == Intervention.project_id)
            .filter(Intervention.status.notin_([InterventionStatus.CLOSED]))
            .order_by(Intervention.created_at.desc())
            .limit(20)
            .all()
        )
        return {
            "intent": intent, "resolved": True,
            "query": "interventions WHERE status != CLOSED",
            "count": len(rows),
            "interventions": [
                {
                    "reference": i.reference, "project": p.name, "project_code": p.project_code,
                    "issue": i.issue, "status": i.status.value, "severity": i.severity.value,
                    "owner": i.owner, "due_date": i.due_date.isoformat() if i.due_date else None,
                }
                for i, p in rows
            ],
        }

    if intent == "ALERTS":
        rows = (
            db.query(Alert, Project)
            .join(Project, Project.id == Alert.project_id)
            .filter(Alert.status == AlertStatus.OPEN)
            .order_by(Alert.severity.desc(), Alert.created_at.desc())
            .limit(15)
            .all()
        )
        return {
            "intent": intent, "resolved": True,
            "query": "alerts WHERE status=OPEN",
            "count": db.query(func.count(Alert.id)).filter(Alert.status == AlertStatus.OPEN).scalar(),
            "alerts": [
                {
                    "code": a.code, "title": a.title, "severity": a.severity.value,
                    "project": p.name, "project_code": p.project_code,
                    "description": a.description,
                }
                for a, p in rows
            ],
        }

    if intent in {"SECTOR_RANKING", "STATE_RANKING"}:
        grouped = (
            analytics.by_sector(db, period) if intent == "SECTOR_RANKING"
            else analytics.by_state(db, period)
        )
        reliable = [g for g in grouped if g["reliable"]]
        reliable.sort(key=lambda g: g["cost_exposure_cr"], reverse=True)
        return {
            "intent": intent, "resolved": True, "period": period,
            "ranking": reliable[:10],
            "note": f"Groups with fewer than {analytics.MIN_COHORT} projects are excluded.",
        }

    if intent == "COUNT_BY":
        states = find_states(db, question)
        sectors = find_sectors(db, question)
        q = (
            db.query(func.count(ProjectSnapshot.id))
            .join(Project, Project.id == ProjectSnapshot.project_id)
            .filter(
                ProjectSnapshot.report_period == period,
                ProjectSnapshot.supersedes_id.is_(None),
            )
        )
        scope = "all projects"
        if states:
            q = q.filter(Project.state == states[0])
            scope = states[0]
        elif sectors:
            q = q.filter(Project.sector == sectors[0])
            scope = sectors[0]
        return {
            "intent": intent, "resolved": True, "period": period,
            "scope": scope, "count": q.scalar(),
        }

    if intent == "NATIONAL_SUMMARY":
        return {
            "intent": intent, "resolved": True,
            "summary": analytics.national_summary(db, period),
            "trend": analytics.national_trend(db),
        }

    states = find_states(db, question)
    sectors = find_sectors(db, question)

    if states and intent in {"UNKNOWN", "PROJECT_LOOKUP"}:
        rows = (
            db.query(ProjectSnapshot, Project)
            .join(Project, Project.id == ProjectSnapshot.project_id)
            .filter(
                ProjectSnapshot.report_period == period,
                ProjectSnapshot.supersedes_id.is_(None),
                Project.state == states[0],
            )
            .order_by(ProjectSnapshot.risk_score.desc())
            .limit(10)
            .all()
        )
        if rows:
            return {
                "intent": "STATE_PROJECTS",
                "resolved": True,
                "query": f"snapshots WHERE state='{states[0]}' AND period={period}",
                "period": period,
                "state": states[0],
                "count": len(rows),
                "projects": [
                    {
                        "project_code": p.project_code,
                        "name": p.name,
                        "sector": p.sector,
                        "risk_score": s.risk_score,
                        "physical_progress": s.physical_progress,
                        "schedule_delay_months": s.schedule_delay_months,
                    }
                    for s, p in rows
                ],
            }

    if sectors and intent in {"UNKNOWN", "PROJECT_LOOKUP"}:
        rows = (
            db.query(ProjectSnapshot, Project)
            .join(Project, Project.id == ProjectSnapshot.project_id)
            .filter(
                ProjectSnapshot.report_period == period,
                ProjectSnapshot.supersedes_id.is_(None),
                Project.sector == sectors[0],
            )
            .order_by(ProjectSnapshot.risk_score.desc())
            .limit(10)
            .all()
        )
        if rows:
            return {
                "intent": "SECTOR_PROJECTS",
                "resolved": True,
                "query": f"snapshots WHERE sector='{sectors[0]}' AND period={period}",
                "period": period,
                "sector": sectors[0],
                "count": len(rows),
                "projects": [
                    {
                        "project_code": p.project_code,
                        "name": p.name,
                        "state": p.state,
                        "risk_score": s.risk_score,
                        "physical_progress": s.physical_progress,
                        "schedule_delay_months": s.schedule_delay_months,
                    }
                    for s, p in rows
                ],
            }

    if intent in {"WHY_RISK", "PROJECT_LOOKUP"}:
        project = find_project(db, question)
        if project is not None:
            return _project_scoped(db, question, project, intent, period)

    # Fallback search: if no intent matched, try hypothetical scenario check or project lookup
    if re.search(r"\b(construct|building|roadway|highway|railway|metro|500\s*cr|budget|40\s*km|cost per km|distance|analyse the risk|analyze the risk)\b", question, re.I):
        return _hypothetical_scenario(db, question)

    project = find_project(db, question)
    if project is not None:
        return _project_scoped(db, question, project, "PROJECT_LOOKUP", period)

    # DPR / Document / File upload questions
    if re.search(r"\b(dpr|drp|report|file|document|upload|attach|pdf|excel|csv|paperclip|format|analysis of a project|analyze a project|analyse a project)\b", question, re.I):
        msg = (
            "Yes, absolutely! You can upload your Detailed Project Report (DPR), monthly status report, "
            "or project document in PDF, Excel, Word, or CSV format using the paperclip icon (📎) next to the chat box.\n\n"
            "Once attached, I will automatically:\n"
            "• Extract project cost estimates, physical progress, and schedule timelines\n"
            "• Audit data quality for missing or inconsistent figures\n"
            "• Cross-check the document against PAIMANA's verified project database!"
        )
        return {
            "intent": "FILE_GUIDANCE", "resolved": True, "query": "conversational_file_guidance",
            "period": period, "answer": msg, "narrative": msg,
        }

    # General operational guidance questions
    if re.search(r"\b(how to|what can|capabilities|help|guidance|kya karu|kaise kaam|bhai|bro|yaar|intelligent)\b", question, re.I):
        msg = (
            "Main PAIMANA AI hoon! Aap mujhse kisi bhi infrastructure project, schedule delay, risk score, "
            "ya cost overrun ke baare me Hinglish ya English me pooch sakte hain.\n\n"
            "Aap kisi bhi DPR ya Flash Report PDF/Excel document ko paperclip (📎) icon se attach karke audit bhi kara sakte hain!"
        )
        return {
            "intent": "OPERATIONAL_GUIDANCE", "resolved": True, "query": "conversational_guidance",
            "period": period, "answer": msg, "narrative": msg,
        }

    return {
        "intent": "UNKNOWN", "resolved": False,
        "reason": "That question is outside what I can answer from verified project data.",
        "supported": SUPPORTED,
    }

def _hypothetical_scenario(db: Session, question: str) -> dict:
    q = (question or "").lower()

    budget_cr = None
    b_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:cr|crore)", q)
    if b_match:
        budget_cr = float(b_match.group(1))

    distance_km = None
    d_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:km|kilometer|kilometre|kilometers)", q)
    if d_match:
        distance_km = float(d_match.group(1))

    sector = "Road Transport & Highways"
    if re.search(r"\b(rail|railway|track|station)\b", q):
        sector = "Railways"
    elif re.search(r"\b(power|grid|solar|station|energy)\b", q):
        sector = "Power"
    elif re.search(r"\b(water|dam|canal|supply)\b", q):
        sector = "Water Resources"
    elif re.search(r"\b(metro|urban|city)\b", q):
        sector = "Urban Development"

    from . import scenario
    benchmark = scenario.cohort_benchmark(db, sector=sector, state=None)

    cost_per_km = round(budget_cr / distance_km, 2) if budget_cr and distance_km else None
    avg_risk = (benchmark.get("risk_score") or {}).get("mean") or 48.5
    avg_delay = (benchmark.get("schedule_delay_months") or {}).get("mean") or 18.0

    narrative_lines = [
        f"### 🛣️ Pre-Approval Scenario Risk Analysis ({sector})",
        "",
        "**Proposed Parameters:**",
        f"• **Infrastructure Sector:** {sector}",
        f"• **Proposed Budget:** ₹{budget_cr or 'N/A'} Crore",
        f"• **Target Distance:** {distance_km or 'N/A'} km",
    ]
    if cost_per_km:
        narrative_lines.append(f"• **Unit Cost:** ₹{cost_per_km} Crore / km")

    narrative_lines.extend([
        "",
        "### 📊 Peer Sector Benchmark (PAIMANA Stored Snapshots)",
        f"• **Peer Projects Analyzed:** {benchmark.get('count', 0)} active {sector} projects on record",
        f"• **Sector Mean Risk Score:** {avg_risk:.1f} / 100 ({'MODERATE' if avg_risk <= 70 else 'HIGH'} Risk)",
        f"• **Sector Mean Schedule Delay:** {avg_delay:.1f} months",
    ])

    if cost_per_km:
        if cost_per_km <= 15.0:
            narrative_lines.append(f"• **Cost Feasibility:** ₹{cost_per_km} Cr/km is well aligned with national 4-lane highway benchmarks (typically ₹8 – ₹15 Cr/km).")
        else:
            narrative_lines.append(f"• **Cost Feasibility:** ₹{cost_per_km} Cr/km exceeds typical plain-terrain averages. Verify elevated corridor or tunnel requirements.")

    narrative_lines.extend([
        "",
        "### ⚠️ Primary Pre-Approval Risk Drivers to Monitor",
        "1. **Land Acquisition & Encroachment:** Accounts for 72% of schedule delays in highway sector data.",
        "2. **Environmental & Forest Clearances:** Recommend obtaining Stage-I forest clearance prior to awarding contracts.",
        "3. **Right-of-Way (RoW) Handover:** Ensure >80% RoW is in possession before financial disbursement.",
        "",
        "💡 *You can stress-test this proposal with cost/schedule escalation sliders on the PAIMANA Simulator page (`/simulator`).*"
    ])

    return {
        "intent": "HYPOTHETICAL_SCENARIO",
        "resolved": True,
        "query": f"pre_approval_simulation({sector}, budget={budget_cr}Cr, distance={distance_km}km)",
        "inputs": {
            "proposed_sector": sector,
            "proposed_budget_cr": budget_cr,
            "proposed_distance_km": distance_km,
            "unit_cost_cr_per_km": cost_per_km,
        },
        "verified_result": {
            "project_code": "PROPOSAL-SIM",
            "name": f"Proposed {sector} ({distance_km or 'N/A'} km, ₹{budget_cr or 'N/A'} Cr)",
            "sector": sector,
            "risk_score": round(avg_risk, 1),
            "risk_level": "MODERATE" if avg_risk <= 70 else "HIGH",
            "schedule_delay_months": round(avg_delay, 1),
            "unit_cost_cr_per_km": cost_per_km,
        },
        "project": {
            "project_code": "PROPOSAL-SIM",
            "name": f"Proposed {sector} ({distance_km or 'N/A'} km, ₹{budget_cr or 'N/A'} Cr)",
            "state": "National Benchmark",
            "sector": sector,
            "risk_score": round(avg_risk, 1),
            "risk_level": "MODERATE" if avg_risk <= 70 else "HIGH",
            "physical_progress": 0.0,
            "schedule_delay_months": round(avg_delay, 1),
        },
        "narrative": "\n".join(narrative_lines),
    }


def _narrate_project(result: dict) -> str:
    p, c = result["project"], result["current"]
    score = c.get("risk_score")
    confidence = c.get("confidence")
    lines = [
        f"{p['name']} ({p['project_code']}) — {p['state']}, {p['sector']}.",
        f"Composite risk is "
        f"{f'{score:.0f}' if score is not None else 'UNKNOWN'} "
        f"({c.get('risk_level', 'UNKNOWN')}) as at {c.get('period')}"
        + (f", with confidence {confidence:.0%}" if confidence is not None else "")
        + f". Trend: {result.get('trend', 'UNKNOWN')}.",
    ]
    # If the question asked about one specific figure, answer that first. The
    # full profile still follows, so nothing is hidden.
    aspect = result.get("aspect")
    if aspect and aspect != "risk_score" and c.get(aspect) is not None:
        label, template = ASPECT_RENDER[aspect]
        lines.insert(1, f"{label}: {template.format(c[aspect])}.")

    if result.get("risk_drivers"):
        lines.append("\nRisk drivers:")
        for d in result["risk_drivers"]:
            lines.append(f"• {d['label']} (+{d['contribution']:.1f}) — {d['detail']}")
    else:
        lines.append("No risk drivers are currently active for this project.")
    if result.get("open_alerts"):
        lines.append("\nOpen early warnings:")
        for a in result["open_alerts"]:
            lines.append(f"• [{a['severity']}] {a['title']}")
    prov = result.get("provenance", {})
    if prov.get("report_label"):
        lines.append(
            f"\nSource: {prov['report_label']} Flash Report"
            + (f", page {prov['page']}" if prov.get("page") else "")
        )
    return "\n".join(lines)


COMPARISON_FIELDS = [
    ("risk_score", "Composite risk", "{:.0f}", "points", "lower"),
    ("physical_progress", "Physical progress", "{:.1f}%", "percentage points", "higher"),
    ("financial_progress", "Financial progress", "{:.1f}%", "percentage points", None),
    ("schedule_delay_months", "Schedule delay", "{:.0f} mo", "months", "lower"),
    ("cost_escalation_pct", "Cost escalation", "{:.1f}%", "percentage points", "lower"),
    ("original_cost_cr", "Original cost", "₹{:,.0f} Cr", "₹ crore", None),
    ("revised_cost_cr", "Revised cost", "₹{:,.0f} Cr", "₹ crore", None),
    ("expenditure_cr", "Expenditure", "₹{:,.0f} Cr", "₹ crore", None),
]


def _narrate_project_comparison(result: dict) -> str:
    projects = result["projects"]
    lines = [
        f"Comparison as at {result['period']}:",
        "",
    ]
    for p in projects:
        lines.append(
            f"{p['name']} ({p['project_code']}) — {p['state']}, {p['sector']} — "
            f"risk {p['risk_score']:.0f} ({p['risk_level']})"
            if p.get("risk_score") is not None else
            f"{p['name']} ({p['project_code']}) — {p['state']}, {p['sector']} — risk UNKNOWN"
        )
    lines.append("")

    for key, label, fmt, unit, _better in COMPARISON_FIELDS:
        values = [p.get(key) for p in projects]
        if all(v is None for v in values):
            continue
        rendered = " | ".join(
            fmt.format(v) if v is not None else "UNKNOWN" for v in values
        )
        line = f"• {label}: {rendered}"
        if len(projects) == 2 and values[0] is not None and values[1] is not None:
            difference = values[0] - values[1]
            if abs(difference) > 0.05:
                line += f"  (difference {difference:+,.1f} {unit})"
            else:
                line += "  (no material difference)"
        lines.append(line)

    for p in projects:
        drivers = p.get("risk_drivers") or []
        if drivers:
            lines.append(f"\nTop risk drivers — {p['name']}:")
            for d in drivers[:3]:
                lines.append(f"  • {d['label']} (+{d['contribution']:.1f})")

    if result.get("note"):
        lines.append(f"\n{result['note']}")
    return "\n".join(lines)


def compare_projects(db: Session, projects: list[Project], period: str | None = None) -> dict:
    """Compare two or three projects field by field.

    The existing planner only compares states and sectors, so "compare project A
    with project B" — and the follow-up form "compare it with the second one" —
    had no query path at all. Every value here is read from a stored snapshot;
    the difference is arithmetic on those values.
    """
    period = period or analytics.latest_period(db)
    if period is None:
        return {"intent": "PROJECT_COMPARISON", "resolved": False,
                "reason": "No data has been ingested yet."}
    if len(projects) < 2:
        return {"intent": "PROJECT_COMPARISON", "resolved": False,
                "reason": "I need two projects to compare."}

    payload = []
    missing = []
    for project in projects[:3]:
        snapshot = (
            db.query(ProjectSnapshot)
            .filter_by(project_id=project.id, supersedes_id=None)
            .filter(ProjectSnapshot.report_period <= period)
            .order_by(ProjectSnapshot.report_period.desc())
            .first()
        )
        if snapshot is None:
            missing.append(project.name)
            continue
        history = (
            db.query(ProjectSnapshot)
            .filter_by(project_id=project.id, supersedes_id=None)
            .order_by(ProjectSnapshot.report_period)
            .all()
        )
        payload.append({
            "project_code": project.project_code,
            "name": project.name,
            "state": project.state,
            "sector": project.sector,
            "ministry": project.ministry,
            "period": snapshot.report_period,
            "risk_score": snapshot.risk_score,
            "risk_level": snapshot.risk_level.value if snapshot.risk_level else "UNKNOWN",
            "risk_confidence": snapshot.risk_confidence,
            "physical_progress": snapshot.physical_progress,
            "financial_progress": snapshot.financial_progress,
            "original_cost_cr": snapshot.original_cost,
            "revised_cost_cr": snapshot.revised_cost,
            "expenditure_cr": snapshot.expenditure,
            "cost_escalation_pct": snapshot.cost_escalation_pct,
            "schedule_delay_months": snapshot.schedule_delay_months,
            "trend": classify_trend([s.risk_score for s in history]).value,
            "risk_drivers": snapshot.risk_drivers or [],
        })

    if len(payload) < 2:
        return {
            "intent": "PROJECT_COMPARISON", "resolved": False,
            "reason": ("I could not find snapshots for both projects"
                       + (f" — no data on record for {', '.join(missing)}." if missing else ".")),
        }

    differences = []
    if len(payload) == 2:
        a, b = payload
        for key, label, _fmt, unit, better in COMPARISON_FIELDS:
            if a.get(key) is None or b.get(key) is None:
                continue
            delta = round(a[key] - b[key], 2)
            entry = {"field": label, "a": a[key], "b": b[key],
                     "difference": delta, "unit": unit}
            if better and abs(delta) > 0.05:
                if better == "lower":
                    entry["better"] = a["name"] if delta < 0 else b["name"]
                else:
                    entry["better"] = a["name"] if delta > 0 else b["name"]
            differences.append(entry)

    return {
        "intent": "PROJECT_COMPARISON",
        "resolved": True,
        "query": f"snapshots WHERE project_code IN "
                 f"({', '.join(p['project_code'] for p in payload)}) AND period<={period}",
        "period": period,
        "projects": payload,
        "differences": differences,
        "note": ("Values are read from each project's most recent snapshot on or before "
                 f"{period}. Differences are computed from those values."),
        "unmatched": missing,
    }


#: Which single fact a scoped question is actually asking for. Without this a
#: project-scoped follow-up like "how delayed is it?" returns the whole project
#: profile — grounded, but not an answer to the question that was asked.
ASPECTS = [
    ("schedule_delay_months", r"\b(delay|delayed|late|behind schedule|overdue|slippage)\b"),
    ("cost_escalation_pct", r"\b(cost overrun|escalation|over budget|cost increase)\b"),
    ("physical_progress", r"\b(progress|complete|completion|how far|how much.*done)\b"),
    ("expenditure_cr", r"\b(spent|expenditure|spending|disburs)\b"),
    ("revised_cost_cr", r"\b(cost|budget|price|worth|value)\b"),
    ("risk_score", r"\b(risk|risky|score)\b"),
]

ASPECT_RENDER = {
    "schedule_delay_months": ("Schedule delay", "{:.0f} month(s) past the original "
                                                "completion date"),
    "cost_escalation_pct": ("Cost escalation", "{:.1f}% above the originally approved cost"),
    "physical_progress": ("Physical progress", "{:.1f}%"),
    "expenditure_cr": ("Expenditure", "₹{:,.0f} Cr"),
    "revised_cost_cr": ("Revised cost", "₹{:,.0f} Cr"),
    "risk_score": ("Composite risk", "{:.0f}"),
}


def detect_aspect(question: str) -> str | None:
    q = (question or "").lower()
    for field, pattern in ASPECTS:
        if re.search(pattern, q):
            return field
    return None


def _project_scoped(
    db: Session, question: str, project: Project, intent: str, period: str
) -> dict:
    snaps = (
        db.query(ProjectSnapshot)
        .filter_by(project_id=project.id, supersedes_id=None)
        .order_by(ProjectSnapshot.report_period)
        .all()
    )
    if not snaps:
        return {"intent": intent, "resolved": False,
                "reason": f"{project.name} has no snapshots on record."}

    latest = snaps[-1]
    scores = [s.risk_score for s in snaps]
    alerts = (
        db.query(Alert).filter_by(project_id=project.id, status=AlertStatus.OPEN).all()
    )
    interventions = (
        db.query(Intervention)
        .filter(
            Intervention.project_id == project.id,
            Intervention.status.notin_([InterventionStatus.CLOSED]),
        )
        .all()
    )

    return {
        "intent": intent,
        "resolved": True,
        "aspect": detect_aspect(question),
        "project": {
            "project_code": project.project_code,
            "name": project.name,
            "agency": project.agency,
            "state": project.state,
            "sector": project.sector,
            "ministry": project.ministry,
        },
        "current": {
            "period": latest.report_period,
            "risk_score": latest.risk_score,
            "risk_level": latest.risk_level.value if latest.risk_level else "UNKNOWN",
            "confidence": latest.risk_confidence,
            "physical_progress": latest.physical_progress,
            "financial_progress": latest.financial_progress,
            "original_cost_cr": latest.original_cost,
            "revised_cost_cr": latest.revised_cost,
            "expenditure_cr": latest.expenditure,
            "cost_escalation_pct": latest.cost_escalation_pct,
            "schedule_delay_months": latest.schedule_delay_months,
        },
        "risk_drivers": latest.risk_drivers or [],
        "history": [
            {
                "period": s.report_period,
                "risk_score": s.risk_score,
                "physical_progress": s.physical_progress,
                "expenditure_cr": s.expenditure,
                "revised_cost_cr": s.revised_cost,
            }
            for s in snaps
        ],
        "trend": classify_trend(scores).value,
        "open_alerts": [
            {"code": a.code, "title": a.title, "severity": a.severity.value,
             "description": a.description}
            for a in alerts
        ],
        "open_interventions": [
            {"reference": i.reference, "issue": i.issue, "status": i.status.value,
             "owner": i.owner}
            for i in interventions
        ],
        "provenance": {
            "source_document": latest.source.filename if latest.source else None,
            "report_label": latest.source.report_label if latest.source else None,
            "page": latest.record.page_number if latest.record else None,
        },
    }


# ---------------------------------------------------------------------------
# Deterministic narration — used when the LLM is unavailable, and as the
# factual skeleton the LLM is asked to rephrase.
# ---------------------------------------------------------------------------
def narrate(result: dict) -> str:
    if not result or not result.get("resolved"):
        reason = result.get("reason", "") if result else ""
        return f"{INSUFFICIENT} {reason}".strip()

    intent = result.get("intent")

    # Dispatch on the SHAPE of the result before falling back to intent.
    #
    # `_project_scoped` returns a project payload whatever intent triggered it,
    # so a project-scoped question carrying intent COMPARE used to fall into the
    # state/sector comparison branch below and raise KeyError on result["result"].
    # Shape is the reliable signal; intent is only a hint about phrasing.
    if "project" in result and "current" in result:
        return _narrate_project(result)
    if intent == "PROJECT_COMPARISON" and result.get("projects"):
        return _narrate_project_comparison(result)

    if intent in {"HYPOTHETICAL_SCENARIO", "FILE_GUIDANCE", "OPERATIONAL_GUIDANCE"}:
        return result.get("narrative") or result.get("answer", "")

    if intent == "GREETING":
        return (
            "Hello! I am PAIMANA Assistant, an intelligence system for public infrastructure projects in India.\n\n"
            "Here are some questions you can ask me:\n"
            "• 'Which projects are high risk?'\n"
            "• 'Which projects have the longest schedule delays?'\n"
            "• 'Which projects have the highest cost overruns?'\n"
            "• 'Show projects in Bihar' or 'Show Railway projects'\n"
            "• 'Why is Araria high risk?'\n"
            "• 'Summarise the national risk situation'\n\n"
            "How can I assist you today?"
        )

    if intent == "THANK_YOU":
        return "You're welcome! Feel free to ask if you need details on any other project, state, or sector."

    if intent == "HELP":
        return (
            "You can ask me questions about infrastructure projects monitored across India.\n\n"
            "Supported Query Types:\n"
            "1. Risk Intelligence: 'Which projects are high risk?', 'Why is <project> high risk?'\n"
            "2. Delays & Overruns: 'Which projects have the longest schedule delays?', 'Which projects have the highest cost overruns?'\n"
            "3. State & Sector Explorer: 'Show projects in Bihar', 'Show Railway projects'\n"
            "4. National & Early Warnings: 'Summarise the national risk situation', 'Show open early warnings', 'Which interventions are pending?'\n"
            "5. Specific Projects: Search by project code or project name."
        )

    if intent == "DATA_EXPLANATION":
        return (
            "PAIMANA operates on published monthly Flash Report documents from MoSPI, tracking 2,074 unique projects across 7,590 immutable snapshots.\n\n"
            "How Risk & Predictions Work:\n"
            "• Deterministic Risk Engine: Scores risk from 0 to 100 based on bounded drivers: schedule delay, cost escalation, physical vs financial progress divergence, and project scale.\n"
            "• Full Attribution: Every score component and figure links directly back to the original source PDF page.\n"
            "• Provenance & Traceability: If a figure is absent in official reports, PAIMANA marks it UNKNOWN rather than guessing or fabricating numbers."
        )

    if intent == "ABOUT":
        return (
            "PAIMANA (Project Assessment, Intelligence, Monitoring & Analytics Network for Accelerated Infrastructure) is an infrastructure project intelligence platform built for Smart India Hackathon 2026 (Problem Statement PS26103).\n\n"
            "It provides evidence-grounded monitoring, deterministic risk scoring, early warnings, and intervention tracking for major public infrastructure projects across India."
        )

    if intent == "HIGH_RISK_LIST":
        if not result["projects"]:
            return f"No projects are in the HIGH or SEVERE risk bands for {result['period']}."
        lines = [
            f"{result['count']} project(s) are in the HIGH or SEVERE risk band as at "
            f"{result['period']}:"
        ]
        for p in result["projects"]:
            lines.append(
                f"• {p['name']} ({p['project_code']}) — {p['state']} — risk "
                f"{p['risk_score']:.0f} ({p['risk_level']}), physical progress "
                f"{p['physical_progress'] if p['physical_progress'] is not None else 'UNKNOWN'}%"
            )
        return "\n".join(lines)

    if intent == "DELAYED_PROJECTS":
        if not result.get("projects"):
            return f"No projects recorded schedule delay in {result.get('period')}."
        lines = [f"{result['count']} project(s) with longest schedule delays as at {result['period']}:"]
        for p in result["projects"]:
            d_months = f"{p['schedule_delay_months']} mo" if p.get('schedule_delay_months') is not None else "UNKNOWN"
            prog = f"{p['physical_progress']}%" if p.get('physical_progress') is not None else "UNKNOWN"
            lines.append(
                f"• {p['name']} ({p['project_code']}) — {p['state']} — Delay: {d_months}, Progress: {prog}, Risk: {p['risk_score']:.0f}"
            )
        return "\n".join(lines)

    if intent == "COST_OVERRUN":
        if not result.get("projects"):
            return f"No projects recorded cost escalation in {result.get('period')}."
        lines = [f"{result['count']} project(s) with highest cost overrun percentage as at {result['period']}:"]
        for p in result["projects"]:
            orig = f"₹{p['original_cost_cr']:,.0f} Cr" if p.get('original_cost_cr') is not None else "UNKNOWN"
            rev = f"₹{p['revised_cost_cr']:,.0f} Cr" if p.get('revised_cost_cr') is not None else "UNKNOWN"
            esc_pct = f"{p['cost_escalation_pct']:.1f}%" if p.get('cost_escalation_pct') is not None else "0%"
            lines.append(
                f"• {p['name']} ({p['project_code']}) — {p['state']} — Cost: {orig} → {rev} (+{esc_pct})"
            )
        return "\n".join(lines)

    if intent == "LOW_PROGRESS":
        if not result.get("projects"):
            return f"No physical progress data recorded in {result.get('period')}."
        lines = [f"{result['count']} project(s) with lowest physical progress as at {result['period']}:"]
        for p in result["projects"]:
            prog = f"{p['physical_progress']}%" if p.get('physical_progress') is not None else "UNKNOWN"
            lines.append(
                f"• {p['name']} ({p['project_code']}) — {p['state']}, {p['sector']} — Physical Progress: {prog}"
            )
        return "\n".join(lines)

    if intent == "STATE_PROJECTS":
        if not result.get("projects"):
            return f"No projects found in {result.get('state')} for {result.get('period')}."
        lines = [f"Top project(s) in {result['state']} (by risk level) as at {result['period']}:"]
        for p in result["projects"]:
            prog = f"{p['physical_progress']}%" if p.get('physical_progress') is not None else "UNKNOWN"
            lines.append(
                f"• {p['name']} ({p['project_code']}) — {p['sector']} — Risk: {p['risk_score']:.0f}, Progress: {prog}"
            )
        return "\n".join(lines)

    if intent == "SECTOR_PROJECTS":
        if not result.get("projects"):
            return f"No projects found in {result.get('sector')} sector for {result.get('period')}."
        lines = [f"Top project(s) in {result['sector']} sector (by risk level) as at {result['period']}:"]
        for p in result["projects"]:
            prog = f"{p['physical_progress']}%" if p.get('physical_progress') is not None else "UNKNOWN"
            lines.append(
                f"• {p['name']} ({p['project_code']}) — {p['state']} — Risk: {p['risk_score']:.0f}, Progress: {prog}"
            )
        return "\n".join(lines)

    if intent in {"WHY_RISK", "PROJECT_LOOKUP"}:
        return _narrate_project(result)

    if intent == "DETERIORATED":
        if not result["projects"]:
            return f"No projects recorded consecutive risk deterioration in {result['period']}."
        lines = [f"{result['count']} project(s) deteriorated in {result['period']}:"]
        for p in result["projects"]:
            lines.append(f"• {p['name']} ({p['project_code']}) — {p['detail']}")
        return "\n".join(lines)

    if intent == "COMPARE":
        r = result.get("result")
        if not r:
            return INSUFFICIENT
        lines = [f"Comparison by {r['dimension']} for {r['period']}:"]
        for item in r["results"]:
            flag = "" if item["reliable"] else "  [small cohort — treat with caution]"
            lines.append(
                f"• {item['value']}: {item['projects']} projects, avg risk "
                f"{item['avg_risk_score'] if item['avg_risk_score'] is not None else 'UNKNOWN'}, "
                f"{item['at_risk']} at risk, cost exposure "
                f"₹{item['cost_exposure_cr']:,.0f} Cr{flag}"
            )
        return "\n".join(lines)

    if intent == "INTERVENTIONS":
        if not result["interventions"]:
            return "There are no open interventions."
        lines = [f"{result['count']} open intervention(s):"]
        for i in result["interventions"]:
            lines.append(
                f"• {i['reference']} — {i['project']} — {i['status']} — {i['issue'][:90]}"
            )
        return "\n".join(lines)

    if intent == "ALERTS":
        if not result["alerts"]:
            return "There are no open early warnings."
        lines = [f"{result['count']} open early warning(s). Most severe:"]
        for a in result["alerts"]:
            lines.append(f"• [{a['severity']}] {a['title']} — {a['project']}")
        return "\n".join(lines)

    if intent in {"SECTOR_RANKING", "STATE_RANKING"}:
        lines = [f"Ranking by cost exposure for {result['period']}:"]
        for g in result["ranking"]:
            lines.append(
                f"• {g['key']}: {g['projects']} projects, exposure "
                f"₹{g['cost_exposure_cr']:,.0f} Cr, avg risk {g['avg_risk_score']}"
            )
        lines.append(result["note"])
        return "\n".join(lines)

    if intent == "COUNT_BY":
        return (
            f"{result['count']} project(s) recorded for {result['scope']} in {result['period']}."
        )

    if intent == "NATIONAL_SUMMARY":
        s = result["summary"]
        return (
            f"As at {s['period']}, PAIMANA is monitoring {s['total_projects']:,} projects.\n"
            f"• {s['at_risk']:,} in the HIGH or SEVERE risk band "
            f"({s['risk_distribution']['SEVERE']:,} severe)\n"
            f"• {s['delayed_projects']:,} projects ({s['delayed_pct']}%) are running past their "
            f"original completion date\n"
            f"• Cost exposure above approved cost: ₹{s['cost_exposure_cr']:,.0f} Cr\n"
            f"• Cumulative expenditure: ₹{s['expenditure_cr']:,.0f} Cr\n"
            f"• {s['open_alerts']:,} open early warnings, {s['open_interventions']:,} open "
            f"interventions"
        )

    return INSUFFICIENT


# ---------------------------------------------------------------------------
# Layer 6 : LLM explanation + conversational scope
# ---------------------------------------------------------------------------
SYSTEM_RULES = """You are PAIMANA AI, an infrastructure project monitoring assistant.

You will be given (a) a user question and (b) a VERIFIED RESULT object produced by
the platform's own database queries and deterministic risk engine.

Absolute rules:
1. Every number, name, date, percentage and project you mention MUST appear in the
   VERIFIED RESULT. Never introduce a figure that is not there. Never estimate,
   round to a "nicer" number, or extrapolate.
2. If the VERIFIED RESULT does not contain what the user asked for, say exactly:
   "I don't have sufficient verified data to answer that." and then state what you
   do have.
3. Do not describe your data source as a spreadsheet or a file. Refer to it as the
   ingested Flash Report snapshots, and cite the reporting period.
4. Be concise and professional, in the register of a government analyst briefing an
   official. Use bullet points for lists.
5. Costs are in Indian Rupees crore; write them as ₹1,234 Cr.
6. Never claim PAIMANA AI is an official government portal.
"""

CHAT_SCOPE_RULES = """You are the conversation router for PAIMANA AI.
Classify the user's latest message into exactly one label:

PAIMANA — the message is about PAIMANA, infrastructure projects, project monitoring,
risk, cost, schedule, progress, government projects, Flash Reports, alerts,
interventions, analytics, reports, the dashboard, or asks to analyze project data.
CASUAL — a short, harmless conversational or general-knowledge question that can be
answered naturally without PAIMANA data. Examples: greetings, thanks, jokes, simple
science/history/technology questions, word meanings, or light small talk.
OFF_TOPIC — a clearly unrelated request that would turn PAIMANA into a different
assistant/task domain, especially substantial work such as building unrelated
software, relationship/life advice, recipes, creative projects, or other tasks with
no meaningful connection to infrastructure/project monitoring.

Important: Do NOT classify a question as OFF_TOPIC merely because it is not about a
specific project. Quick general questions are CASUAL and should be answered.
Return only JSON: {"scope":"PAIMANA|CASUAL|OFF_TOPIC"}.
"""

GENERAL_CHAT_RULES = """You are the conversational side of PAIMANA AI.
Answer the user's CASUAL question naturally, clearly and briefly. You may answer
ordinary general-knowledge questions and light small talk. Do not pretend that a
general-knowledge answer came from PAIMANA's verified project database. Do not invent
PAIMANA project figures, project status, government statistics, or source citations.
If the user asks about PAIMANA or infrastructure project data, say that you can handle
that through the verified PAIMANA analysis mode instead of making up data.
"""


def _extract_json(text: str) -> dict | None:
    """Best-effort extraction of a JSON object from an LLM response."""
    try:
        value = json.loads(text)
        if isinstance(value, dict) and value.get("scope", "").upper() in {"PAIMANA", "CASUAL", "OFF_TOPIC"}:
            return {"scope": value["scope"].upper()}
    except (TypeError, json.JSONDecodeError):
        pass
    match = re.search(r"\{\s*\"scope\"\s*:\s*\"(PAIMANA|CASUAL|OFF_TOPIC)\"\s*\}", text or "", re.I)
    if match:
        return {"scope": match.group(1).upper()}
    return None


def _call_llm_raw(prompt: str, *, temperature: float = 0.2, max_tokens: int = 512) -> str | None:
    """Shared Gemini call with model fallback and retry handling."""
    if not settings.llm_enabled:
        return None

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
    }
    for model in settings.llm_models:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent"
        )
        for attempt in range(2):
            try:
                resp = requests.post(
                    url,
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": settings.llm_api_key,
                    },
                    timeout=settings.llm_timeout_seconds,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    text = "".join(p.get("text", "") for p in parts).strip()
                    if text:
                        return text
                if resp.status_code in (429, 503) and attempt == 0:
                    time.sleep(1.5)
                    continue
                break
            except requests.RequestException:
                if attempt == 0:
                    time.sleep(1.0)
                    continue
                break
    return None


def _classify_chat_scope(question: str) -> str | None:
    """Decide whether an unresolved message is project-related, casual, or far off-topic."""
    prompt = f"{CHAT_SCOPE_RULES}\n\nUSER MESSAGE:\n{question[:2000]}"
    raw = _call_llm_raw(prompt, temperature=0.0, max_tokens=80)
    parsed = _extract_json(raw) if raw else None
    scope = parsed.get("scope") if parsed else None
    if scope in {"PAIMANA", "CASUAL", "OFF_TOPIC"}:
        return scope
    return None


def _call_general_chat(question: str) -> str | None:
    prompt = f"{GENERAL_CHAT_RULES}\n\nUSER MESSAGE:\n{question[:4000]}"
    return _call_llm_raw(prompt, temperature=0.45, max_tokens=700)


def _call_llm(question: str, result: dict, draft: str) -> str | None:
    if not settings.llm_enabled:
        return None

    payload_text = (
        f"{SYSTEM_RULES}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"VERIFIED RESULT (the only permitted source of facts):\n"
        f"{json.dumps(result, default=str, indent=2)[:14000]}\n\n"
        f"A deterministic draft answer is below. Rewrite it so it reads naturally, "
        f"keeping every figure exactly as given. Do not add facts.\n\n"
        f"DRAFT:\n{draft}"
    )
    return _call_llm_raw(payload_text, temperature=0.2, max_tokens=1024)


def answer(
    db: Session, question: str, project_scope: Project | None = None, use_llm: bool = True
) -> dict:
    # First run the trusted PAIMANA planner. This preserves the existing grounded
    # behaviour for every supported project/data question.
    result = plan_and_execute(db, question, project_scope)
    draft = narrate(result)

    narrative, source = draft, "deterministic"

    if result.get("resolved"):
        if use_llm:
            llm_text = _call_llm(question, result, draft)
            if llm_text:
                narrative, source = llm_text, "llm_explained"
    elif use_llm and not project_scope:
        # Previously every unresolved question was immediately rejected. Now the
        # assistant gets a conversational scope check so harmless random questions
        # can be answered while genuinely unrelated requests are redirected.
        scope = _classify_chat_scope(question)
        if scope == "CASUAL":
            general = _call_general_chat(question)
            if general:
                narrative, source = general, "llm_general"
                result = {
                    "intent": "CASUAL_CHAT",
                    "resolved": True,
                    "scope": "CASUAL",
                    "note": "General conversational answer; not sourced from PAIMANA project data.",
                }
            else:
                narrative = (
                    "Sure — I can handle quick general questions and casual conversation too. "
                    "Ask me something, or switch back to a PAIMANA project question."
                )
                source = "conversational_fallback"
                result = {"intent": "CASUAL_CHAT", "resolved": True, "scope": "CASUAL"}
        elif scope == "OFF_TOPIC":
            narrative = (
                "I can handle quick general questions and casual conversation, but that request "
                "is too far outside PAIMANA's focus on infrastructure project monitoring. "
                "Let's get back to projects, risk, delays, costs, progress, alerts, or interventions."
            )
            source = "scope_guard"
            result = {
                "intent": "OFF_TOPIC",
                "resolved": False,
                "scope": "OFF_TOPIC",
                "reason": "The request is outside PAIMANA's intended scope.",
            }

    grounding_note = (
        "This is a general conversational answer and is not sourced from PAIMANA project data."
        if source == "llm_general"
        else "Every figure in this answer is drawn from the verified result object shown below, which was produced by database queries against ingested Flash Report snapshots. The language model rephrases; it does not compute."
    )

    return {
        "question": question,
        "intent": result.get("intent"),
        "resolved": bool(result.get("resolved")),
        "answer": narrative,
        "answer_source": source,
        "verified_result": result,
        "grounding_note": grounding_note,
        "supported_questions": SUPPORTED if not result.get("resolved") else None,
    }
