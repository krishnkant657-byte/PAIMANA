"""Report generation (ReportLab).

Every report states its data sources and generation timestamp, and carries the
disclaimer that PAIMANA is an analytical prototype rather than an official
government publication.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Alert, AlertStatus, DataSource, Intervention, Project, ProjectSnapshot
from . import analytics
from .risk_engine import classify_trend

REPORT_DIR = settings.data_dir / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

NAVY = colors.HexColor("#0B2545")
ORANGE = colors.HexColor("#D9581E")
GREY = colors.HexColor("#5A6472")
LINE = colors.HexColor("#D8DDE4")

DISCLAIMER = (
    "PAIMANA is an analytical prototype developed for Smart India Hackathon 2026. "
    "It is not an official Government of India portal. Figures are derived from published "
    "Flash Report documents ingested by the platform and are reproduced for analysis only."
)


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("Title2", parent=ss["Title"], fontSize=18, textColor=NAVY,
                          spaceAfter=4, alignment=TA_CENTER))
    ss.add(ParagraphStyle("Sub", parent=ss["Normal"], fontSize=9, textColor=GREY,
                          alignment=TA_CENTER, spaceAfter=12))
    ss.add(ParagraphStyle("H", parent=ss["Heading2"], fontSize=11.5, textColor=NAVY,
                          spaceBefore=12, spaceAfter=5))
    ss.add(ParagraphStyle("Body", parent=ss["Normal"], fontSize=9.2, leading=13.5))
    ss.add(ParagraphStyle("Small", parent=ss["Normal"], fontSize=7.6, textColor=GREY,
                          leading=10.5))
    return ss


def _table(data, widths=None, align_right=None):
    t = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F6F7F9")]),
    ]
    for col in align_right or []:
        style.append(("ALIGN", (col, 1), (col, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def _fmt(v, suffix="", nd=2):
    if v is None:
        return "UNKNOWN"
    if isinstance(v, float):
        return f"{v:,.{nd}f}{suffix}"
    return f"{v:,}{suffix}" if isinstance(v, int) else f"{v}{suffix}"


def _header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, h - 16 * mm, w, 16 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(18 * mm, h - 10.5 * mm, "PAIMANA")
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(
        18 * mm, h - 14 * mm,
        "Project Assessment, Intelligence, Monitoring & Analytics Network for "
        "Accelerated Infrastructure",
    )
    canvas.setFillColor(ORANGE)
    canvas.rect(0, h - 17.2 * mm, w, 1.2 * mm, fill=1, stroke=0)

    canvas.setFillColor(GREY)
    canvas.setFont("Helvetica", 7)
    canvas.drawString(18 * mm, 10 * mm, "Analytical prototype — not an official Government of India portal")
    canvas.drawRightString(w - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _doc(path: Path, title: str) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        str(path), pagesize=A4, title=title, author="PAIMANA",
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=24 * mm, bottomMargin=18 * mm,
    )


def _sources_block(db: Session, ss) -> list:
    sources = db.query(DataSource).order_by(DataSource.report_period).all()
    rows = [["Document", "Period", "Pages", "SHA-256 (first 16)"]]
    for s in sources:
        rows.append([s.filename, s.report_label or s.report_period, str(s.page_count or "-"),
                     (s.sha256 or "")[:16]])
    return [Paragraph("Data Sources", ss["H"]),
            _table(rows, widths=[75 * mm, 30 * mm, 18 * mm, 45 * mm])]


# ---------------------------------------------------------------------------
def generate(
    db: Session,
    *,
    report_type: str,
    project_code: str | None = None,
    period: str | None = None,
    generated_by: str = "PAIMANA",
) -> tuple[Path, str]:
    ts = dt.datetime.now(dt.timezone.utc)
    stamp = ts.strftime("%Y%m%d%H%M%S")

    if report_type == "PROJECT":
        if not project_code:
            raise ValueError("A project code is required for a project report.")
        return _project_report(db, project_code, stamp, ts, generated_by)
    if report_type in {"MONTHLY_MONITORING", "EXECUTIVE_BRIEF", "RISK"}:
        return _national_report(db, report_type, period, stamp, ts, generated_by)
    raise ValueError(f"Unsupported report type: {report_type}")


def _project_report(db, project_code, stamp, ts, generated_by):
    project = db.query(Project).filter_by(project_code=project_code).one_or_none()
    if project is None:
        raise ValueError("Project not found.")
    snaps = (
        db.query(ProjectSnapshot)
        .filter_by(project_id=project.id, supersedes_id=None)
        .order_by(ProjectSnapshot.report_period)
        .all()
    )
    if not snaps:
        raise ValueError("Project has no snapshots.")
    latest = snaps[-1]

    ss = _styles()
    title = f"Project Report — {project.name[:70]}"
    path = REPORT_DIR / f"project_{project_code}_{stamp}.pdf"
    story = [
        Paragraph("PROJECT REPORT", ss["Title2"]),
        Paragraph(
            f"{project.name}<br/>Project Code {project.project_code} · "
            f"Reporting period {latest.report_period}",
            ss["Sub"],
        ),
    ]

    story += [
        Paragraph("Project Information", ss["H"]),
        _table(
            [
                ["Field", "Value"],
                ["Project name", project.name],
                ["Project code", project.project_code],
                ["Executing agency", project.agency],
                ["Ministry / Department", project.ministry],
                ["Sector", f"{project.sector} ({project.sector_origin.value})"],
                ["State", f"{project.state} ({project.state_origin.value})"],
                ["First observed", project.first_seen_period or "UNKNOWN"],
                ["Latest observation", project.last_seen_period or "UNKNOWN"],
            ],
            widths=[45 * mm, 125 * mm],
        ),
    ]

    story += [
        Paragraph("Current Snapshot", ss["H"]),
        _table(
            [
                ["Indicator", "Value"],
                ["Original cost", _fmt(latest.original_cost, " Cr")],
                ["Revised cost", _fmt(latest.revised_cost, " Cr")],
                ["Cumulative expenditure", _fmt(latest.expenditure, " Cr")],
                ["Physical progress", _fmt(latest.physical_progress, " %")],
                ["Financial progress", _fmt(latest.financial_progress, " %")],
                ["Cost escalation", _fmt(latest.cost_escalation_pct, " %")],
                ["Progress divergence", _fmt(latest.progress_divergence, " pp")],
                ["Schedule delay", _fmt(latest.schedule_delay_months, " months", 0)],
                ["Original completion", str(latest.original_completion or "UNKNOWN")],
                ["Revised completion", str(latest.revised_completion or "UNKNOWN")],
                ["Composite risk score", _fmt(latest.risk_score, "", 1)],
                ["Risk level", latest.risk_level.value if latest.risk_level else "UNKNOWN"],
                ["Risk confidence", _fmt((latest.risk_confidence or 0) * 100, " %", 0)],
                ["Data completeness", _fmt(latest.completeness, " %", 1)],
            ],
            widths=[55 * mm, 115 * mm],
            align_right=[1],
        ),
    ]

    hist = [["Period", "Risk", "Level", "Phys %", "Fin %", "Expenditure", "Revised cost"]]
    for s in snaps:
        hist.append([
            s.report_period, _fmt(s.risk_score, "", 1),
            s.risk_level.value if s.risk_level else "UNKNOWN",
            _fmt(s.physical_progress, "", 2), _fmt(s.financial_progress, "", 2),
            _fmt(s.expenditure, "", 2), _fmt(s.revised_cost, "", 2),
        ])
    story += [
        Paragraph("Historical Trend", ss["H"]),
        _table(hist, widths=[22 * mm, 18 * mm, 24 * mm, 20 * mm, 20 * mm, 33 * mm, 33 * mm],
               align_right=[1, 3, 4, 5, 6]),
        Paragraph(
            f"Trend direction: <b>{classify_trend([s.risk_score for s in snaps]).value}</b>",
            ss["Body"],
        ),
    ]

    story.append(Paragraph("Risk Drivers", ss["H"]))
    drivers = latest.risk_drivers or []
    if drivers:
        rows = [["Driver", "Contribution", "Detail"]]
        for d in drivers:
            rows.append([d["label"], f"+{d['contribution']:.1f}",
                         Paragraph(d["detail"], ss["Small"])])
        story.append(_table(rows, widths=[45 * mm, 22 * mm, 103 * mm], align_right=[1]))
    else:
        story.append(Paragraph("No risk drivers are currently active.", ss["Body"]))

    alerts = db.query(Alert).filter_by(project_id=project.id, status=AlertStatus.OPEN).all()
    story.append(Paragraph("Open Early Warnings", ss["H"]))
    if alerts:
        rows = [["Severity", "Warning", "Detail"]]
        for a in alerts:
            rows.append([a.severity.value, a.title, Paragraph(a.description or "", ss["Small"])])
        story.append(_table(rows, widths=[20 * mm, 55 * mm, 95 * mm]))
    else:
        story.append(Paragraph("None.", ss["Body"]))

    ints = db.query(Intervention).filter_by(project_id=project.id).all()
    story.append(Paragraph("Interventions", ss["H"]))
    if ints:
        rows = [["Reference", "Status", "Owner", "Due", "Issue"]]
        for i in ints:
            rows.append([i.reference, i.status.value, i.owner or "—",
                         str(i.due_date or "—"), Paragraph(i.issue[:200], ss["Small"])])
        story.append(_table(rows, widths=[30 * mm, 24 * mm, 26 * mm, 20 * mm, 70 * mm]))
    else:
        story.append(Paragraph("None recorded.", ss["Body"]))

    story.append(Paragraph("Provenance", ss["H"]))
    story.append(
        _table(
            [
                ["Metric", "Source document", "Page", "Confidence"],
                [
                    "Latest snapshot values",
                    latest.source.report_label if latest.source else "UNKNOWN",
                    str(latest.record.page_number if latest.record else "—"),
                    _fmt((latest.record.extraction_confidence or 0) * 100, " %", 1)
                    if latest.record else "UNKNOWN",
                ],
            ],
            widths=[50 * mm, 60 * mm, 20 * mm, 40 * mm],
        )
    )

    story += _sources_block(db, ss)
    story += [
        Spacer(1, 8),
        Paragraph(
            f"Generated {ts.strftime('%d %B %Y, %H:%M UTC')} by {generated_by}. "
            f"Risk engine {latest.risk_engine_version}.<br/>{DISCLAIMER}",
            ss["Small"],
        ),
    ]

    _doc(path, title).build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return path, title


def _national_report(db, report_type, period, stamp, ts, generated_by):
    period = period or analytics.latest_period(db)
    summary = analytics.national_summary(db, period)
    trend = analytics.national_trend(db)
    ss = _styles()

    titles = {
        "MONTHLY_MONITORING": "MONTHLY MONITORING REPORT",
        "EXECUTIVE_BRIEF": "EXECUTIVE BRIEF",
        "RISK": "NATIONAL RISK REPORT",
    }
    title = f"{titles[report_type]} — {period}"
    path = REPORT_DIR / f"{report_type.lower()}_{period}_{stamp}.pdf"

    story = [
        Paragraph(titles[report_type], ss["Title2"]),
        Paragraph(f"Reporting period {period} · Generated {ts.strftime('%d %B %Y')}", ss["Sub"]),
        Paragraph("National Position", ss["H"]),
        _table(
            [
                ["Indicator", "Value"],
                ["Projects monitored", _fmt(summary["total_projects"])],
                ["In HIGH or SEVERE risk band", _fmt(summary["at_risk"])],
                ["Severe risk", _fmt(summary["risk_distribution"]["SEVERE"])],
                ["Running past original completion", 
                 f"{summary['delayed_projects']:,} ({summary['delayed_pct']}%)"],
                ["Projects with cost escalation", _fmt(summary["cost_escalated_projects"])],
                ["Approved cost", _fmt(summary["original_cost_cr"], " Cr")],
                ["Revised cost", _fmt(summary["revised_cost_cr"], " Cr")],
                ["Cost exposure above approved", _fmt(summary["cost_exposure_cr"], " Cr")],
                ["Cumulative expenditure", _fmt(summary["expenditure_cr"], " Cr")],
                ["Mean physical progress", _fmt(summary["mean_physical_progress"], " %")],
                ["Open early warnings", _fmt(summary["open_alerts"])],
                ["Open interventions", _fmt(summary["open_interventions"])],
            ],
            widths=[80 * mm, 90 * mm],
            align_right=[1],
        ),
    ]

    rows = [["Period", "Projects", "Avg risk", "At risk", "Cost exposure", "Expenditure"]]
    for t in trend:
        rows.append([t["period"], _fmt(t["projects"]), _fmt(t["avg_risk_score"], "", 1),
                     _fmt(t["at_risk"]), _fmt(t["cost_exposure_cr"], " Cr"),
                     _fmt(t["expenditure_cr"], " Cr")])
    story += [Paragraph("Trend Across Reporting Periods", ss["H"]),
              _table(rows, widths=[24 * mm, 24 * mm, 24 * mm, 22 * mm, 38 * mm, 38 * mm],
                     align_right=[1, 2, 3, 4, 5])]

    if report_type in {"RISK", "MONTHLY_MONITORING"}:
        top = (
            db.query(ProjectSnapshot, Project)
            .join(Project, Project.id == ProjectSnapshot.project_id)
            .filter(ProjectSnapshot.report_period == period,
                    ProjectSnapshot.supersedes_id.is_(None))
            .order_by(ProjectSnapshot.risk_score.desc())
            .limit(20)
            .all()
        )
        rows = [["Risk", "Level", "Project", "State", "Phys %", "Delay (m)"]]
        for s, p in top:
            rows.append([_fmt(s.risk_score, "", 1),
                         s.risk_level.value if s.risk_level else "UNKNOWN",
                         Paragraph(p.name[:80], ss["Small"]), p.state[:24],
                         _fmt(s.physical_progress, "", 1),
                         _fmt(s.schedule_delay_months, "", 0)])
        story += [PageBreak(), Paragraph("Highest Risk Projects", ss["H"]),
                  _table(rows, widths=[16 * mm, 22 * mm, 72 * mm, 30 * mm, 16 * mm, 18 * mm],
                         align_right=[0, 4, 5])]

    sectors = [g for g in analytics.by_sector(db, period) if g["reliable"]][:12]
    if sectors:
        rows = [["Sector", "Projects", "Avg risk", "Cost exposure", "Avg progress"]]
        for g in sectors:
            rows.append([g["key"], _fmt(g["projects"]), _fmt(g["avg_risk_score"], "", 1),
                         _fmt(g["cost_exposure_cr"], " Cr"),
                         _fmt(g["avg_physical_progress"], " %")])
        story += [Paragraph("Sector Position", ss["H"]),
                  _table(rows, widths=[52 * mm, 22 * mm, 22 * mm, 40 * mm, 28 * mm],
                         align_right=[1, 2, 3, 4])]

    story += _sources_block(db, ss)
    story += [
        Spacer(1, 8),
        Paragraph(
            f"Generated {ts.strftime('%d %B %Y, %H:%M UTC')} by {generated_by}.<br/>{DISCLAIMER}",
            ss["Small"],
        ),
    ]

    _doc(path, title).build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)
    return path, title
