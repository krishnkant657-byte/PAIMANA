"""Agency -> Ministry / Sector resolution.

IMPORTANT — this is the module that replaces the old `random.choice()` block.

The Flash Report's "All Ongoing Projects" table carries the executing agency but
not the administrative ministry or the sector. Rather than guessing, we resolve
them through an explicit, human-reviewable token table below. The ministry and
sector vocabulary is taken verbatim from Table 1 (Ministry-wise Ongoing
Projects) of the same report, so the values we assign are values the source
document itself uses.

Anything that does not match is returned as UNKNOWN with origin=UNKNOWN. It is
never filled in with a plausible-looking value.
"""
from __future__ import annotations

import re

UNKNOWN = "UNKNOWN"

# (regex over the agency string, ministry, sector)
# Ordered: the first match wins, so put specific patterns before general ones.
AGENCY_RULES: list[tuple[str, str, str]] = [
    # --- Coal -------------------------------------------------------------
    (r"\b(CIL|COAL INDIA)\b", "Ministry of Coal", "Coal"),
    (r"\b(ECL|BCCL|CCL|NCL|WCL|SECL|MCL|NEC)\b", "Ministry of Coal", "Coal"),
    (r"EASTERN COAL|BHARAT COKING|CENTRAL COALFIELDS|NORTHERN COALFIELDS",
     "Ministry of Coal", "Coal"),
    (r"WESTERN COALFIELDS|SOUTH EASTERN COALFIELDS|MAHANADI COALFIELDS",
     "Ministry of Coal", "Coal"),
    (r"SINGARENI", "Ministry of Coal", "Coal"),

    # --- Petroleum & Natural Gas -----------------------------------------
    (r"\b(IOCL|INDIAN OIL)\b", "Ministry of Petroleum & Natural Gas", "Oil & Gas"),
    (r"\b(ONGC|OIL AND NATURAL GAS)\b", "Ministry of Petroleum & Natural Gas", "Oil & Gas"),
    (r"\b(GAIL)\b", "Ministry of Petroleum & Natural Gas", "Oil & Gas"),
    (r"\b(HPCL|HINDUSTAN PETROLEUM)\b", "Ministry of Petroleum & Natural Gas", "Oil & Gas"),
    (r"\b(BPCL|BHARAT PETROLEUM)\b", "Ministry of Petroleum & Natural Gas", "Oil & Gas"),
    (r"\b(CPCL|NRL|MRPL|OIL INDIA|NUMALIGARH)\b",
     "Ministry of Petroleum & Natural Gas", "Oil & Gas"),
    (r"PETRONET|INDRADHANUSH|IGGL", "Ministry of Petroleum & Natural Gas", "Oil & Gas"),

    # --- Power ------------------------------------------------------------
    (r"\b(NTPC)\b", "Ministry of Power", "Electricity Generation"),
    (r"\b(POWERGRID|POWER GRID|PGCIL)\b", "Ministry of Power", "Electricity Transmission"),
    (r"\b(NHPC|SJVN|THDC|NEEPCO)\b", "Ministry of Power", "Electricity Generation"),
    (r"DAMODAR VALLEY|\bDVC\b", "Ministry of Power", "Electricity Generation"),
    (r"TRANSMISSION LIMITED|TRANSMISSION LTD", "Ministry of Power", "Electricity Transmission"),

    # --- Railways ---------------------------------------------------------
    (r"\b(RVNL|IRCON|DFCCIL|RLDA|NHSRCL)\b", "Ministry of Railways", "Railways"),
    (r"RAIL VIKAS|DEDICATED FREIGHT|HIGH SPEED RAIL", "Ministry of Railways", "Railways"),
    (r"\bRAILWAY[S]?\b", "Ministry of Railways", "Railways"),

    # --- Roads ------------------------------------------------------------
    (r"\b(NHAI|NHIDCL)\b", "Ministry of Road Transport & Highways", "Roads & Highways"),
    (r"NATIONAL HIGHWAYS", "Ministry of Road Transport & Highways", "Roads & Highways"),

    # --- Urban / Metro ----------------------------------------------------
    (r"METRO RAIL|\bDMRC\b|\bMMRDA\b|\bCMRL\b|\bBMRCL\b|\bMAHA-METRO\b|\bNCRTC\b",
     "Ministry of Housing & Urban Affairs", "Urban Public Transport"),
    (r"URBAN DEVELOPMENT|MUNICIPAL CORPORATION",
     "Ministry of Housing & Urban Affairs", "Urban Public Transport"),

    # --- Ports / Shipping / Waterways ------------------------------------
    (r"PORT TRUST|PORT AUTHORITY|\bIWAI\b|SAGARMALA|SHIPYARD|COCHIN SHIPYARD",
     "Ministry of Ports, Shipping and Waterways", "Shipping"),

    # --- Civil aviation ---------------------------------------------------
    (r"\b(AAI)\b|AIRPORTS AUTHORITY", "Ministry of Civil Aviation",
     "Aviation & Aviation Infrastructure"),

    # --- Telecom ----------------------------------------------------------
    (r"\b(BSNL|MTNL|BBNL)\b|BHARAT BROADBAND", "Department of Telecommunications",
     "Telecommunication"),

    # --- Health -----------------------------------------------------------
    (r"\b(AIIMS)\b|ALL INDIA INSTITUTE OF MEDICAL", "Ministry of Health & Family Welfare",
     "Healthcare"),
    (r"\b(ESIC)\b", "Ministry of Labour and Employment", "Healthcare"),

    # --- Education --------------------------------------------------------
    (r"\b(IIT|IIM|NIT|IISER|IIIT)\b|INDIAN INSTITUTE OF TECHNOLOGY",
     "Department of Higher Education", "Education"),
    (r"CENTRAL UNIVERSITY|NAVODAYA", "Department of Higher Education", "Education"),

    # --- Water resources --------------------------------------------------
    (r"WATER RESOURCES|IRRIGATION|\bWAPCOS\b|\bNPCC\b",
     "Department of Water Resources, River Development & Ganga Rejuvenation",
     "Water Resources"),

    # --- Steel / mining ---------------------------------------------------
    (r"\b(SAIL|NMDC|RINL|HCL|MOIL)\b|STEEL AUTHORITY", "Ministry of Steel", "Metals & Mining"),

    # --- Atomic energy / space -------------------------------------------
    (r"\b(NPCIL|BARC)\b|NUCLEAR POWER", "Department of Atomic Energy", "Electricity Generation"),
    (r"\b(ISRO|VSSC|SDSC)\b", "Department of Space", "Space"),

    # --- Fertilizers / chemicals -----------------------------------------
    (r"\b(RCF|NFL|FCIL|HURL|BVFCL)\b|FERTILIZER", "Department of Fertilizers", "Chemicals"),

    # --- Agencies recorded as the administrative ministry itself ----------
    # A sizeable share of rows name the ministry in the agency field rather
    # than an executing body. These are matched verbatim (whitespace and
    # punctuation are normalised first) rather than inferred.
    (r"\bMORTH\b|ROAD TRANSPORT (AND|&) HIGHWAYS",
     "Ministry of Road Transport & Highways", "Roads & Highways"),
    (r"PETROLEUM\s*(AND|&)?\s*NATURAL\s*GAS", "Ministry of Petroleum & Natural Gas", "Oil & Gas"),
    (r"MEDICAL EDUCATION|HEALTH (AND|&) FAMILY WELFARE|\bPMSSY\b",
     "Ministry of Health & Family Welfare", "Healthcare"),
    (r"NATIONAL IMPORTANCE", "Department of Higher Education", "Education"),
    (r"TELECOMMUNICATION|\bDOT\b", "Department of Telecommunications", "Telecommunication"),
    (r"MINISTRY\s*OF\s*COAL|\bMOCOAL\b", "Ministry of Coal", "Coal"),
    (r"HOUSING (AND|&) URBAN AFFAIRS", "Ministry of Housing & Urban Affairs", "Real Estate"),
    (r"CLEAN GANGA|\bNMCG\b",
     "Department of Water Resources, River Development & Ganga Rejuvenation",
     "Water Resources"),
    (r"INDUSTRIAL CORRIDOR|\bNICDC\b",
     "Department for Promotion of Industry & Internal Trade", "Logistics Infrastructure"),
    (r"\bNLCIL?\b|NLC INDIA", "Ministry of Coal", "Electricity Generation"),
    (r"\bNALCO\b|NATIONAL ALUMINIUM", "Ministry of Mines", "Metals & Mining"),
    (r"NORTH EASTERN ELECTRIC POWER", "Ministry of Power", "Electricity Generation"),
    (r"\bK-?RIDE\b|RAIL INFRASTRUCTURE DEVELOPMENT", "Ministry of Railways", "Railways"),
    (r"RAIL LAND DEVELOPMENT", "Ministry of Railways", "Railways"),

    # Indian Railways zonal and construction-office codes. These appear as
    # bare abbreviations (e.g. "SECR", "PCE/WR", "CAO/C/NCR") and are all
    # Ministry of Railways units.
    (r"\b(CR|ER|ECR|ECOR|NR|NCR|NER|NFR|NWR|SR|SCR|SER|SECR|SWR|WR|WCR)\b",
     "Ministry of Railways", "Railways"),
    (r"\b(PCE|CAO|GM|DRM|CPM)\s*/", "Ministry of Railways", "Railways"),
]

_COMPILED = [(re.compile(p, re.IGNORECASE), m, s) for p, m, s in AGENCY_RULES]


def _normalise(agency: str) -> str:
    """Upper-case and space out run-together names.

    The source PDF sometimes loses spaces during text extraction, producing
    values like 'MinistryofPetroleumNaturalGas'. We re-insert boundaries at
    lower-to-upper transitions so those rows resolve to the same ministry as
    their correctly-spaced siblings, instead of falling through to UNKNOWN.
    """
    spaced = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", agency)
    return re.sub(r"\s+", " ", spaced).upper().strip()


def resolve_agency(agency: str | None) -> tuple[str, str, bool]:
    """Return (ministry, sector, matched).

    `matched` is False when no rule fired — the caller must then store UNKNOWN
    and set the provenance origin to UNKNOWN rather than inventing a value.
    """
    if not agency:
        return UNKNOWN, UNKNOWN, False
    # Try the literal text first (preserves acronyms like MoRTH), then the
    # de-run-together form (recovers 'MinistryofPetroleumNaturalGas').
    candidates = (re.sub(r"\s+", " ", agency).upper().strip(), _normalise(agency))
    for pattern, ministry, sector in _COMPILED:
        if any(pattern.search(text) for text in candidates):
            return ministry, sector, True
    return UNKNOWN, UNKNOWN, False


def coverage_report(agencies: list[str]) -> dict:
    """Diagnostic: how much of the corpus the mapping actually covers."""
    total = len(agencies)
    matched = sum(1 for a in agencies if resolve_agency(a)[2])
    unmatched = sorted({a for a in agencies if not resolve_agency(a)[2]})
    return {
        "total": total,
        "matched": matched,
        "unmatched": total - matched,
        "coverage_pct": round(100.0 * matched / total, 2) if total else 0.0,
        "unmatched_examples": unmatched[:40],
    }
