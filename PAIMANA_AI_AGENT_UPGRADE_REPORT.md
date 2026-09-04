# PAIMANA AI — Agent Upgrade Report

Implemented directly in the provided codebase. Verified by running the test
suite, starting the application, and driving it in a real Chromium browser.

**Final state:** 242 tests passing, 0 failing. Application starts, all 10 pages
serve 200, complete user journey verified in-browser with zero page or console
errors.

---

## 1. Existing architecture discovered

| Layer | What is actually there |
|---|---|
| Backend | FastAPI 0.141 + SQLAlchemy 2.0 ORM, Uvicorn |
| Database | SQLite (`data/paimana.db`), `create_all` schema management, no migration tool |
| Frontend | Static HTML served by a catch-all route. One shared `static/js/paimana.js` runtime, one `static/css/paimana.css` design system. **No build step, no npm, no framework** |
| Auth | Bearer tokens signed with `itsdangerous`, PBKDF2-SHA256 passwords, `require_analyst` / `require_admin` role dependencies |
| Assistant | `services/assistant.py` — regex intent detection → deterministic SQL planner → `narrate()` draft → optional Gemini rephrase |
| Risk | `services/risk_engine.py` — fully deterministic, 5 weighted drivers summing to 100, no ML (deliberately, and the docstring explains why) |
| Uploads | `/api/ingest/upload` — PDF only, analyst-gated, feeds `services/ingestion.py` |
| Tests | 3 files, pytest |

**Search results.** `assistant`/`chat`/`LLM`/`AI` → `services/assistant.py`,
`routers/platform.py`, `frontend/assistant.html`. `upload`/`file`/`PDF` →
`ingestion.py`, `security.py` (`validate_pdf_bytes`), `platform.py`. `risk`/
`delay`/`intervention` → `risk_engine.py`, `warnings.py`, `analytics.py`,
`scenario.py`. **No pre-existing chat attachment, image, Excel, or CSV handling
anywhere in the repository.**

### Five problems found before any code was written

1. **The shipped database is empty.** `data/paimana.db` is 4 KB with **zero
   tables**, despite `data/README.md` claiming 2,074 projects and 7,590
   snapshots. The `scripts/` directory that README instructs you to run is also
   absent from the archive.
2. **`pytest` failed at collection on a fresh clone.** `tests/test_api.py`
   queried the database at *import* time to build its skip marker, before any
   `init_db()` had run — an `OperationalError`, not a skip, contradicting its own
   docstring.
3. **Conversation was stateless.** No follow-up context of any kind.
4. **Off-topic and casual routing depended entirely on the LLM.** With no
   `GEMINI_API_KEY` configured (none is), "what is Python?" fell through to
   *"I don't have sufficient verified data."* — the exact failure the brief asks
   to eliminate, and not fixable by prompt-tuning.
5. **No attachment capability** on the assistant surface at all.

---

## 2. Architecture changes

The new agent sits **above** the existing verified assistant. `plan_and_execute`
and `narrate` are still the only things that produce a project figure.

```
user message (+ attachments)
        │
        ▼
  reference resolution        ← deterministic, before any query runs
  ("it", "the first one")
        │
        ▼
  route classification        ← deterministic; LLM only breaks ties
        │
   ┌────┼──────────┬────────────────┬──────────────┐
   ▼    ▼          ▼                ▼              ▼
PAIMANA  FILE   FILE × PAIMANA   GENERAL AI    OFF-TOPIC
planner analysis  cross-check    (needs LLM)   (3 levels)
   │      │          │                │              │
   └──────┴──────────┴────────────────┴──────────────┘
                     ▼
        answer + explicit source label
```

Two invariants are enforced structurally, not by prompting:

- **Project figures come from SQL.** The model may rephrase a computed result;
  it can never produce one. Where the model is absent, verified answers are
  unaffected.
- **Every statistic about a file is computed by pandas**, not by the model. A
  60-row spreadsheet never reaches the LLM; its computed profile does.

### Design decision: deterministic-first routing

The original router asked the LLM to classify every unresolved message. Because
no API key is configured, that meant **greetings and general questions were
being refused as "insufficient verified data."**

Routing is now deterministic first. The consequence is that removing the API key
degrades *only* general conversation — which genuinely requires a model — while
PAIMANA answers, file analysis, data-quality assessment, cross-checking, context
tracking, and off-topic handling all continue to work fully. That is verified:
the entire browser journey in §14 ran with `llm_enabled: false`.

---

## 3. Files modified

| File | Change |
|---|---|
| `backend/app/models.py` | **+3 tables** (`Conversation`, `ChatMessage`, `ChatAttachment`). No existing table, column or relationship altered |
| `backend/app/config.py` | Added chat-attachment settings block. Existing ingestion settings untouched |
| `backend/app/schemas.py` | Added `ChatMessageRequest` |
| `backend/app/security.py` | Added `safe_attachment_name`, `validate_attachment_bytes`, `resolve_within`. **PDF ingestion path untouched** |
| `backend/app/main.py` | Two lines: import and `include_router(chat.router)` |
| `backend/app/services/assistant.py` | Fixed a crash; added `find_projects`, `compare_projects`, `detect_aspect`, `_narrate_project`, `_narrate_project_comparison`. All existing behaviour preserved |
| `backend/tests/test_api.py` | Fixed the collection-time crash |
| `frontend/assistant.html` | Rewritten (chat panel + attachments) |
| `frontend/static/css/paimana.css` | **Appended only.** No existing rule modified |

## 4. New files

| File | Lines | Purpose |
|---|---|---|
| `services/files/detect.py` | 255 | Extension + MIME + magic-byte triangulation |
| `services/files/extractors.py` | 795 | Per-format extraction; never raises |
| `services/files/tabular.py` | 406 | Deterministic profiling + quality scoring |
| `services/files/sanitize.py` | 118 | Prompt-injection neutralisation, truncation |
| `services/files/analysis.py` | 381 | Orchestrator; normalised result |
| `services/agent.py` | 791 | Hybrid router |
| `services/conversation.py` | 336 | Context + reference resolution |
| `services/crosscheck.py` | 348 | File ⟷ PAIMANA comparison |
| `routers/chat.py` | 364 | Chat API |
| `scripts/dev_seed.py` | 295 | Demo dataset (see §16) |
| 5 test files + conftest | 1,254 | See §13 |

**~3,800 lines of implementation, ~1,900 lines of tests.**

## 5. APIs added

All additive. `POST /api/assistant/ask` is unchanged and still serves the
stateless verified path the Digital Twin panel depends on (regression-tested).

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/chat/capabilities` | Accepted formats, limits, LLM status |
| `POST` | `/api/chat/conversations` | Start a conversation |
| `GET` | `/api/chat/conversations/{id}` | Replay history, context, attachments |
| `POST` | `/api/chat/attachments` | Upload + analyse one file |
| `DELETE` | `/api/chat/attachments/{id}` | Remove |
| `POST` | `/api/chat/message` | Send a turn |

## 6. Database changes

Three new tables, created by the existing `create_all`. **No existing table
altered, so no migration is required and no existing query is affected.**
`ChatAttachment.analysis` caches the analysis so follow-up questions never
re-parse a file.

## 7. Supported file types

**Fully analysed** — PDF (text + tables + metadata + OCR fallback), DOCX, PPTX,
XLSX/XLSM, CSV, TSV, JSON, XML, TXT, MD, PNG, JPEG, WEBP, GIF, BMP.

**Inspected safely** — ZIP (listed, never extracted).

**Identified but not parsed, and said so** — `.xls` (legacy BIFF), `.doc`/`.ppt`/
`.msg` (OLE), unparseable `.drp`.

Detection is by **magic bytes first**. A PNG renamed `.pdf` is analysed as a PNG
and the mismatch is surfaced to the user (tested).

## 8. DRP handling

`.drp` is not one format — it is used by several unrelated products. The
analyser identifies the actual container and never executes anything:

- **ZIP-backed** → contents listed, readable entries named, schema not guessed.
- **JSON/XML-backed** → genuinely parsed; values are reported, but PAIMANA field
  meanings are **not** assumed.
- **Binary/unrecognised** → reported as `IDENTIFIED`, not `ANALYSED`, with the
  export handoff message.

> A bug found during browser testing: the card originally read **ANALYSED** over
> a binary DRP that nothing had been extracted from, because the extractor
> returns `ok=True` when *identification* succeeds. That is precisely the
> dishonesty Step 14 warns against. Status now reflects whether project
> information was actually recovered.

## 9. Conversation routing

Ordered by specificity, first confident match wins, so the decision is always
explainable (`route_reason` is returned in the API response):

1. Small talk / capability question
2. Attachment + comparison verb + PAIMANA reference → **cross-check**
3. Attachment present → **file analysis**
4. Far off-topic (and no project vocabulary) → **gentle redirect**
5. Project vocabulary → **verified PAIMANA planner**
6. Back-reference with context → **PAIMANA follow-up**
7. Otherwise → **general**

A PAIMANA query the planner cannot resolve falls through to a general answer
where sensible, explicitly labelled — it never fabricates project data.

## 10. Context management

Compact structured state, **not** a replayed transcript: current project,
recently listed projects in display order, active state/sector/period filters,
attachment references, off-topic strike count.

References are resolved **before** any query runs. This is a grounding decision,
not an optimisation: if the model chose which project "it" meant, a wrong guess
would produce a confidently-worded answer about the wrong project. Instead the
reference either resolves to a specific code, or the assistant asks.

Verified working: `which projects are high risk?` → `why is the first one
risky?` → `how delayed is it?` → `compare it with the second one` → `what about
its cost overrun?`

## 11. Off-topic behaviour

Three levels, and general questions are **never** blocked:

- **Level 1** — reasonable general question → answered normally, labelled `GENERAL AI`.
- **Level 2** — first far-off-topic request → gentle nudge, still offers to help.
- **Level 3** — repeated → escalates to suggesting a general-purpose assistant.

A single on-topic turn **clears the strike counter** — users should not be
punished for one stray question ten messages ago. No aggressive blocking, no
error language (tested).

## 12. Security

| Control | Implementation |
|---|---|
| Path traversal | Stored names are **server-generated**; traversal is impossible by construction, not by filtering. `resolve_within()` re-checks any DB-value-to-path conversion |
| Magic-byte validation | Signature is authoritative over extension and MIME |
| Size limits | 25 MB per attachment, 5 per message, capped per conversation |
| Prompt injection | Neutralised with a **visible marker** (so the user can still see what the document said) + delimited untrusted blocks stating the content cannot instruct. Zero-width/bidi characters stripped |
| ZIP bombs | Compression ratio and uncompressed size checked **before** any read; expansion refused above 100:1 or 500 MB |
| Archive traversal | Entries escaping the root are refused and excluded from the listing |
| XML entity expansion | `defusedxml` only; the stdlib parser is never used on uploads |
| Execution | Nothing from an upload is executed, evaluated or resolved. Archives are **listed**, never extracted |
| Error handling | No stack trace or filesystem path reaches the client (tested across endpoints) |
| Rate limiting | Separate buckets for messages and uploads |

## 13. Test results

```
242 passed, 0 failed
```

| File | Tests | Covers |
|---|---|---|
| `test_core.py`, `test_api.py`, `test_assistant_chat.py` | 79 | Pre-existing (was 59 passed / 20 **skipped**; now all run) |
| `test_files.py` | 38 | Detection, extraction, quality, DRP, file security |
| `test_agent_routing.py` | 54 | Routing, off-topic escalation, references, context |
| `test_chat_api.py` | 39 | Endpoints, grounding, attachments, upload security |
| `test_crosscheck.py` | 29 | Column mapping, discrepancy detection, neutrality |
| `test_rate_limiting.py` | 3 | Rate limits |

Notable assertions include: every project code in an answer exists in the
verified result; a quality verdict is never a bare word; neither source is ever
declared correct in a cross-check; a clean white-background screenshot is not
flagged as "washed out"; an out-of-range ordinal asks rather than guesses.

### Bugs found by running the code

Each of these was caught by execution, not review:

1. **`narrate()` `KeyError: 'result'`** — *pre-existing*. It dispatched on intent
   alone, so a project-scoped query carrying intent `COMPARE` fell into the
   state/sector branch and crashed. Now dispatches on result **shape**.
2. **Single-project answers wiped the ordinal list** — after listing ten
   projects, the follow-up replaced `recent_projects` with one, so "the second
   one" had nothing to count against. Fixed with merge semantics.
3. **Image quality gave false warnings** — global brightness/contrast is
   meaningless for a document screenshot (mostly white), so every clean report
   scored as "washed out, low contrast". Rewritten to classify document-like vs
   photographic and measure blur over the **ink region**.
4. **DRP reported ANALYSED when nothing was extracted** (§8).
5. **Ordinal anaphora only covered first/second/third** — "the fifth one"
   wasn't recognised as a back-reference, so the planner silently answered a
   *different question*.
6. **"national situation" wasn't PAIMANA vocabulary** — a documented example
   query routed to general.
7. **Name-column detection grabbed the code column** — `"project code"` starts
   with the loose hint `"project"`.
8. **Rate limiter bled across test modules** — shared in-process state made
   unrelated tests fail with 429s. Isolated in `conftest.py`; rate limiting is
   now tested deliberately in one place.
9. **Greeting off-by-one**, **seed had no HIGH/SEVERE projects**, **test
   collection crash**, **`files` package shadowed its own `detect` submodule**.

## 14. Browser validation

Chromium via Playwright, against the running application at
`http://127.0.0.1:8000`.

| # | Step | Result |
|---|---|---|
| 1 | Page loads, chrome/nav intact | ✅ |
| 2 | `hello` | ✅ Natural greeting, `PAIMANA ASSISTANT` |
| 3 | `what is Python?` | ✅ Honest "needs a language model", **not** refused |
| 4 | `which projects are high risk?` | ✅ `PAIMANA VERIFIED DATA` badge |
| 5 | `why is the first one risky?` | ✅ Resolved from previous answer |
| 6 | `give me a recipe for pasta` | ✅ Gentle redirect |
| 7 | Upload image | ✅ Thumbnail, progress, `ANALYSED` |
| 8 | Ask about image | ✅ OCR reported `PARTIALLY_READABLE` at 69% confidence — did **not** assert unreadable values |
| 9 | Upload corrupted PDF | ✅ `FAILED` card + recovery advice, no stack trace |
| 10 | Upload binary `.drp` | ✅ `IDENTIFIED` card, honest message |
| 11 | Upload XLSX, ask "good or bad?" | ✅ `NEEDS ATTENTION` + itemised evidence |
| 12 | Multi-file upload | ✅ Both cards, per-file quality |
| 13 | Cross-check vs PAIMANA | ✅ 8 discrepancies found, 1 unmatched row reported, 4 provenance labels |
| 14 | Remove attachment | ✅ Tray clears, server DELETE fires |
| 15 | Mobile 390 px | ✅ Single column, no horizontal overflow, attach button visible |
| 16 | Console / Network | ✅ **Zero** page errors, zero server non-200s |

All 10 existing pages return 200 (regression check).

> The only console entry observed is a `403` from Chromium's own blocked
> telemetry endpoint — the application logged **no** non-200 responses.

**Cross-check result against a fixture with deliberately injected drift:** all 8
discrepancies detected at exactly the injected magnitude (+7 pp progress, +5%
cost, +3 months), the unmatched row reported rather than dropped, and no claim
made about which source is correct.

## 15. Build result

No frontend build exists — the project serves static HTML with no npm or
bundler. "Build succeeds" therefore means the application starts, all pages
serve, and the JS runs without error, all of which is verified above. No build
tooling was introduced, per Step 26.

**Dependencies added: zero.** Every library used (`pdfplumber`, `pandas`,
`openpyxl`, `python-docx`, `python-pptx`, `Pillow`, `defusedxml`, `pytesseract`,
`pypdfium2`) was already present. Optional ones are imported lazily, so a
missing library degrades one format with an honest message rather than breaking
uploads.

## 16. Known limitations

1. **No language model is configured**, so general conversation reports that
   honestly rather than answering. Everything else works fully. This is the
   deployment's state, not a defect — but the LLM-enabled prose paths (rather
   than the deterministic fallbacks) are therefore **untested against a live
   model**.
2. **The demo dataset is synthetic.** `scripts/dev_seed.py` exists only because
   the shipped database is empty and the ingestion scripts are absent. Every row
   is flagged `is_demo=True` and attributed to a source named
   `DEMO_SeedData_*.synthetic` with the publisher string *"NOT a published Flash
   Report"*. It is a development fixture; real ingestion via
   `/api/ingest/upload` is unchanged and untouched.
3. **OCR quality.** Tesseract is available and works, but confidence on small
   anti-aliased text runs ~70%, so values are correctly reported as
   `PARTIALLY_READABLE`. The system is honest about this rather than accurate.
4. **PDF analysis caps at 60 pages** (12 for OCR) to bound processing time.
   Truncation is disclosed in the answer.
5. **Cross-check compares up to 1,000 rows** and matches on project code or
   exact/prefix name. Fuzzy name matching is not implemented — an unmatched row
   is reported as unmatched rather than guessed at.
6. **Rate limiting is in-process.** Fine for one worker; a multi-worker
   deployment needs Redis. This was already true of the original code.
7. **Attachment retention is configured but not swept.** `attachment_retention_hours`
   is respected by the setting but no scheduled cleanup job runs — needs a cron
   or startup task in a real deployment.
8. **Scanned-PDF OCR is untested against a real scan.** The code path exists and
   degrades honestly, but no genuine scanned document was available as a fixture.

## 17. Not implemented, and why

- **Fuzzy/ML project-name matching in cross-check** — a wrong match produces a
  confidently-wrong discrepancy report, which is worse than reporting the row as
  unmatched. Deliberately omitted.
- **Server-side virus scanning** — needs ClamAV or equivalent, outside this
  environment. Uploads are never executed, which is the control that matters here.
- **Streaming responses** — the existing UI is request/response and Step 21
  explicitly rules out fake typing animations.
- **Automatic project-code extraction from PDF/image text for cross-check** —
  only tabular files are cross-checked. Regex-extracting codes from OCR'd prose
  would tie verified PAIMANA data to low-confidence character recognition.
- **`.doc`/`.ppt`/`.xls` parsing** — would require LibreOffice or `antiword`.
  Reported honestly as identified-not-supported with conversion advice.

---

## Running it

```bash
cd paimana-ai/backend
pip install fastapi "uvicorn[standard]" sqlalchemy python-multipart python-dotenv itsdangerous requests
cd .. && python scripts/dev_seed.py          # only if the database is empty
cd backend && python -m pytest -q            # 242 passed
python -m uvicorn app.main:app --reload      # then open http://127.0.0.1:8000/assistant
```

To enable general conversation, set `GEMINI_API_KEY` in `.env`. Nothing else
changes: verified PAIMANA answers and all file analysis are computed by the
platform either way.
