# data/

`paimana.db` ships pre-built from four Flash Reports (April–July 2026):
**2,074 projects, 7,590 immutable snapshots, 1,282 early warnings**, three demo
accounts and three demonstration interventions. You can run the platform
immediately without re-ingesting anything.

The source PDFs are not bundled, to keep the archive small. To rebuild from
scratch, drop the four `FlashReport_*.pdf` files into this directory and run:

    python scripts/ingest_reports.py --reset
    python scripts/seed_users.py
    python scripts/seed_demo_interventions.py

Ingestion takes roughly 50 seconds per report.

`uploads/` and `reports/` are created at runtime and are gitignored.
