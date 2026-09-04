"""PAIMANA AI — application entrypoint."""
from __future__ import annotations

import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import settings
from .db import init_db
from .routers import chat, monitor, platform, projects

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("paimana")

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    log.info("PAIMANA AI started in %s mode", settings.environment)
    if not settings.llm_enabled:
        log.info(
            "No LLM API key configured. The assistant will serve deterministic answers "
            "generated directly from the database."
        )
    yield


app = FastAPI(
    lifespan=lifespan,
    title="PAIMANA AI",
    description=(
        "Project Assessment, Intelligence, Monitoring & Analytics Network for "
        "Accelerated Infrastructure. An analytical prototype — not an official "
        "Government of India portal."
    ),
    version="2.0.0",
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None,
)

# Explicit origins only. Never "*" alongside credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail},
                        headers=getattr(exc, "headers", None))


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "The request could not be processed.",
            "errors": [
                {"field": ".".join(str(p) for p in e.get("loc", [])[1:]), "message": e.get("msg")}
                for e in exc.errors()
            ],
        },
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    """Never leak a stack trace or internal path to the client."""
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. The incident has been logged."},
    )


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "environment": settings.environment,
        "llm_enabled": settings.llm_enabled,
        "version": app.version,
        "disclaimer": "Analytical prototype. Not an official Government of India portal.",
    }


app.include_router(projects.router)
app.include_router(monitor.router)
app.include_router(platform.router)
app.include_router(chat.router)

# --- static frontend --------------------------------------------------------
if settings.frontend_dir.exists():
    app.mount(
        "/static",
        StaticFiles(directory=settings.frontend_dir / "static"),
        name="static",
    )

    PAGES = {
        "": "index.html",
        "projects": "projects.html",
        "project": "project.html",
        "monitor": "monitor.html",
        "warnings": "warnings.html",
        "interventions": "interventions.html",
        "analytics": "analytics.html",
        "simulator": "simulator.html",
        "assistant": "assistant.html",
        "data": "data.html",
        "reports": "reports.html",
    }

    @app.get("/{page:path}", include_in_schema=False)
    def serve_page(page: str):
        filename = PAGES.get(page.strip("/"), None)
        if filename is None:
            filename = "404.html"
        target = settings.frontend_dir / filename
        if not target.exists():
            target = settings.frontend_dir / "index.html"
        return FileResponse(target)
