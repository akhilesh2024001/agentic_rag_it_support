
import logging

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router
from app.core.config import get_settings, BASE_DIR
from app.core.logging import configure_logging
from app.services.audit import init_db


# --------------------------------------------------
# APPLICATION INITIALIZATION
# --------------------------------------------------

configure_logging()

logger = logging.getLogger(__name__)

settings = get_settings()


# --------------------------------------------------
# CREATE REQUIRED DIRECTORIES
# --------------------------------------------------

(BASE_DIR / "data").mkdir(parents=True, exist_ok=True)
(BASE_DIR / "uploads").mkdir(parents=True, exist_ok=True)

static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"

if not static_dir.exists():
    raise RuntimeError("Static directory is missing")

if not templates_dir.exists():
    raise RuntimeError("Templates directory is missing")


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

try:
    init_db()
    logger.info("Audit database initialized successfully")

except Exception:
    logger.exception("Audit database initialization failed")
    raise


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
)


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

app.include_router(router)


# --------------------------------------------------
# STATIC FILES
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=str(static_dir)),
    name="static",
)


# --------------------------------------------------
# HTML TEMPLATES
# --------------------------------------------------

templates = Jinja2Templates(
    directory=str(templates_dir)
)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "app_name": settings.app_name,
        },
    )


# --------------------------------------------------
# STARTUP AND SHUTDOWN EVENTS
# --------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info(
        "%s started successfully",
        settings.app_name,
    )


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down")
