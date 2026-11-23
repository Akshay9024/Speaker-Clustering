"""Starlette upload service with warm models and serialised background inference."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Literal

from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.formparsers import MultiPartException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.types import Message

from app.infer import AudioTooLong, CountResult, InvalidAudio, SpeakerCounter, ffmpeg_executable

_LOG = logging.getLogger(__name__)
_ROOT = Path(__file__).resolve().parent.parent
_STATIC = Path(__file__).parent / "static"
_UPLOAD_CHUNK = 1024 * 1024
_BODY_OVERHEAD = 64 * 1024
_SIZE_ERROR = "The upload exceeds the server's file-size limit. Choose a smaller file."


@dataclass(frozen=True)
class WebSettings:
    config_path: Path = _ROOT / "configs/voxconverse.yaml"
    max_upload_bytes: int = 100 * 1024 * 1024
    max_duration_sec: float = 3600.0
    decode_timeout_sec: float = 120.0
    max_pending_jobs: int = 8
    retained_jobs: int = 128
    job_ttl_sec: float = 3600.0

    def __post_init__(self) -> None:
        for name in ("max_upload_bytes", "max_pending_jobs", "retained_jobs"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("decode_timeout_sec", "job_ttl_sec"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.max_duration_sec) or self.max_duration_sec < 0:
            raise ValueError("max_duration_sec must be finite and nonnegative")

    @classmethod
    def from_env(cls) -> WebSettings:
        """Read service limits without changing offline experiment parameters."""
        config = Path(os.environ.get("SPK_CONFIG", "configs/voxconverse.yaml"))
        return cls(
            config_path=config if config.is_absolute() else _ROOT / config,
            max_upload_bytes=int(os.environ.get("SPK_MAX_UPLOAD_MB", "100")) * 1024 * 1024,
            max_duration_sec=float(os.environ.get("SPK_MAX_DURATION_SEC", "3600")),
            decode_timeout_sec=float(os.environ.get("SPK_DECODE_TIMEOUT_SEC", "120")),
            max_pending_jobs=int(os.environ.get("SPK_MAX_PENDING_JOBS", "8")),
            retained_jobs=int(os.environ.get("SPK_RETAINED_JOBS", "128")),
            job_ttl_sec=float(os.environ.get("SPK_JOB_TTL_SEC", "3600")),
        )


@dataclass
class Job:
    status: Literal["queued", "running", "done", "error"] = "queued"
    updated_at: float = field(default_factory=time.monotonic)
    result: CountResult | None = None
    error: str | None = None
    code: str | None = None


def _prune_jobs(jobs: dict[str, Job], settings: WebSettings) -> None:
    """Expire terminal jobs; callers hold the application's job lock."""
    now = time.monotonic()
    terminal = sorted(
        ((key, job) for key, job in jobs.items() if job.status in ("done", "error")),
        key=lambda item: item[1].updated_at,
    )
    for key, job in terminal:
        if now - job.updated_at > settings.job_ttl_sec or len(jobs) > settings.retained_jobs:
            del jobs[key]


def _load_models(settings: WebSettings) -> SpeakerCounter:
    counter = SpeakerCounter(
        config_path=settings.config_path,
        token=os.environ.get("HF_TOKEN"),
        max_duration_sec=settings.max_duration_sec,
        decode_timeout_sec=settings.decode_timeout_sec,
    )
    counter.warmup()
    return counter


def _run_job(app: Starlette, job_id: str, upload_path: Path) -> None:
    state = app.state
    try:
        with state.infer_lock:
            with state.job_lock:
                state.jobs[job_id] = Job(status="running")
            result = state.counter.count(upload_path)
        job = Job(status="done", result=result)
    except (AudioTooLong, InvalidAudio) as exc:
        job = Job(
            status="error", error=str(exc),
            code="too_long" if isinstance(exc, AudioTooLong) else "invalid_audio",
        )
    except Exception:
        _LOG.exception("Speaker analysis failed for job %s", job_id)
        job = Job(
            status="error", code="analysis_failed",
            error="Analysis failed. Please try a shorter clip or check the server logs.",
        )
    finally:
        try:
            shutil.rmtree(upload_path.parent)
        except OSError:
            _LOG.exception("Could not remove temporary audio for job %s", job_id)
    with state.job_lock:
        state.jobs[job_id] = job


def _save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> None:
    size = 0
    with destination.open("wb") as out:
        while chunk := upload.file.read(_UPLOAD_CHUNK):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(413, _SIZE_ERROR)
            out.write(chunk)
    if not size:
        raise HTTPException(400, "The file is empty. Choose a recording with audio.")


async def diarize(request: Request) -> JSONResponse:
    """Spool one bounded upload and enqueue its background analysis."""
    state = request.app.state
    settings: WebSettings = state.settings
    if state.counter is None:
        return JSONResponse({"detail": "The analysis service is not ready."}, status_code=503)
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            length = int(content_length)
        except ValueError:
            return JSONResponse({"detail": "Invalid upload length."}, status_code=400)
        if length < 0:
            return JSONResponse({"detail": "Invalid upload length."}, status_code=400)
        if length > settings.max_upload_bytes + _BODY_OVERHEAD:
            return JSONResponse({"detail": _SIZE_ERROR}, status_code=413)

    received = 0

    async def limited_receive() -> Message:
        nonlocal received
        message = await request.receive()
        if message["type"] == "http.request":
            received += len(message.get("body", b""))
            if received > settings.max_upload_bytes + _BODY_OVERHEAD:
                # The multipart parser closes its spool files for this exception.
                raise MultiPartException(_SIZE_ERROR)
        return message

    bounded_request = Request(request.scope, receive=limited_receive)
    directory = None
    job_id = None
    accepted = False
    try:
        async with bounded_request.form(max_files=1, max_fields=0) as form:
            upload = form.get("file")
            if not isinstance(upload, UploadFile) or not upload.filename:
                raise HTTPException(400, "Choose an audio file or record from your microphone.")
            if upload.size is not None and upload.size > settings.max_upload_bytes:
                raise HTTPException(413, _SIZE_ERROR)
            with state.job_lock:
                _prune_jobs(state.jobs, settings)
                pending = sum(job.status in ("queued", "running") for job in state.jobs.values())
                if pending >= settings.max_pending_jobs:
                    raise HTTPException(429, "The analysis queue is full. Please try again shortly.")
                job_id = uuid.uuid4().hex
                state.jobs[job_id] = Job()
            directory = Path(tempfile.mkdtemp(prefix="spk_upload_"))
            suffix = Path(upload.filename.replace("\\", "/")).suffix[:16]
            destination = directory / f"audio{suffix}"
            await run_in_threadpool(_save_upload, upload, destination, settings.max_upload_bytes)
        accepted = True
        return JSONResponse(
            {"job_id": job_id, "status": "queued"}, status_code=202,
            headers={"Cache-Control": "no-store"},
            background=BackgroundTask(_run_job, request.app, job_id, destination),
        )
    except HTTPException as exc:
        code = 413 if exc.detail == _SIZE_ERROR else exc.status_code
        return JSONResponse({"detail": exc.detail}, status_code=code)
    finally:
        if not accepted:
            if directory is not None:
                shutil.rmtree(directory)
            if job_id is not None:
                with state.job_lock:
                    state.jobs.pop(job_id, None)


async def job_status(request: Request) -> JSONResponse:
    """Return a snapshot of a queued, running or completed job."""
    state = request.app.state
    job_id = request.path_params["job_id"]
    with state.job_lock:
        _prune_jobs(state.jobs, state.settings)
        job = state.jobs.get(job_id)
        if job is None:
            return JSONResponse({"detail": "This job is unknown or has expired."}, status_code=404)
        payload = {"job_id": job_id, "status": job.status}
        if job.result is not None:
            payload["result"] = job.result
        if job.error is not None:
            payload.update(error=job.error, code=job.code)
    return JSONResponse(payload, headers={"Cache-Control": "no-store"})


async def healthz(request: Request) -> JSONResponse:
    """Expose readiness and the actual input limits to the browser."""
    counter = request.app.state.counter
    settings: WebSettings = request.app.state.settings
    return JSONResponse(
        {
            "ok": counter is not None,
            "status": request.app.state.model_status,
            "detail": (
                "The analysis service is ready." if counter is not None else
                "The analysis service is starting. Please wait a moment."
                if request.app.state.model_status == "loading" else
                "Audio analysis is currently unavailable. Check the server setup and restart the service."
            ),
            "device": counter.device if counter else None,
            "max_upload_bytes": settings.max_upload_bytes,
            "max_duration_sec": settings.max_duration_sec,
            "ffmpeg_available": ffmpeg_executable() is not None,
        },
        status_code=200 if counter is not None else 503,
        headers={"Cache-Control": "no-store"},
    )


async def index(request: Request) -> FileResponse:
    """Serve the application shell without caching an outdated UI."""
    return FileResponse(_STATIC / "index.html", headers={"Cache-Control": "no-cache"})


def create_app(
    counter: SpeakerCounter | None = None, settings: WebSettings | None = None
) -> Starlette:
    """Create the service; tests may supply a model-free counter."""
    settings = settings or WebSettings.from_env()

    @asynccontextmanager
    async def lifespan(application: Starlette) -> AsyncIterator[None]:
        async def prepare_models() -> None:
            try:
                application.state.counter = await run_in_threadpool(_load_models, settings)
                application.state.model_status = "ready"
            except Exception:
                _LOG.exception("Model startup failed. Check model access and requirements.txt.")
                application.state.model_status = "error"

        loading = None
        if application.state.counter is None:
            loading = asyncio.create_task(prepare_models())
        try:
            yield
        finally:
            if loading is not None:
                loading.cancel()
                with suppress(asyncio.CancelledError):
                    await loading

    application = Starlette(
        lifespan=lifespan,
        routes=[
            Route("/", index),
            Route("/diarize", diarize, methods=["POST"]),
            Route("/jobs/{job_id}", job_status),
            Route("/healthz", healthz),
            Mount("/static", app=StaticFiles(directory=_STATIC), name="static"),
        ],
    )
    application.state.settings = settings
    application.state.counter = counter
    application.state.model_status = "ready" if counter is not None else "loading"
    application.state.jobs = {}
    application.state.job_lock = threading.Lock()
    application.state.infer_lock = threading.Lock()
    return application


app = create_app()
