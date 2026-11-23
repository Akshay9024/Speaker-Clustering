"""Starlette service: upload audio, get the estimated number of speakers.

Models load once at startup (warm). Inference runs as a background task so a long
upload doesn't hold the HTTP connection open; the browser polls ``/jobs/{id}``.
A single global lock serialises inference so concurrent uploads can't OOM the GPU.

Starlette (not FastAPI) on purpose: the API is trivial (upload in, dict out), so we
avoid pulling pydantic into this repo's tightly-pinned dependency closure.

ponytail: in-process job store + global lock. Fine for review/single-user. Swap in a
real task queue (Celery/RQ) + worker pool only if concurrent users are needed.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path

from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from app.infer import AudioTooLong, SpeakerCounter

_STATIC = Path(__file__).parent / "static"
_UPLOAD_CHUNK = 1024 * 1024

_jobs: dict[str, dict] = {}
_infer_lock = threading.Lock()
_counter: SpeakerCounter | None = None


def _load_models() -> None:
    global _counter
    _counter = SpeakerCounter(
        config_path=os.environ.get("SPK_CONFIG", "configs/voxconverse.yaml"),
        token=os.environ.get("HF_TOKEN"),
        max_duration_sec=float(os.environ.get("SPK_MAX_DURATION_SEC", "3600")),
    )
    _counter.warmup()


def _run_job(job_id: str, upload_path: Path) -> None:
    try:
        with _infer_lock:
            assert _counter is not None
            result = _counter.count(upload_path)
        _jobs[job_id] = {"status": "done", "result": result}
    except AudioTooLong as exc:
        _jobs[job_id] = {"status": "error", "error": str(exc), "code": "too_long"}
    except Exception as exc:  # surface the real error; no silent swallow
        _jobs[job_id] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(upload_path.parent, ignore_errors=True)


async def diarize(request):
    form = await request.form()
    upload = form.get("file")
    if upload is None or not getattr(upload, "filename", ""):
        return JSONResponse({"detail": "no file uploaded"}, status_code=400)

    tmp_dir = Path(tempfile.mkdtemp(prefix="upload_"))
    dest = tmp_dir / Path(upload.filename).name
    with dest.open("wb") as out:
        while chunk := await upload.read(_UPLOAD_CHUNK):
            out.write(chunk)

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "queued"}
    return JSONResponse(
        {"job_id": job_id, "status": "queued"},
        background=BackgroundTask(_run_job, job_id, dest),
    )


async def job_status(request):
    job_id = request.path_params["job_id"]
    job = _jobs.get(job_id)
    if job is None:
        return JSONResponse({"detail": "unknown job id"}, status_code=404)
    return JSONResponse({"job_id": job_id, **job})


async def healthz(request):
    return JSONResponse(
        {"ok": _counter is not None, "device": _counter.device if _counter else None}
    )


async def index(request):
    return FileResponse(_STATIC / "index.html")


app = Starlette(
    on_startup=[_load_models],
    routes=[
        Route("/", index),
        Route("/diarize", diarize, methods=["POST"]),
        Route("/jobs/{job_id}", job_status),
        Route("/healthz", healthz),
        Mount("/static", app=StaticFiles(directory=_STATIC), name="static"),
    ],
)
