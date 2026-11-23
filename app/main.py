"""FastAPI service: upload audio, get the estimated number of speakers.

Models load once at startup (warm). Inference runs as a background job so a long
upload doesn't hold the HTTP connection open; the browser polls ``/jobs/{id}``.
A single global lock serialises inference so concurrent uploads can't OOM the GPU.

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

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.infer import AudioTooLong, SpeakerCounter

_STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Speaker Counter")
_jobs: dict[str, dict] = {}
_infer_lock = threading.Lock()
_counter: SpeakerCounter | None = None


@app.on_event("startup")
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
    except Exception as exc:  # surface the real error to the client; no silent swallow
        _jobs[job_id] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(upload_path.parent, ignore_errors=True)


@app.post("/diarize")
async def diarize(file: UploadFile, background: BackgroundTasks) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="no file uploaded")
    tmp_dir = Path(tempfile.mkdtemp(prefix="upload_"))
    dest = tmp_dir / file.filename
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "queued"}
    background.add_task(_run_job, job_id, dest)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    return {"job_id": job_id, **job}


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": _counter is not None, "device": _counter.device if _counter else None}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


app.mount("/static", StaticFiles(directory=_STATIC), name="static")
