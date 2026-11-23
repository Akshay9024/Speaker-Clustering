"""HTTP and background-job regressions without downloading or loading ML models."""

from __future__ import annotations

import io
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from starlette.testclient import TestClient

from app.infer import CountResult, InvalidAudio, SpeakerCounter
from app.main import Job, WebSettings, create_app


class StubCounter(SpeakerCounter):
    def __init__(self) -> None:
        self.device = "cpu"
        self.paths: list[Path] = []
        self.error: Exception | None = None

    def count(self, path: str | Path) -> CountResult:
        self.paths.append(Path(path))
        if self.error:
            raise self.error
        return {
            "num_speakers": 2, "num_segments": 3, "speech_sec": 2.5,
            "duration_sec": 3.0, "device": self.device,
        }


class BlockingCounter(StubCounter):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def count(self, path: str | Path) -> CountResult:
        self.entered.set()
        if not self.release.wait(5):
            raise RuntimeError("test counter was not released")
        return super().count(path)


def wav_bytes() -> bytes:
    """Build a real PCM payload for multipart uploads."""
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(1600, dtype=np.float32), 16000, format="WAV")
    return buffer.getvalue()


class WebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.counter = StubCounter()
        self.app = create_app(counter=self.counter)
        self.client = TestClient(self.app)

    def test_shell_and_local_assets_are_served(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("recordTab", response.text)
        for path in ("/static/styles.css", "/static/app.js"):
            self.assertEqual(self.client.get(path).status_code, 200)

    def test_health_exposes_actual_limits(self) -> None:
        response = self.client.get("/healthz")
        self.assertTrue(response.json()["ok"])
        self.assertEqual(response.json()["max_upload_bytes"], 100 * 1024 * 1024)
        self.assertIn("no-store", response.headers["cache-control"])

    def test_upload_returns_result_and_removes_temporary_files(self) -> None:
        response = self.client.post("/diarize", files={"file": ("voice.wav", wav_bytes(), "audio/wav")})
        self.assertEqual(response.status_code, 202)
        job = self.client.get("/jobs/" + response.json()["job_id"])
        self.assertEqual(job.json()["status"], "done")
        self.assertEqual(job.json()["result"]["num_speakers"], 2)
        self.assertTrue(self.counter.paths)
        self.assertFalse(self.counter.paths[0].parent.exists())

    def test_uploaded_filenames_cannot_choose_the_destination(self) -> None:
        response = self.client.post(
            "/diarize", files={"file": ("../../untrusted.wav", wav_bytes(), "audio/wav")}
        )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(self.counter.paths[0].name, "audio.wav")
        self.assertTrue(self.counter.paths[0].parent.name.startswith("spk_upload_"))

    def test_missing_empty_and_multiple_files_are_rejected(self) -> None:
        self.assertEqual(self.client.post("/diarize").status_code, 400)
        self.assertEqual(self.client.post(
            "/diarize", files={"file": ("empty.wav", b"", "audio/wav")}
        ).status_code, 400)
        self.assertEqual(self.client.post("/diarize", files=[
            ("file", ("a.wav", wav_bytes(), "audio/wav")),
            ("file", ("b.wav", wav_bytes(), "audio/wav")),
        ]).status_code, 400)
        self.assertFalse(self.app.state.jobs)

    def test_bad_multipart_is_a_clear_client_error(self) -> None:
        response = self.client.post(
            "/diarize", content=b"bad multipart",
            headers={"Content-Type": "multipart/form-data"},
        )
        self.assertEqual(response.status_code, 400)

    def test_file_size_is_enforced_even_with_small_multipart_overhead(self) -> None:
        client = TestClient(create_app(self.counter, WebSettings(max_upload_bytes=1024)))
        response = client.post("/diarize", files={"file": ("large.wav", b"x" * 2048, "audio/wav")})
        self.assertEqual(response.status_code, 413)
        self.assertFalse(self.counter.paths)

    def test_chunked_upload_cannot_bypass_the_request_limit(self) -> None:
        client = TestClient(create_app(self.counter, WebSettings(max_upload_bytes=1024)))
        parts = [
            b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="large.wav"\r\n'
            b'Content-Type: audio/wav\r\n\r\n',
            b"x" * (128 * 1024), b"\r\n--boundary--\r\n",
        ]
        response = client.post(
            "/diarize", content=iter(parts),
            headers={"Content-Type": "multipart/form-data; boundary=boundary"},
        )
        self.assertEqual(response.status_code, 413)
        self.assertFalse(self.counter.paths)

    def test_invalid_audio_is_a_structured_job_error(self) -> None:
        self.counter.error = InvalidAudio("No readable audio stream")
        response = self.client.post("/diarize", files={"file": ("bad.webm", b"bad", "audio/webm")})
        job = self.client.get("/jobs/" + response.json()["job_id"]).json()
        self.assertEqual(job["status"], "error")
        self.assertEqual(job["code"], "invalid_audio")
        self.assertFalse(self.counter.paths[0].parent.exists())

    def test_unknown_and_expired_jobs_are_not_polled_forever(self) -> None:
        self.assertEqual(self.client.get("/jobs/missing").status_code, 404)
        self.app.state.jobs["expired"] = Job(status="done", updated_at=time.monotonic() - 7200)
        self.app.state.jobs["active"] = Job(status="running", updated_at=0)
        self.assertEqual(self.client.get("/jobs/expired").status_code, 404)
        self.assertEqual(self.client.get("/jobs/active").json()["status"], "running")

    def test_running_job_does_not_block_http_and_queue_limits_work(self) -> None:
        counter = BlockingCounter()
        app = create_app(counter, WebSettings(max_pending_jobs=1))
        client = TestClient(app)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(
                client.post, "/diarize",
                files={"file": ("first.wav", wav_bytes(), "audio/wav")},
            )
            try:
                self.assertTrue(counter.entered.wait(3))
                with app.state.job_lock:
                    job_id = next(iter(app.state.jobs))
                self.assertEqual(client.get("/jobs/" + job_id).json()["status"], "running")
                self.assertEqual(client.get("/healthz").status_code, 200)
                other = client.post("/diarize", files={"file": ("second.wav", wav_bytes(), "audio/wav")})
                self.assertEqual(other.status_code, 429)
            finally:
                counter.release.set()
            self.assertEqual(pending.result(timeout=5).status_code, 202)
        self.assertEqual(client.get("/jobs/" + job_id).json()["status"], "done")


if __name__ == "__main__":
    unittest.main()
