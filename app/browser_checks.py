"""Browser QA with a deliberately model-free server; never used by production."""

from __future__ import annotations

import os
import shutil
import socket
import threading
import time
import unittest
from pathlib import Path

import uvicorn
from playwright.sync_api import Route, expect, sync_playwright

from app.main import create_app
from app.infer import CountResult, SpeakerCounter
from app.test_web import StubCounter, wav_bytes


class BrowserCounter(StubCounter):
    """Use real media decoding with a fixed, explicitly test-only speaker result."""

    def __init__(self) -> None:
        super().__init__()
        self.decoder = SpeakerCounter(device="cpu")

    def count(self, path: str | Path) -> CountResult:
        source = Path(path)
        decoded = self.decoder._ensure_decodable(source)
        try:
            return super().count(path)
        finally:
            if decoded != source:
                shutil.rmtree(decoded.parent)


class BrowserChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.counter = BrowserCounter()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        cls.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        cls.server = uvicorn.Server(uvicorn.Config(
            create_app(cls.counter), log_level="warning", lifespan="on"
        ))
        cls.server_thread = threading.Thread(
            target=cls.server.run, kwargs={"sockets": [sock]}, daemon=True
        )
        cls.server_thread.start()
        for _ in range(100):
            if cls.server.started:
                break
            time.sleep(.05)
        if not cls.server.started:
            raise RuntimeError("QA server did not start")
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(
            headless=True,
            args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"],
        )
        cls.artifacts = Path(os.environ.get("SPK_QA_ARTIFACTS", "/private/tmp/speaker-counter-qa"))
        cls.artifacts.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.browser.close()
        cls.playwright.stop()
        cls.server.should_exit = True
        cls.server_thread.join(timeout=5)

    def setUp(self) -> None:
        self.context = self.browser.new_context(
            viewport={"width": 1440, "height": 1100}, permissions=["microphone"]
        )
        self.page = self.context.new_page()
        self.errors: list[str] = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))

    def tearDown(self) -> None:
        self.context.close()
        self.assertEqual(self.errors, [], "Uncaught browser JavaScript errors")

    def open(self) -> None:
        self.page.goto(self.url)
        expect(self.page.locator("#connectionText")).to_have_text("Service ready")

    def upload(self, name: str = "conversation.wav") -> None:
        self.page.locator("#fileInput").set_input_files(
            {"name": name, "mimeType": "audio/wav", "buffer": wav_bytes()}
        )
        expect(self.page.locator("#preview")).to_be_visible()

    def hold_upload(self) -> None:
        """Control upload events without transferring hundreds of megabytes in QA."""
        self.page.add_init_script("""
          window.uploadSubmissions = 0;
          window.XMLHttpRequest = class extends EventTarget {
            constructor() {
              super();
              this.upload = new EventTarget();
              this.aborted = false;
              window.heldUpload = this;
            }
            open() {}
            send() { window.uploadSubmissions += 1; }
            abort() {
              this.aborted = true;
              this.dispatchEvent(new Event('abort'));
            }
          };
        """)

    def pause_clock(self) -> None:
        """Advance timeout tests explicitly instead of waiting several minutes."""
        self.page.clock.install(time=0)
        self.page.clock.pause_at(1000)

    def test_responsive_layouts_have_no_overflow_or_panel_overlaps(self) -> None:
        self.open()
        self.page.screenshot(path=str(self.artifacts / "desktop.png"), full_page=True)
        for width in (320, 375, 390, 680, 768, 1024, 1440):
            with self.subTest(width=width):
                self.page.set_viewport_size({"width": width, "height": 1000})
                result = self.page.evaluate("""() => {
                  const input = document.querySelector('.input-panel').getBoundingClientRect();
                  const output = document.querySelector('.result-panel').getBoundingClientRect();
                  const drop = document.querySelector('#dropZone').getBoundingClientRect();
                  const button = document.querySelector('#analyse').getBoundingClientRect();
                  return {
                    overflow: document.documentElement.scrollWidth > innerWidth,
                    overlap: input.left < output.right && input.right > output.left
                      && input.top < output.bottom && input.bottom > output.top,
                    controlsOverlap: drop.bottom > button.top
                  };
                }""")
                self.assertFalse(result["overflow"], f"Horizontal overflow at {width}px")
                self.assertFalse(result["overlap"], f"Overlapping panels at {width}px")
                self.assertFalse(result["controlsOverlap"], f"Overlapping controls at {width}px")

    def test_upload_preview_analysis_and_long_filename_on_mobile(self) -> None:
        self.open()
        self.page.set_viewport_size({"width": 375, "height": 900})
        self.upload("a-very-long-conversation-name-" * 7 + ".wav")
        expect(self.page.locator("#analyse")).to_be_enabled()
        self.assertFalse(self.page.evaluate("document.documentElement.scrollWidth > innerWidth"))
        self.page.locator("#analyse").click()
        expect(self.page.locator("#speakerCount")).to_have_text("2")
        expect(self.page.locator("#resultBadge")).to_have_text("Analysis complete")
        expect(self.page.locator("#metrics")).to_be_visible()
        self.page.screenshot(path=str(self.artifacts / "mobile-result.png"), full_page=True)
        self.page.locator("#startAgain").click()
        expect(self.page.locator("#preview")).to_be_hidden()
        expect(self.page.locator("#analyse")).to_be_disabled()

    def test_actual_mediarecorder_capture_can_be_previewed_and_submitted(self) -> None:
        self.open()
        self.page.locator("#recordTab").click()
        self.page.locator("#recordButton").click()
        try:
            expect(self.page.locator("#recordingLabel")).to_have_text(
                "RECORDING IN PROGRESS", timeout=15000
            )
        except AssertionError as exc:
            self.fail(f"Microphone did not start: {self.page.locator('#noticeText').inner_text()}; {exc}")
        expect(self.page.locator("#uploadTab")).to_be_disabled()
        self.page.wait_for_timeout(1800)
        self.page.screenshot(path=str(self.artifacts / "microphone.png"), full_page=True)
        self.page.locator("#recordButton").click()
        expect(self.page.locator("#preview")).to_be_visible()
        expect(self.page.locator("#fileName")).to_contain_text("recording-")
        expect(self.page.locator("#analyse")).to_be_enabled()
        self.assertFalse(self.page.evaluate("document.documentElement.scrollWidth > innerWidth"))
        self.page.locator("#analyse").click()
        expect(self.page.locator("#speakerCount")).to_have_text("2")

    def test_microphone_denial_recovers_without_disabling_upload(self) -> None:
        self.page.add_init_script("""
          navigator.mediaDevices.getUserMedia = async () => {
            throw new DOMException('Permission denied', 'NotAllowedError');
          };
        """)
        self.open()
        self.page.locator("#recordTab").click()
        self.page.locator("#recordButton").click()
        expect(self.page.locator("#noticeText")).to_contain_text("denied")
        expect(self.page.locator("#uploadTab")).to_be_enabled()
        self.page.locator("#uploadTab").click()
        self.upload()
        expect(self.page.locator("#analyse")).to_be_enabled()

    def test_pending_microphone_permission_can_be_cancelled(self) -> None:
        self.page.add_init_script("""
          navigator.mediaDevices.getUserMedia = () => new Promise(() => {});
        """)
        self.open()
        self.page.locator("#recordTab").click()
        self.page.locator("#recordButton").click()
        expect(self.page.locator("#recordButtonText")).to_contain_text("Waiting")
        self.page.locator("#cancelRecording").click()
        expect(self.page.locator("#uploadTab")).to_be_enabled()
        expect(self.page.locator("#recordButtonText")).to_have_text("Start recording")

    def test_upload_errors_are_readable_and_controls_recover(self) -> None:
        self.open()
        self.upload()
        self.page.route("**/diarize", lambda route: route.fulfill(
            status=413, json={"detail": "Choose a smaller file."}
        ))
        self.page.locator("#analyse").click()
        expect(self.page.locator("#noticeText")).to_have_text("Choose a smaller file.")
        expect(self.page.locator("#analyse")).to_be_enabled()
        expect(self.page.locator("#uploadTab")).to_be_enabled()

    def test_status_connection_failure_can_resume_without_another_upload(self) -> None:
        self.pause_clock()
        self.open()
        self.upload()
        uploads: list[str] = []
        self.page.on("request", lambda request: uploads.append(request.url)
                     if request.method == "POST" and request.url.endswith("/diarize") else None)
        attempts: list[bool] = []

        def status(route: Route) -> None:
            attempts.append(True)
            if len(attempts) <= 4:
                route.fulfill(status=503, json={"detail": "Connection interrupted"})
            else:
                route.fulfill(json={"status": "done", "result": {
                    "num_speakers": 2, "speech_sec": 2, "duration_sec": 3,
                }})

        self.page.route("**/jobs/*", status)
        self.page.locator("#analyse").click()
        for attempt, delay in ((1, 1000), (2, 2000), (3, 4000)):
            expect(self.page.locator("#processingHint")).to_contain_text(f"({attempt} of 3)")
            self.page.clock.run_for(delay)
        expect(self.page.locator("#checkResult")).to_be_visible()
        expect(self.page.locator("#analyseText")).to_have_text("Check analysis status")
        self.page.locator("#analyse").click()
        expect(self.page.locator("#speakerCount")).to_have_text("2")
        self.assertEqual(len(uploads), 1)
        self.assertEqual(len(attempts), 5)

    def test_temporary_status_failures_retry_automatically_without_another_upload(self) -> None:
        self.pause_clock()
        self.open()
        self.upload()
        uploads: list[str] = []
        self.page.on("request", lambda request: uploads.append(request.url)
                     if request.method == "POST" and request.url.endswith("/diarize") else None)
        attempts: list[bool] = []

        def status(route: Route) -> None:
            attempts.append(True)
            if len(attempts) == 1:
                route.fulfill(status=502, content_type="text/html", body="<h1>Bad gateway</h1>")
            elif len(attempts) == 2:
                route.abort("failed")
            elif len(attempts) == 3:
                route.fulfill(status=503, json={"detail": "Temporarily unavailable"})
            else:
                route.fulfill(json={"status": "done", "result": {
                    "num_speakers": 2, "speech_sec": 2, "duration_sec": 3,
                }})

        self.page.route("**/jobs/*", status)
        self.page.locator("#analyse").click()
        for attempt, delay in ((1, 1000), (2, 2000), (3, 4000)):
            expect(self.page.locator("#processingHint")).to_contain_text(f"({attempt} of 3)")
            self.page.clock.run_for(delay)
        expect(self.page.locator("#speakerCount")).to_have_text("2")
        expect(self.page.locator("#notice")).to_be_hidden()
        expect(self.page.locator("#checkResult")).to_be_hidden()
        self.assertEqual(len(uploads), 1)
        self.assertEqual(len(attempts), 4)

    def test_status_response_body_failures_retry_without_another_upload(self) -> None:
        self.pause_clock()
        self.page.add_init_script("""
          const originalFetch = window.fetch;
          window.statusRequests = 0;
          window.fetch = (url, options) => {
            if (String(url).startsWith('/jobs/') && ++window.statusRequests === 1) {
              return Promise.resolve({
                ok: true,
                json: () => new Promise((resolve, reject) => {
                  if (window.disconnectResponseBody) {
                    reject(new TypeError('Connection lost reading response body'));
                    return;
                  }
                  options.signal.addEventListener('abort', () => {
                    reject(new DOMException('Response body timed out', 'AbortError'));
                  }, { once: true });
                }),
              });
            }
            return originalFetch(url, options);
          };
        """)
        uploads: list[str] = []
        self.page.on("request", lambda request: uploads.append(request.url)
                     if request.method == "POST" and request.url.endswith("/diarize") else None)
        for disconnect in (False, True):
            with self.subTest(disconnect=disconnect):
                uploads.clear()
                self.open()
                self.upload()
                self.page.evaluate("disconnect => { window.disconnectResponseBody = disconnect; }", disconnect)
                self.page.locator("#analyse").click()
                self.page.wait_for_function("window.statusRequests === 1")
                if not disconnect:
                    self.page.clock.run_for(20000)
                expect(self.page.locator("#processingHint")).to_contain_text("(1 of 3)")
                self.page.clock.run_for(1000)
                expect(self.page.locator("#speakerCount")).to_have_text("2")
                self.assertEqual(self.page.evaluate("window.statusRequests"), 2)
                self.assertEqual(len(uploads), 1)

    def test_upload_progress_allows_transfers_longer_than_two_minutes(self) -> None:
        self.pause_clock()
        self.hold_upload()
        self.open()
        self.upload()
        self.page.route("**/jobs/slow-upload", lambda route: route.fulfill(json={
            "status": "done", "result": {"num_speakers": 2, "speech_sec": 2, "duration_sec": 3},
        }))
        self.page.locator("#analyse").click()
        for loaded in (25, 50, 75):
            self.page.clock.run_for(60000)
            self.page.evaluate("""loaded => heldUpload.upload.dispatchEvent(
              new ProgressEvent('progress', { lengthComputable: true, loaded, total: 100 })
            )""", loaded)
            expect(self.page.locator("#processingText")).to_have_text(f"Uploading your audio · {loaded}%")
            expect(self.page.locator("#notice")).to_be_hidden()
            expect(self.page.locator("#analyse")).to_be_disabled()
            self.assertFalse(self.page.evaluate("heldUpload.aborted"))
        self.page.evaluate("heldUpload.upload.dispatchEvent(new Event('load'))")
        expect(self.page.locator("#processingText")).to_have_text("Waiting for upload confirmation")
        self.page.evaluate("""() => {
          heldUpload.status = 202;
          heldUpload.response = { job_id: 'slow-upload', status: 'queued' };
          heldUpload.dispatchEvent(new Event('load'));
        }""")
        expect(self.page.locator("#speakerCount")).to_have_text("2")
        expect(self.page.locator("#analyse")).to_be_enabled()
        self.assertEqual(self.page.evaluate("window.uploadSubmissions"), 1)

    def test_stalled_upload_stops_without_automatically_resubmitting(self) -> None:
        self.pause_clock()
        self.hold_upload()
        self.open()
        self.upload()
        self.page.locator("#analyse").click()
        self.page.clock.run_for(60000)
        self.page.evaluate("""() => heldUpload.upload.dispatchEvent(
          new ProgressEvent('progress', { lengthComputable: true, loaded: 25, total: 100 })
        )""")
        self.page.clock.run_for(60000)
        self.page.evaluate("""() => heldUpload.upload.dispatchEvent(
          new ProgressEvent('progress', { lengthComputable: true, loaded: 25, total: 100 })
        )""")
        self.page.clock.run_for(60001)
        expect(self.page.locator("#noticeText")).to_contain_text("no progress for two minutes")
        expect(self.page.locator("#analyse")).to_be_enabled()
        expect(self.page.locator("#checkResult")).to_be_hidden()
        self.assertTrue(self.page.evaluate("heldUpload.aborted"))
        self.page.clock.run_for(8000)
        self.assertEqual(self.page.evaluate("window.uploadSubmissions"), 1)

    def test_sent_upload_times_out_waiting_for_server_confirmation(self) -> None:
        self.pause_clock()
        self.hold_upload()
        self.open()
        self.upload()
        self.page.locator("#analyse").click()
        self.page.evaluate("heldUpload.upload.dispatchEvent(new Event('load'))")
        self.page.clock.run_for(120001)
        expect(self.page.locator("#noticeText")).to_contain_text("server did not confirm receipt")
        expect(self.page.locator("#analyse")).to_be_enabled()
        self.assertTrue(self.page.evaluate("heldUpload.aborted"))
        self.assertEqual(self.page.evaluate("window.uploadSubmissions"), 1)

    def test_missing_job_finishes_polling_with_a_clear_error(self) -> None:
        self.open()
        self.upload()
        self.page.route("**/jobs/*", lambda route: route.fulfill(
            status=404, json={"detail": "This job has expired."}
        ))
        self.page.locator("#analyse").click()
        expect(self.page.locator("#noticeText")).to_have_text("This job has expired.")
        expect(self.page.locator("#analyse")).to_be_enabled()
        expect(self.page.locator("#checkResult")).to_be_hidden()

    def test_keyboard_tabs_and_empty_file_handling(self) -> None:
        self.open()
        self.page.locator("#uploadTab").focus()
        self.page.keyboard.press("ArrowRight")
        expect(self.page.locator("#recordPanel")).to_be_visible()
        self.page.keyboard.press("ArrowLeft")
        expect(self.page.locator("#uploadPanel")).to_be_visible()
        self.page.locator("#fileInput").set_input_files(
            {"name": "empty.wav", "mimeType": "audio/wav", "buffer": b""}
        )
        expect(self.page.locator("#noticeText")).to_contain_text("empty")
        expect(self.page.locator("#analyse")).to_be_disabled()

    def test_model_setup_errors_leave_audio_preparation_available(self) -> None:
        self.page.route("**/healthz", lambda route: route.fulfill(
            status=503, json={"ok": False, "status": "error", "detail": "Analysis unavailable"}
        ))
        self.page.goto(self.url)
        expect(self.page.locator("#connectionText")).to_have_text("Analysis unavailable")
        self.upload()
        expect(self.page.locator("#audioPreview")).to_be_visible()
        expect(self.page.locator("#analyse")).to_be_disabled()
        expect(self.page.locator("#actionHint")).to_contain_text("still record and preview")
        self.assertFalse(self.page.evaluate("document.documentElement.scrollWidth > innerWidth"))


if __name__ == "__main__":
    unittest.main()
