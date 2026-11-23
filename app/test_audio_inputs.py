"""Model-free regressions for real audio decoding and short-recording handling."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from app.infer import AudioTooLong, InvalidAudio, SpeakerCounter, ffmpeg_executable


class AudioInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.counter = SpeakerCounter(device="cpu")
        self.wav = self.root / "input.wav"
        samples = np.sin(np.arange(38400, dtype=np.float32) * (440 * 2 * np.pi / 16000))
        sf.write(self.wav, samples * .15, 16000)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_content_is_inspected_instead_of_trusting_the_extension(self) -> None:
        renamed = self.root / "voice.unknown"
        shutil.copyfile(self.wav, renamed)
        self.assertEqual(self.counter._ensure_decodable(renamed), renamed)

    def test_empty_audio_is_rejected_before_model_use(self) -> None:
        empty = self.root / "empty.wav"
        sf.write(empty, np.zeros(0, dtype=np.float32), 16000)
        self.counter.vad.intervals = Mock(side_effect=AssertionError("VAD must not run"))
        with self.assertRaises(InvalidAudio):
            self.counter.count(empty)

    def test_native_flac_is_accepted(self) -> None:
        flac = self.root / "voice.flac"
        waveform, sample_rate = sf.read(self.wav)
        sf.write(flac, waveform, sample_rate)
        self.assertEqual(self.counter._ensure_decodable(flac), flac)

    def test_duration_limit_is_checked_before_model_use(self) -> None:
        self.counter.max_duration_sec = .5
        self.counter.vad.intervals = Mock(side_effect=AssertionError("VAD must not run"))
        with self.assertRaises(AudioTooLong):
            self.counter.count(self.wav)

    def test_brief_speech_has_measured_speech_but_no_usable_windows(self) -> None:
        self.counter.vad.intervals = Mock(return_value=[(0.0, .5)])
        result = self.counter.count(self.wav)
        self.assertEqual(result["num_speakers"], 0)
        self.assertEqual(result["num_segments"], 0)
        self.assertAlmostEqual(result["speech_sec"], .5)

    def test_two_windows_do_not_exceed_the_neighbour_pruning_floor(self) -> None:
        self.counter.vad.intervals = Mock(return_value=[(0.0, 2.25)])
        embeddings = np.zeros((2, 192), dtype=np.float32)
        embeddings[0, 0] = embeddings[1, 1] = 1
        self.counter.embedder.encode = Mock(return_value=embeddings)
        result = self.counter.count(self.wav)
        self.assertEqual(result["num_segments"], 2)
        self.assertGreaterEqual(result["num_speakers"], 1)
        self.assertLessEqual(result["num_speakers"], 2)

    def test_failed_conversion_removes_the_entire_temporary_directory(self) -> None:
        invalid = self.root / "broken.wav"
        invalid.write_bytes(b"this is not audio")
        decoded_dir = self.root / "conversion"
        decoded_dir.mkdir()
        with patch("app.infer.tempfile.mkdtemp", return_value=str(decoded_dir)), \
             patch("app.infer.ffmpeg_executable", return_value="/test/ffmpeg"), \
             patch("app.infer.subprocess.run", side_effect=subprocess.CalledProcessError(
                 1, ["ffmpeg"], stderr="invalid input"
             )):
            with self.assertRaises(InvalidAudio):
                self.counter.count(invalid)
        self.assertFalse(decoded_dir.exists())

    def test_conversion_timeout_is_reported_and_cleaned_up(self) -> None:
        invalid = self.root / "clip.webm"
        invalid.write_bytes(b"unreadable input")
        decoded_dir = self.root / "timed-out"
        decoded_dir.mkdir()
        with patch("app.infer.tempfile.mkdtemp", return_value=str(decoded_dir)), \
             patch("app.infer.ffmpeg_executable", return_value="/test/ffmpeg"), \
             patch("app.infer.subprocess.run", side_effect=subprocess.TimeoutExpired(["ffmpeg"], 1)):
            with self.assertRaisesRegex(InvalidAudio, "timed out"):
                self.counter.count(invalid)
        self.assertFalse(decoded_dir.exists())

    def test_real_compressed_audio_inputs_decode_and_cleanup(self) -> None:
        executable = ffmpeg_executable()
        if executable is None:
            self.skipTest("FFmpeg is not available")
        formats = [
            ("mp3", "libmp3lame"), ("m4a", "aac"), ("webm", "libopus"),
            ("ogg", "libvorbis"), ("aac", "aac"), ("mp4", "aac"),
        ]
        for extension, codec in formats:
            with self.subTest(extension=extension):
                compressed = self.root / f"voice.{extension}"
                subprocess.run(
                    [executable, "-nostdin", "-y", "-loglevel", "error", "-i", str(self.wav),
                     "-c:a", codec, str(compressed)],
                    check=True, capture_output=True, timeout=15,
                )
                decoded = self.counter._ensure_decodable(compressed)
                try:
                    info = sf.info(decoded)
                    self.assertEqual(info.samplerate, 16000)
                    self.assertEqual(info.channels, 1)
                    self.assertGreater(info.duration, 2)
                finally:
                    if decoded != compressed:
                        shutil.rmtree(decoded.parent)
                self.counter.vad.intervals = Mock(return_value=[])
                created = []
                original = self.counter._ensure_decodable

                def capture(path: Path) -> Path:
                    result = original(path)
                    created.append(result.parent)
                    return result

                with patch.object(self.counter, "_ensure_decodable", side_effect=capture):
                    self.assertEqual(self.counter.count(compressed)["num_speakers"], 0)
                self.assertTrue(created)
                self.assertFalse(created[0].exists())


if __name__ == "__main__":
    unittest.main()
