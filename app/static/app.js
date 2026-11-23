"use strict";

(() => {
  const byId = (id) => document.getElementById(id);
  const ui = Object.fromEntries([
    "inputPanel", "uploadTab", "recordTab", "uploadPanel", "recordPanel", "dropZone", "fileInput",
    "fileLimit", "preview", "fileName", "fileMeta", "removeFile", "audioPreview",
    "previewHint", "notice", "noticeText", "dismissNotice", "analyse", "analyseText",
    "analyseIcon", "analyseSpinner", "actionHint", "resultPanel", "resultBadge",
    "speakerCount", "resultHeading", "resultDescription", "processing", "processingText",
    "processingHint", "analysisTime", "metrics", "audioDuration", "speechDuration",
    "checkResult", "startAgain", "connection", "connectionText", "reconnect",
    "recordButton", "recordButtonText", "recorder", "recordingLabel", "recordTime",
    "micHelp", "cancelRecording", "levelMeter",
  ].map((id) => [id, byId(id)]));
  const state = {
    file: null, source: "upload", mode: "upload", previewUrl: null, duration: null,
    ready: false, busy: false, jobId: null, pendingMic: false, recording: false,
    stopping: false, recorder: null, stream: null, discardRecording: false,
    audioContext: null, meterFrame: null, recordTimer: null, elapsedTimer: null,
    captureVersion: 0, maxBytes: 100 * 1024 * 1024, maxDuration: 3600,
  };
  const MAX_WAIT_MS = 30 * 60 * 1000;
  let activeRequest = null;

  function clock(seconds) {
    const safe = Math.max(0, Math.floor(Number(seconds) || 0));
    const hours = Math.floor(safe / 3600);
    const minutes = Math.floor(safe / 60) % 60;
    const rest = safe % 60;
    return (hours ? String(hours).padStart(2, "0") + ":" : "")
      + String(minutes).padStart(2, "0") + ":" + String(rest).padStart(2, "0");
  }

  function sizeLabel(bytes) {
    return bytes < 1024 * 1024
      ? Math.max(1, Math.round(bytes / 1024)) + " KB"
      : (bytes / (1024 * 1024)).toFixed(bytes < 10 * 1024 * 1024 ? 1 : 0) + " MB";
  }

  function showNotice(message) {
    ui.noticeText.textContent = message;
    ui.notice.hidden = false;
  }

  function clearNotice() {
    ui.notice.hidden = true;
    ui.noticeText.textContent = "";
  }

  function updateControls() {
    const capturing = state.pendingMic || state.recording || state.stopping;
    const locked = state.busy || capturing;
    const tooLong = state.maxDuration > 0 && Number.isFinite(state.duration)
      && state.duration > state.maxDuration;
    const tooLarge = state.file && state.file.size > state.maxBytes;
    ui.uploadTab.disabled = locked;
    ui.recordTab.disabled = locked;
    ui.dropZone.disabled = locked;
    ui.fileInput.disabled = locked;
    ui.removeFile.disabled = locked;
    ui.analyse.disabled = locked || !state.file || !state.ready || tooLong || tooLarge;
    ui.recordButton.disabled = state.busy || state.pendingMic || state.stopping;
    ui.cancelRecording.hidden = !state.pendingMic && !state.recording;
    if (state.recording) {
      ui.recordButtonText.textContent = "Stop recording";
    } else if (state.pendingMic) {
      ui.recordButtonText.textContent = "Waiting for permission…";
    } else if (state.stopping) {
      ui.recordButtonText.textContent = "Saving recording…";
    } else {
      ui.recordButtonText.textContent = state.source === "microphone" && state.file
        ? "Record again" : "Start recording";
    }
    if (state.busy) {
      ui.actionHint.textContent = "Your audio is being analysed. Longer recordings take more time.";
    } else if (state.pendingMic) {
      ui.actionHint.textContent = "Allow microphone access in your browser, or cancel to use a file.";
    } else if (state.recording || state.stopping) {
      ui.actionHint.textContent = "Stop recording to preview and analyse your audio.";
    } else if (!state.ready) {
      ui.actionHint.textContent = "Connect to the analysis service to continue.";
    } else if (tooLong || tooLarge) {
      ui.actionHint.textContent = "Choose a shorter or smaller recording within the server limits.";
    } else if (state.file) {
      ui.actionHint.textContent = "Listen to your clip, then estimate the speaker count.";
    } else {
      ui.actionHint.textContent = "Choose a file or record some audio to begin.";
    }
  }

  function resetResult() {
    ui.resultPanel.dataset.state = "idle";
    ui.resultBadge.dataset.state = "idle";
    ui.resultBadge.textContent = state.file ? "Ready to analyse" : "Ready when you are";
    ui.speakerCount.textContent = "—";
    ui.resultHeading.textContent = "Every conversation starts somewhere.";
    ui.resultDescription.textContent = "Add your audio, then estimate the number of distinct speakers.";
    ui.processing.hidden = true;
    ui.metrics.hidden = true;
    ui.checkResult.hidden = true;
    ui.startAgain.hidden = true;
  }

  function updateFileMeta() {
    if (!state.file) return;
    const details = [state.source === "microphone" ? "Microphone recording" : "Selected file",
      sizeLabel(state.file.size)];
    if (Number.isFinite(state.duration)) details.push(clock(state.duration));
    ui.fileMeta.textContent = details.join(" · ");
  }

  function clearSelection() {
    ui.audioPreview.pause();
    ui.audioPreview.removeAttribute("src");
    ui.audioPreview.load();
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.file = null;
    state.previewUrl = null;
    state.duration = null;
    state.jobId = null;
    ui.fileInput.value = "";
    ui.preview.hidden = true;
    ui.previewHint.hidden = true;
    resetResult();
    updateControls();
  }

  function selectFile(file, source = "upload", recordedDuration = null) {
    if (!file || state.busy) return;
    if (!file.size) {
      showNotice("This file is empty. Choose a recording that contains audio.");
      return;
    }
    if (file.size > state.maxBytes) {
      showNotice("This recording is larger than the " + sizeLabel(state.maxBytes)
        + " limit. Choose a smaller file.");
      return;
    }
    clearSelection();
    clearNotice();
    state.file = file;
    state.source = source;
    state.duration = recordedDuration;
    state.previewUrl = URL.createObjectURL(file);
    ui.fileName.textContent = file.name;
    ui.fileName.title = file.name;
    ui.audioPreview.src = state.previewUrl;
    ui.preview.hidden = false;
    updateFileMeta();
    resetResult();
    updateControls();
  }

  function setMode(mode) {
    if (state.busy || state.pendingMic || state.recording || state.stopping) return;
    state.mode = mode;
    const upload = mode === "upload";
    ui.uploadTab.setAttribute("aria-selected", String(upload));
    ui.recordTab.setAttribute("aria-selected", String(!upload));
    ui.uploadTab.tabIndex = upload ? 0 : -1;
    ui.recordTab.tabIndex = upload ? -1 : 0;
    ui.uploadPanel.hidden = !upload;
    ui.recordPanel.hidden = upload;
    if (!upload) {
      if (!window.isSecureContext) {
        ui.micHelp.textContent = "Open this site over HTTPS or localhost to use your microphone.";
      } else if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        ui.micHelp.textContent = "Recording is unavailable in this browser. You can upload a voice note instead.";
      }
      requestAnimationFrame(() => drawMeter());
    }
    updateControls();
  }

  async function requestJSON(url, options = {}, timeoutMs = 20000, analysisRequest = false) {
    const controller = new AbortController();
    if (analysisRequest) activeRequest = controller;
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, { ...options, signal: controller.signal, cache: "no-store" });
      let body;
      try {
        body = await response.json();
      } catch {
        throw new Error("The server returned an unreadable response. Please try again.");
      }
      if (!response.ok) {
        const error = new Error(typeof body.detail === "string" ? body.detail
          : "The request failed (" + response.status + "). Please try again.");
        error.status = response.status;
        throw error;
      }
      return body;
    } catch (error) {
      if (error.name === "AbortError") {
        throw new Error("The server took too long to respond. Check your connection and try again.");
      }
      if (error instanceof TypeError) {
        throw new Error("Cannot reach the analysis service. Check your connection and try again.");
      }
      throw error;
    } finally {
      clearTimeout(timer);
      if (activeRequest === controller) activeRequest = null;
    }
  }

  async function checkConnection() {
    ui.reconnect.disabled = true;
    try {
      const health = await requestJSON("/healthz", {}, 10000);
      if (!health.ok) throw new Error("Service is not ready");
      state.ready = true;
      if (Number.isFinite(health.max_upload_bytes) && health.max_upload_bytes > 0) {
        state.maxBytes = health.max_upload_bytes;
      }
      if (Number.isFinite(health.max_duration_sec) && health.max_duration_sec >= 0) {
        state.maxDuration = health.max_duration_sec;
      }
      ui.connection.dataset.state = "ready";
      ui.connectionText.textContent = "Service ready";
      ui.reconnect.hidden = true;
      ui.fileLimit.textContent = "Up to " + sizeLabel(state.maxBytes)
        + (state.maxDuration ? " / " + Math.round(state.maxDuration / 60) + " min" : "");
      if (!health.ffmpeg_available) {
        ui.fileLimit.textContent += " · WAV / FLAC recommended";
      }
    } catch {
      state.ready = false;
      ui.connection.dataset.state = "offline";
      ui.connectionText.textContent = "Service offline";
      ui.reconnect.hidden = false;
      ui.fileLimit.textContent = "Server connection needed";
    } finally {
      ui.reconnect.disabled = false;
      updateControls();
    }
  }

  function setBusy(busy) {
    state.busy = busy;
    ui.analyseText.textContent = busy ? "Analysing audio…" : "Estimate speakers";
    ui.analyseIcon.toggleAttribute("hidden", busy);
    ui.analyseSpinner.hidden = !busy;
    ui.inputPanel?.setAttribute("aria-busy", String(busy));
    if (!busy) clearInterval(state.elapsedTimer);
    updateControls();
  }

  function startProcessing() {
    clearNotice();
    resetResult();
    setBusy(true);
    ui.resultPanel.dataset.state = "busy";
    ui.resultBadge.textContent = "Analysis in progress";
    ui.resultHeading.textContent = "Listening for distinct voices.";
    ui.resultDescription.textContent = "Your estimate will appear here when the analysis is complete.";
    ui.processing.hidden = false;
    ui.analysisTime.textContent = "00:00";
    const started = performance.now();
    state.elapsedTimer = setInterval(() => {
      ui.analysisTime.textContent = clock((performance.now() - started) / 1000);
    }, 1000);
    if (window.matchMedia("(max-width: 680px)").matches) {
      ui.resultPanel.scrollIntoView({
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth",
        block: "nearest",
      });
    }
  }

  async function pollJob(jobId) {
    const started = performance.now();
    while (performance.now() - started < MAX_WAIT_MS) {
      const job = await requestJSON("/jobs/" + encodeURIComponent(jobId), {}, 20000, true);
      if (job.status === "done") {
        const result = job.result;
        if (!result || !Number.isInteger(result.num_speakers) || result.num_speakers < 0
          || !Number.isFinite(result.duration_sec) || !Number.isFinite(result.speech_sec)) {
          state.jobId = null;
          throw new Error("The server returned an incomplete estimate. Please try again.");
        }
        return result;
      }
      if (job.status === "error") {
        state.jobId = null;
        throw new Error(typeof job.error === "string" ? job.error : "Audio analysis failed.");
      }
      if (job.status !== "queued" && job.status !== "running") {
        throw new Error("The server returned an unknown analysis status. Try checking the result again.");
      }
      ui.processingText.textContent = job.status === "queued" ? "Waiting in the analysis queue" : "Analysing your recording";
      ui.processingHint.textContent = job.status === "queued"
        ? "Another recording is being processed. Yours will start when it is ready."
        : "Detecting speech and comparing voices. Longer clips can take more time.";
      await new Promise((resolve) => setTimeout(resolve, 1500));
    }
    throw new Error("This analysis is taking longer than expected. Use “Check analysis status” to reconnect.");
  }

  function showResult(result) {
    const count = result.num_speakers;
    state.jobId = null;
    ui.resultPanel.dataset.state = "done";
    ui.resultBadge.dataset.state = "done";
    ui.resultBadge.textContent = "Analysis complete";
    ui.speakerCount.textContent = String(count);
    ui.resultHeading.textContent = count ? (count === 1 ? "One distinct speaker." : count + " distinct speakers.")
      : "No analysable speech found.";
    ui.resultDescription.textContent = count
      ? "The estimated number of distinct voices in your recording."
      : result.speech_sec > 0
        ? "Some speech was found, but it was too brief to analyse. Try a longer, clearer clip."
        : "Try a recording with clear, audible conversation.";
    ui.audioDuration.textContent = clock(result.duration_sec);
    ui.speechDuration.textContent = clock(result.speech_sec);
    ui.processing.hidden = true;
    ui.metrics.hidden = false;
    ui.startAgain.hidden = false;
    ui.checkResult.hidden = true;
  }

  async function analyse(resume = false) {
    if (state.busy || state.pendingMic || state.recording || state.stopping) return;
    if (!resume && (!state.file || !state.ready)) return;
    if (resume && !state.jobId) return;
    ui.audioPreview.pause();
    startProcessing();
    try {
      if (!resume) {
        ui.processingText.textContent = "Uploading your audio";
        const form = new FormData();
        form.append("file", state.file, state.file.name);
        const response = await requestJSON("/diarize", { method: "POST", body: form }, 120000, true);
        if (typeof response.job_id !== "string" || !response.job_id) {
          throw new Error("The server did not return an analysis ID. Please try again.");
        }
        state.jobId = response.job_id;
      }
      showResult(await pollJob(state.jobId));
    } catch (error) {
      if (error.status === 404) state.jobId = null;
      ui.resultPanel.dataset.state = "error";
      ui.resultBadge.textContent = state.jobId ? "Connection interrupted" : "Needs another try";
      ui.resultHeading.textContent = state.jobId ? "Your result may still be on its way." : "Let’s try that again.";
      ui.resultDescription.textContent = state.jobId
        ? "Your job is saved on the server. Check its status to continue waiting."
        : "Review the message beside your audio, then try again.";
      ui.processing.hidden = true;
      ui.checkResult.hidden = !state.jobId;
      showNotice(error.message || "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  function drawMeter(analyser = null, samples = null) {
    const canvas = ui.levelMeter;
    const context = canvas.getContext("2d");
    if (!context) return;
    const width = Math.max(1, canvas.getBoundingClientRect().width);
    const height = 55;
    const ratio = window.devicePixelRatio || 1;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    if (analyser) analyser.getByteTimeDomainData(samples);
    context.lineWidth = 2;
    context.strokeStyle = state.recording ? "#b4aaff" : "#465370";
    context.beginPath();
    const length = samples ? samples.length : 2;
    for (let index = 0; index < length; index += 1) {
      const x = (index / (length - 1)) * width;
      const y = samples ? height / 2 + ((samples[index] - 128) / 128) * height * .45 : height / 2;
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    }
    context.stroke();
    if (state.recording && analyser) {
      state.meterFrame = requestAnimationFrame(() => drawMeter(analyser, samples));
    }
  }

  function releaseMicrophone() {
    clearInterval(state.recordTimer);
    cancelAnimationFrame(state.meterFrame);
    if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
    if (state.audioContext) state.audioContext.close().catch(() => {});
    state.audioContext = null;
    state.recorder = null;
    state.recording = false;
    state.pendingMic = false;
    state.stopping = false;
    ui.recorder.dataset.state = "idle";
    ui.recordingLabel.textContent = "YOUR MICROPHONE";
    updateControls();
    drawMeter();
  }

  function stopRecording(discard = false) {
    if (state.pendingMic) {
      state.captureVersion += 1;
      state.pendingMic = false;
      updateControls();
      return;
    }
    state.discardRecording = discard;
    if (state.recorder && state.recorder.state !== "inactive") {
      state.stopping = true;
      state.recording = false;
      state.recorder.stop();
      clearInterval(state.recordTimer);
      updateControls();
    }
  }

  async function startRecording() {
    if (state.busy || state.stopping) return;
    if (state.recording) return stopRecording();
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      showNotice(!window.isSecureContext
        ? "Microphone recording needs HTTPS or localhost. Open a secure connection or upload an audio file."
        : "This browser does not support audio recording. Upload a voice note instead.");
      return;
    }
    clearNotice();
    const version = ++state.captureVersion;
    state.pendingMic = true;
    updateControls();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        video: false,
      });
      if (version !== state.captureVersion) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      state.stream = stream;
      const mime = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4"]
        .find((type) => MediaRecorder.isTypeSupported(type));
      const recorder = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream);
      const chunks = [];
      let bytes = 0;
      let recordingError = false;
      const started = performance.now();
      state.recorder = recorder;
      state.pendingMic = false;
      state.recording = true;
      state.discardRecording = false;
      clearSelection();
      ui.recordTime.textContent = "00:00";
      ui.recorder.dataset.state = "recording";
      ui.recordingLabel.textContent = "RECORDING IN PROGRESS";
      ui.micHelp.textContent = "Recording from your microphone. Stop when your conversation is ready.";
      recorder.ondataavailable = (event) => {
        if (event.data.size) {
          chunks.push(event.data);
          bytes += event.data.size;
          if (bytes > state.maxBytes && recorder.state !== "inactive") {
            showNotice("The recording reached the file-size limit. Try a shorter clip.");
            stopRecording();
          }
        }
      };
      recorder.onerror = () => {
        recordingError = true;
        showNotice("The microphone stopped unexpectedly. Check the device and try recording again.");
        stopRecording(true);
        if (recorder.state === "inactive") releaseMicrophone();
      };
      recorder.onstop = () => {
        const duration = (performance.now() - started) / 1000;
        const discard = state.discardRecording || recordingError;
        const type = recorder.mimeType || chunks[0]?.type || "audio/webm";
        releaseMicrophone();
        ui.micHelp.textContent = "Preview your recording below, or record a fresh take.";
        if (!discard) {
          const blob = new Blob(chunks, { type });
          const extension = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
          const name = "recording-" + new Date().toISOString().replace(/[:.]/g, "-") + "." + extension;
          selectFile(new File([blob], name, { type }), "microphone", duration);
        }
      };
      stream.getAudioTracks().forEach((track) => {
        track.onended = () => { if (state.recording) stopRecording(); };
      });
      recorder.start(1000);
      state.recordTimer = setInterval(() => {
        const duration = (performance.now() - started) / 1000;
        ui.recordTime.textContent = clock(duration);
        if (state.maxDuration && duration >= state.maxDuration - .25) {
          stopRecording();
        }
      }, 200);
      try {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (AudioContextClass) {
          state.audioContext = new AudioContextClass();
          const analyser = state.audioContext.createAnalyser();
          analyser.fftSize = 256;
          state.audioContext.createMediaStreamSource(stream).connect(analyser);
          drawMeter(analyser, new Uint8Array(analyser.fftSize));
        }
      } catch {
        // The level meter is optional; MediaRecorder still captures the audio.
        drawMeter();
      }
      updateControls();
    } catch (error) {
      if (version !== state.captureVersion) return;
      releaseMicrophone();
      const messages = {
        NotAllowedError: "Microphone access was denied. Allow it in your browser’s site settings, or upload an audio file.",
        NotFoundError: "No microphone was found. Connect one, or upload an audio file.",
        NotReadableError: "The microphone is unavailable. Close other apps using it and try again.",
        SecurityError: "This browser blocked microphone access. Use HTTPS or localhost, or upload a file.",
      };
      showNotice(messages[error.name] || "Recording could not start. Check your microphone and try again.");
    }
  }

  ui.uploadTab.addEventListener("click", () => setMode("upload"));
  ui.recordTab.addEventListener("click", () => setMode("record"));
  [ui.uploadTab, ui.recordTab].forEach((tab) => tab.addEventListener("keydown", (event) => {
    if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      const mode = event.key === "Home" ? "upload" : event.key === "End" ? "record"
        : state.mode === "upload" ? "record" : "upload";
      setMode(mode);
      (mode === "upload" ? ui.uploadTab : ui.recordTab).focus();
    }
  }));
  ui.dropZone.addEventListener("click", () => ui.fileInput.click());
  ui.fileInput.addEventListener("change", () => selectFile(ui.fileInput.files[0]));
  ["dragenter", "dragover"].forEach((name) => ui.dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    if (!state.busy) ui.dropZone.classList.add("over");
  }));
  ["dragleave", "drop"].forEach((name) => ui.dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    ui.dropZone.classList.remove("over");
  }));
  ui.dropZone.addEventListener("drop", (event) => {
    if (state.busy) return;
    if (event.dataTransfer.files.length > 1) {
      showNotice("Choose one recording at a time.");
      return;
    }
    selectFile(event.dataTransfer.files[0]);
  });
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("drop", (event) => event.preventDefault());
  ui.audioPreview.addEventListener("loadedmetadata", () => {
    if (!state.file) return;
    if (Number.isFinite(ui.audioPreview.duration)) state.duration = ui.audioPreview.duration;
    updateFileMeta();
    updateControls();
    if (state.maxDuration && state.duration > state.maxDuration) {
      showNotice("This recording exceeds the " + Math.round(state.maxDuration / 60)
        + "-minute limit. Choose a shorter clip.");
    }
  });
  ui.audioPreview.addEventListener("error", () => {
    if (!state.file) return;
    ui.previewHint.textContent = "Your browser cannot preview this format. The server can still try to analyse it.";
    ui.previewHint.hidden = false;
  });
  ui.removeFile.addEventListener("click", () => { clearSelection(); clearNotice(); });
  ui.analyse.addEventListener("click", () => analyse());
  ui.checkResult.addEventListener("click", () => analyse(true));
  ui.startAgain.addEventListener("click", () => {
    clearSelection(); clearNotice(); setMode("upload"); ui.dropZone.focus();
  });
  ui.dismissNotice.addEventListener("click", clearNotice);
  ui.reconnect.addEventListener("click", checkConnection);
  ui.recordButton.addEventListener("click", startRecording);
  ui.cancelRecording.addEventListener("click", () => stopRecording(true));
  const connectionTimer = setInterval(() => { if (!state.busy) checkConnection(); }, 30000);
  window.addEventListener("pagehide", () => {
    state.captureVersion += 1;
    state.discardRecording = true;
    if (state.recorder && state.recorder.state !== "inactive") state.recorder.stop();
    releaseMicrophone();
    if (activeRequest) activeRequest.abort();
    clearInterval(connectionTimer);
    clearInterval(state.elapsedTimer);
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  });
  resetResult();
  updateControls();
  checkConnection();
})();
