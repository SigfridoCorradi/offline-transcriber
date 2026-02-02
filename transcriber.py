import html
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import gradio as gr
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

# === LLM model to be used locally (offline) via Ollama to generate the summary and bulleted list. ===
OLLAMA_MODEL = "qwen3:30b-a3b-instruct-2507-q4_K_M"

# === Configurable texts (IT/EN) ===
LANG_DEFAULT = "it" #default language for labels, prompts, and transcription

#list of supported languages (can be expanded as desired with any other language)
LANG_CHOICES = [
    ("Italiano", "it"),
    ("English", "en"),
]

#labels and prompts based on language
TEXTS = {
    "it": {
        "prompts": {
            "summary": (
                "Fai un riassunto schematico del testo seguente. "
                "Includi solo gli elementi chiave in un testo sintetico. "
                "Non usare elenchi puntati e non aggiungere testo extra.\n\n"
                "Testo da riassumere:\n{content}"
            ),
            "bullets": (
                "Crea una lista puntata con gli elementi essenziali del testo seguente. "
                "Frasi brevi, senza testo extra.\n\n"
                "Testo:\n{content}"
            ),
        },
        "errors": {
            "ffmpeg_not_found": "ffmpeg non trovato per la conversione in WAV.",
            "wav_conversion_failed": "Conversione in WAV non riuscita.",
            "file_not_found": "File non trovato.",
            "transcription_error": "Errore durante conversione/trascrizione: {error}",
        },
        "ui": {
            "markdown_header": (
                "# Trascrivi audio!\n"
                "Carica uno o più file audio. I file vengono trascritti in sequenza!"
            ),
            "alert_ffmpeg_missing": (
                "<div class='alert'>ATTENZIONE: utility ffmpeg non trovata. "
                "I file diversi da WAV non verranno convertiti!</div>"
            ),
            "label_files": "Registrazioni da trascrivere:",
            "label_summary": "Crea riassunto schematico dal testo trascritto",
            "label_bullets": "Crea lista puntata dal testo trascritto",
            "button_transcribe": "Trascrivi",
            "output_label": "Risultati",
            "empty_results": "Nessun risultato da mostrare.",
            "result_time": "Tempo: {seconds:.2f}s",
            "pane_transcription_title": "Trascrizione",
            "pane_summary_title": "Riassunto",
            "pane_bullets_title": "Lista puntata",
            "copy_label": "Copia",
            "copy_done": "Copiato",
            "copy_transcription_aria": "Copia trascrizione",
            "copy_summary_aria": "Copia riassunto",
            "copy_bullets_aria": "Copia lista puntata",
            "transcription_empty": "Trascrizione vuota.",
            "summary_placeholder": "Riassunto non disponibile.",
            "bullets_placeholder": "Lista puntata non disponibile.",
            "default_audio_name": "audio",
            "language_label": "Lingua interfaccia",
            "label_record": "Registra audio (microfono):",
            "button_add_recording": "Aggiungi registrazione alla coda",
            "record_added": "Registrazione aggiunta alla coda: {name}",
            "record_missing": "Nessuna registrazione da aggiungere.",
        },
    },
    "en": {
        "prompts": {
            "summary": (
                "Provide a structured summary of the following text. "
                "Include only the key elements in a concise text. "
                "Do not use bullet points and do not add extra text.\n\n"
                "Text to summarize:\n{content}"
            ),
            "bullets": (
                "Create a bullet list with the essential elements of the following text. "
                "Short sentences, without extra text.\n\n"
                "Text:\n{content}"
            ),
        },
        "errors": {
            "ffmpeg_not_found": "ffmpeg not found for WAV conversion.",
            "wav_conversion_failed": "WAV conversion failed.",
            "file_not_found": "File not found.",
            "transcription_error": "Error during conversion/transcription: {error}",
        },
        "ui": {
            "markdown_header": (
                "# Transcribe audio!\n"
                "Upload one or more audio files. Files are transcribed sequentially!"
            ),
            "alert_ffmpeg_missing": (
                "<div class='alert'>WARNING: ffmpeg utility not found. "
                "Non-WAV files will not be converted!</div>"
            ),
            "label_files": "Recordings to transcribe:",
            "label_summary": "Create a structured summary from the transcribed text",
            "label_bullets": "Create a bullet list from the transcribed text",
            "button_transcribe": "Transcribe",
            "output_label": "Results",
            "empty_results": "No results to display.",
            "result_time": "Time: {seconds:.2f}s",
            "pane_transcription_title": "Transcription",
            "pane_summary_title": "Summary",
            "pane_bullets_title": "Bullet list",
            "copy_label": "Copy",
            "copy_done": "Copied",
            "copy_transcription_aria": "Copy transcription",
            "copy_summary_aria": "Copy summary",
            "copy_bullets_aria": "Copy bullet list",
            "transcription_empty": "Empty transcription.",
            "summary_placeholder": "Summary not available.",
            "bullets_placeholder": "Bullet list not available.",
            "default_audio_name": "audio",
            "language_label": "Interface language",
            "label_record": "Record audio (microphone):",
            "button_add_recording": "Add recording to queue",
            "record_added": "Recording added to the queue: {name}",
            "record_missing": "No recording to add.",
        },
    },
}

def _get_text(lang: str) -> dict:
    return TEXTS.get(lang, TEXTS[LANG_DEFAULT])

def _asr_language(lang: str) -> str:
    return "italian" if lang == "it" else "english"

def _base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

HERE = _base_dir()
MODEL_DIR = os.path.join(HERE, "whisper-large-v3")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE.startswith("cuda") else torch.float32

_PROCESSOR = WhisperProcessor.from_pretrained(MODEL_DIR)
_MODEL = WhisperForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    torch_dtype=TORCH_DTYPE,
).to(DEVICE)

_ASR = pipeline(
    "automatic-speech-recognition",
    model=_MODEL,
    tokenizer=_PROCESSOR.tokenizer,
    feature_extractor=_PROCESSOR.feature_extractor,
    device=0 if DEVICE.startswith("cuda") else -1,
)

_MODEL.generation_config.language = _asr_language(LANG_DEFAULT)
_MODEL.generation_config.task = "transcribe"
_MODEL.generation_config.forced_decoder_ids = None

_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None

def _ollama_summary(text: str, lang: str) -> str:
    prompt = _get_text(lang)["prompts"]["summary"].format(content=text)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
        return parsed.get("response", "").strip()
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return ""


def _ollama_bullets(text: str, lang: str) -> str:
    prompt = _get_text(lang)["prompts"]["bullets"].format(content=text)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            return parsed.get("response", "").strip()
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return ""

def _transcribe_file(
    filepath: str,
    summarize: bool,
    bullet_list: bool,
    lang: str,
    display_name: str | None = None,
) -> dict:
    start = time.perf_counter()
    result = _ASR(
        filepath,
        return_timestamps=True,
        generate_kwargs={"language": _asr_language(lang), "task": "transcribe"},
    )
    text = result.get("text", "").strip()
    summary = ""
    if summarize and text:
        summary = _ollama_summary(text, lang)
    bullets = ""
    if bullet_list and text:
        bullets = _ollama_bullets(text, lang)
    elapsed = time.perf_counter() - start
    return {
        "file": display_name or os.path.basename(filepath),
        "text": text,
        "summary": summary,
        "bullets": bullets,
        "seconds": elapsed,
    }

def _needs_wav_conversion(filepath: str) -> bool:
    _, ext = os.path.splitext(filepath)
    return ext.lower() != ".wav"

def _convert_to_temp_wav(filepath: str, lang: str) -> str:
    errors = _get_text(lang)["errors"]
    temp_file = tempfile.NamedTemporaryFile(prefix="whisper_tmp_", suffix=".wav", delete=False)
    temp_file.close()
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        filepath,
        "-ac",
        "1",
        "-ar",
        "16000",
        temp_file.name,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        raise RuntimeError(errors["ffmpeg_not_found"]) from exc
    except subprocess.CalledProcessError as exc:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        raise RuntimeError(errors["wav_conversion_failed"]) from exc
    return temp_file.name

def _render_results(
    items: list[dict],
    summarize_enabled: bool = False,
    bullets_enabled: bool = False,
    lang: str = LANG_DEFAULT,
) -> str:
    ui = _get_text(lang)["ui"]
    if not items:
        return ""
    cards = []
    for item in items:
        filename = html.escape(item.get("file", ui["default_audio_name"]))
        text_raw = item.get("text", "")
        summary_raw = item.get("summary", "")
        bullets_raw = item.get("bullets", "")
        text = html.escape(text_raw)
        text_attr = html.escape(text_raw, quote=True)
        summary = html.escape(summary_raw)
        summary_attr = html.escape(summary_raw, quote=True)
        bullets = html.escape(bullets_raw)
        bullets_attr = html.escape(bullets_raw, quote=True)
        copy_label = html.escape(ui["copy_label"])
        copy_done = html.escape(ui["copy_done"], quote=True)
        copy_transcription_aria = html.escape(ui["copy_transcription_aria"], quote=True)
        copy_summary_aria = html.escape(ui["copy_summary_aria"], quote=True)
        copy_bullets_aria = html.escape(ui["copy_bullets_aria"], quote=True)
        pane_transcription_title = html.escape(ui["pane_transcription_title"])
        pane_summary_title = html.escape(ui["pane_summary_title"])
        pane_bullets_title = html.escape(ui["pane_bullets_title"])
        transcription_empty = html.escape(ui["transcription_empty"])
        summary_placeholder = html.escape(ui["summary_placeholder"])
        bullets_placeholder = html.escape(ui["bullets_placeholder"])
        seconds = item.get("seconds", 0.0)
        result_time = html.escape(ui["result_time"].format(seconds=seconds))
        is_empty = not text_raw
        text_disabled_attr = "disabled" if is_empty else ""
        summary_disabled_attr = "disabled" if not summary_raw else ""
        bullets_disabled_attr = "disabled" if not bullets_raw else ""
        panes = [
            "<div class='result-pane'>"
            "<div class='pane-header'>"
            f"<div class='pane-title'>{pane_transcription_title}</div>"
            f"<button class='copy-btn' type='button' data-copy='{text_attr}' "
            f"data-copied-label='{copy_done}' "
            f"aria-label='{copy_transcription_aria}' {text_disabled_attr}>{copy_label}</button>"
            "</div>"
            f"<div class='pane-body'>{text or transcription_empty}</div>"
            "</div>"
        ]
        if summarize_enabled:
            panes.append(
                "<div class='result-pane'>"
                "<div class='pane-header'>"
                f"<div class='pane-title'>{pane_summary_title}</div>"
                f"<button class='copy-btn' type='button' data-copy='{summary_attr}' "
                f"data-copied-label='{copy_done}' "
                f"aria-label='{copy_summary_aria}' {summary_disabled_attr}>{copy_label}</button>"
                "</div>"
                f"<div class='pane-body'>{summary or summary_placeholder}</div>"
                "</div>"
            )
        if bullets_enabled:
            panes.append(
                "<div class='result-pane'>"
                "<div class='pane-header'>"
                f"<div class='pane-title'>{pane_bullets_title}</div>"
                f"<button class='copy-btn' type='button' data-copy='{bullets_attr}' "
                f"data-copied-label='{copy_done}' "
                f"aria-label='{copy_bullets_aria}' {bullets_disabled_attr}>{copy_label}</button>"
                "</div>"
                f"<div class='pane-body'>{bullets or bullets_placeholder}</div>"
                "</div>"
            )
        cards.append(
            "<div class='result-card'>"
            "<div class='result-header'>"
            f"<div class='result-title'>{filename}</div>"
            "</div>"
            f"<div class='result-meta'>{result_time}</div>"
            "<div class='result-body'>"
            f"{''.join(panes)}"
            "</div>"
            "</div>"
        )
    return "<div class='results-grid'>" + "".join(cards) + "</div>"

def _extract_recording_path(recording) -> str | None:
    if recording is None:
        return None
    if isinstance(recording, str):
        return recording
    if isinstance(recording, dict):
        for key in ("path", "name", "file"):
            value = recording.get(key)
            if value:
                return value
    return None

def _normalize_files(files: list[str] | None) -> list[str]:
    if not files:
        return []
    normalized = []
    for entry in files:
        if isinstance(entry, str):
            normalized.append(entry)
        elif isinstance(entry, dict):
            path = entry.get("path") or entry.get("name") or entry.get("file")
            if path:
                normalized.append(path)
    return normalized

def add_recording_to_queue(
    recording,
    files: list[str] | None,
    lang: str,
) -> tuple[list[str], str, None]:
    ui = _get_text(lang)["ui"]
    recording_path = _extract_recording_path(recording)
    if not recording_path:
        return _normalize_files(files), ui["record_missing"], None
    updated_files = _normalize_files(files)
    updated_files.append(recording_path)
    display_name = os.path.basename(recording_path)
    return updated_files, ui["record_added"].format(name=display_name), None

def transcribe_many(
    files: list[str] | None,
    summarize: bool,
    bullet_list: bool,
    lang: str,
) -> tuple[str, list[dict]]:
    if not files:
        return _render_results([], summarize, bullet_list, lang), []
    ui = _get_text(lang)["ui"]
    errors = _get_text(lang)["errors"]
    results = []
    for filepath in files:
        if not filepath or not os.path.exists(filepath):
            results.append(
                {
                    "file": os.path.basename(filepath or ui["default_audio_name"]),
                    "text": errors["file_not_found"],
                    "seconds": 0.0,
                }
            )
            continue
        original_name = os.path.basename(filepath)
        temp_wav = None
        try:
            if _needs_wav_conversion(filepath):
                temp_wav = _convert_to_temp_wav(filepath, lang)
                source_path = temp_wav
            else:
                source_path = filepath
            results.append(
                _transcribe_file(
                    source_path,
                    summarize,
                    bullet_list,
                    lang,
                    display_name=original_name,
                )
            )
        except Exception as exc:
            results.append(
                {
                    "file": original_name,
                    "text": errors["transcription_error"].format(error=exc),
                    "summary": "",
                    "bullets": "",
                    "seconds": 0.0,
                }
            )
        finally:
            if temp_wav and os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except OSError:
                    pass
    return _render_results(results, summarize, bullet_list, lang), results


CSS = """
:root {
  --bg: #0f1115;
  --surface: #171a21;
  --surface-2: #1f232d;
  --card: #1b1f28;
  --ink: #e6e8ee;
  --muted: #a0a7b4;
  --accent: #ccc;
  --accent-strong: #ddd;
  --border: #2b313b;
  --upload-bg: #d9dee5;
  --upload-ink: #1b1f28;
  --upload-border: #c7cdd6;
  --upload-muted: #5c6470;
  --shadow: 0 16px 36px rgba(0, 0, 0, 0.35);
  --font-body: "Segoe UI", "Helvetica Neue", "Noto Sans", "Liberation Sans", "Arial", sans-serif;
  --button-secondary-background-fill: #3b82f6;
}

.filename {
color: black !important;
}

* {
  box-sizing: border-box;
}

:root {
  color-scheme: dark;
}

body, .gradio-container {
  background-color: #dfe9ff;
  color: black;
  font-family: var(--font-body);
}

.gradio-container {
  min-height: 100vh;
}

.app-shell {
  max-width: 1100px;
  margin: 0 auto;
  padding-bottom: 32px;
}

.hero {
  padding: 20px 18px 8px 18px;
  color: black !important;
}

.top-row {
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}

.lang-col {
  display: flex;
  justify-content: flex-end;
  padding: 16px 12px 0 0;
}

.lang-col .gr-dropdown {
  min-width: 180px;
  max-width: 240px;
}

.hero h1 {
  font-size: clamp(1.7rem, 2.5vw, 2.4rem);
  letter-spacing: 0.01em;
  margin-bottom: 6px;
  color: black !important;
}

.hero p {
  color: black !important;
  font-size: clamp(0.98rem, 1.4vw, 1.08rem);
}

.controls {
  border: 1px solid var(--border);
  background: var(--surface);
  border-radius: 16px;
  padding: 16px;
  box-shadow: var(--shadow);
  backdrop-filter: blur(6px);
}

.alert {
  border-radius: 12px;
  padding: 10px 12px;
  border: 1px solid #3a2b1b;
  background: rgba(255, 166, 77, 0.08);
  color: #f0c19b;
  font-size: 0.92rem;
  margin-bottom: 12px;
}

.gradio-container label,
.gradio-container .gr-form label {
  color: black;
  font-weight: 600;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  background-color: #3b82f6;
  color: var(--ink);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding-left:10px;
  padding-right:10px;
}

.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 2px rgba(102, 178, 255, 0.2);
}

.gradio-container .gr-file,
.gradio-container .file,
.gradio-container .file-upload,
.gradio-container .file-preview,
.gradio-container .upload-box {
  background: var(--upload-bg);
  border: 1px dashed var(--upload-border);
  border-radius: 14px;
  color: white !important;
}

.gradio-container .file-preview,
.gradio-container .file-preview-row,
.gradio-container .file-preview-table {
  background: #eef1f5;
  color: var(--upload-ink);
}

.gradio-container .file-preview-row *,
.gradio-container .file-preview-table * {
  color: var(--upload-ink);
}

.gradio-container .file-preview-row .file-name,
.gradio-container .file-preview-row .file-size {
  color: var(--upload-ink);
}

.gradio-container .file-upload svg,
.gradio-container .upload-box svg,
.gradio-container .file-preview svg {
  color: var(--upload-muted);
  fill: currentColor;
}

.gradio-container .file-upload button,
.gradio-container .file-preview button,
.gradio-container .upload-btn,
.gradio-container .file-clear-button,
.gradio-container .file-clear {
  background: #e3e7ee;
  color: var(--upload-ink);
  border: 1px solid var(--upload-border);
  border-radius: 10px;
  box-shadow: none;
}

.gradio-container .file-upload button:hover,
.gradio-container .file-preview button:hover,
.gradio-container .upload-btn:hover,
.gradio-container .file-clear-button:hover,
.gradio-container .file-clear:hover {
  background: #d5dbe3;
}

.gradio-container .gr-checkbox,
.gradio-container .gr-checkbox label {
  color: var(--ink);
  font-weight: 600;
}

.gradio-container input[type="checkbox"] {
  accent-color: var(--accent);
  width: 16px;
  height: 16px;
}

.gradio-container .gr-button,
.gradio-container button {
  color: #0b0d12;
  border: none;
  border-radius: 12px;
  font-weight: 700;
  box-shadow: 0 10px 24px rgba(61, 139, 253, 0.35);
  transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}

.gradio-container .gr-button:hover,
.gradio-container button:hover {
  background: var(--accent-strong);
  transform: translateY(-1px);
  box-shadow: 0 14px 30px rgba(61, 139, 253, 0.45);
}

.results-grid {
  display: grid;
  gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
}

.result-card {
  background: #4d4d4d;
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px 14px 14px 14px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-height: 160px;
  box-shadow: 0 10px 22px rgba(0, 0, 0, 0.25);
  animation: fadeUp 0.4s ease-out;
}

.result-header {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  justify-content: space-between;
}

.result-title {
  font-weight: 700;
  font-size: 1.05rem;
  color: white;
}

.copy-btn {
  background: #2a3140;
  color: var(--ink);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 4px 10px;
  font-size: 0.78rem;
  font-weight: 600;
  cursor: pointer;
  box-shadow: none;
  transition: background 0.2s ease, border-color 0.2s ease;
}

.copy-btn:hover {
  background: #333c50;
  border-color: #3f485c;
}

.copy-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.copy-btn.copied {
  background: #dfe7f2;
  color: #141720;
  border-color: #c7cdd6;
}

.result-meta {
  font-size: 0.85rem;
  color: var(--muted);
}

.result-body {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
}

.result-pane {
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px 12px 12px 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.pane-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.pane-title {
  font-weight: 700;
  font-size: 0.9rem;
  color: #e2e8f0;
}

.pane-body {
  white-space: pre-wrap;
  font-size: 0.95rem;
  line-height: 1.4;
  color: #d6dbe4;
}

.empty {
  padding: 16px;
  border-radius: 12px;
  background: var(--surface-2);
  border: 1px dashed var(--border);
  color: var(--muted);
  text-align: center;
}

.form {
  background-color:white;
}

.lang-select {
  max-width:250px;
  position: absolute;
  right: 0px;
}

.mic-select {
  color: black;
}

.pause-button {
  background-color:white;
}

footer {
    display: none;
}

@keyframes fadeUp {
  from { transform: translateY(8px); opacity: 0; }
  to { transform: translateY(0); opacity: 1; }
}

@media (max-width: 640px) {
  .top-row {
    flex-direction: column;
    align-items: stretch;
  }
  .lang-col {
    justify-content: flex-start;
    padding: 0 12px 0 12px;
  }
  .controls {
    padding: 12px;
  }
  .hero {
    padding: 12px 12px 2px 12px;
  }
  .result-body {
    grid-template-columns: 1fr;
  }
}
"""

COPY_JS = """
const copyText = (btn) => {
  if (!btn) return;
  const text = btn.getAttribute('data-copy') || '';
  if (!text) return;
  const resetLabel = btn.textContent;
  const markDone = () => {
    const doneLabel = btn.getAttribute('data-copied-label') || resetLabel;
    btn.classList.add('copied');
    btn.textContent = doneLabel;
    window.setTimeout(() => {
      btn.textContent = resetLabel;
      btn.classList.remove('copied');
    }, 1400);
  };
  const fallback = () => {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'absolute';
    ta.style.backgroundColor = '#303030';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); } catch (err) {}
    document.body.removeChild(ta);
    markDone();
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(markDone).catch(fallback);
  } else {
    fallback();
  }
};

element.addEventListener('click', (evt) => {
  const target = evt.target && evt.target.closest ? evt.target.closest('button') : null;
  if (!target || !element.contains(target)) return;
  if (target.classList.contains('copy-btn')) {
    evt.preventDefault();
    copyText(target);
  }
});
"""

def _apply_language(
    lang: str,
    files: list[str] | None,
    summarize: bool,
    bullet_list: bool,
    results: list[dict],
):
    _MODEL.generation_config.language = _asr_language(lang)
    ui = _get_text(lang)["ui"]
    header_value = ui["markdown_header"]
    alert_update = gr.update(
        value=ui["alert_ffmpeg_missing"],
        visible=not _FFMPEG_AVAILABLE,
    )
    file_update = gr.update(label=ui["label_files"], value=files)
    summary_update = gr.update(label=ui["label_summary"], value=summarize)
    bullets_update = gr.update(label=ui["label_bullets"], value=bullet_list)
    button_update = gr.update(value=ui["button_transcribe"])
    output_update = gr.update(
        label=ui["output_label"],
        value=_render_results(results or [], summarize, bullet_list, lang),
    )
    lang_update = gr.update(label=ui["language_label"], value=lang)
    record_update = gr.update(label=ui["label_record"])
    add_record_update = gr.update(value=ui["button_add_recording"])
    record_status_update = gr.update(value="")
    return (
        header_value,
        alert_update,
        file_update,
        summary_update,
        bullets_update,
        button_update,
        output_update,
        lang_update,
        record_update,
        add_record_update,
        record_status_update,
    )

with gr.Blocks() as demo:
    ui_default = _get_text(LANG_DEFAULT)["ui"]
    results_state = gr.State([])
    with gr.Column(elem_classes=["app-shell"]):
        with gr.Row(elem_classes=["top-row"]):
            with gr.Column(elem_classes=["hero"]):
                header_md = gr.Markdown(ui_default["markdown_header"])
            with gr.Column(elem_classes=["lang-col"]):
                lang_select = gr.Dropdown(
                    label=ui_default["language_label"],
                    choices=LANG_CHOICES,
                    value=LANG_DEFAULT,
                    elem_classes=["lang-select"],
                )
        with gr.Column(elem_classes=["controls"]):
            ffmpeg_alert = gr.Markdown(
                value=ui_default["alert_ffmpeg_missing"],
                visible=not _FFMPEG_AVAILABLE,
            )
            file_input = gr.Files(
                label=ui_default["label_files"],
                file_types=["audio"],
                type="filepath",
            )
            record_input = gr.Audio(
                label=ui_default["label_record"],
                sources=["microphone"],
                type="filepath",
            )
            add_record_btn = gr.Button(ui_default["button_add_recording"], variant="primary")
            record_status = gr.Markdown(value="")
            summary_input = gr.Checkbox(
                value=False,
                label=ui_default["label_summary"],
            )
            bullets_input = gr.Checkbox(
                value=False,
                label=ui_default["label_bullets"],
            )
            transcribe_btn = gr.Button(ui_default["button_transcribe"], variant="primary")
        output_html = gr.HTML(label=ui_default["output_label"], js_on_load=COPY_JS)

    transcribe_btn.click(
        fn=transcribe_many,
        inputs=[file_input, summary_input, bullets_input, lang_select],
        outputs=[output_html, results_state],
    )
    add_record_btn.click(
        fn=add_recording_to_queue,
        inputs=[record_input, file_input, lang_select],
        outputs=[file_input, record_status, record_input],
    )
    lang_select.change(
        fn=_apply_language,
        inputs=[lang_select, file_input, summary_input, bullets_input, results_state],
        outputs=[
            header_md,
            ffmpeg_alert,
            file_input,
            summary_input,
            bullets_input,
            transcribe_btn,
            output_html,
            lang_select,
            record_input,
            add_record_btn,
            record_status,
        ],
    )


if __name__ == "__main__":
    def _pick_port() -> int:
        env_port = os.getenv("WHISPER_WEBAPP_PORT")
        if env_port:
            try:
                return int(env_port)
            except ValueError:
                pass
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    port = _pick_port()
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=False,
        css=CSS,
        theme=gr.themes.Soft(primary_hue="blue"),
    )
