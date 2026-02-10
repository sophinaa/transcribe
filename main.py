import os
import threading
import time
import uuid
import json
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

try:
    from openai import APIError, APITimeoutError, OpenAI, RateLimitError
    OPENAI_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - runtime environment guard
    APIError = Exception
    APITimeoutError = Exception
    RateLimitError = Exception
    OpenAI = None
    OPENAI_IMPORT_ERROR = e

try:
    from faster_whisper import WhisperModel
    WHISPER_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - runtime environment guard
    WhisperModel = None
    WHISPER_IMPORT_ERROR = e

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe")
TRANSLATE_MODEL = os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4.1-mini")
ROLE_LABEL_MODEL = os.getenv("OPENAI_ROLE_LABEL_MODEL", "gpt-4.1-mini")
SOURCE_LANGUAGE = os.getenv("SOURCE_LANGUAGE", "ar")
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "en")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "medium")
LOCAL_WHISPER_DEVICE = os.getenv("LOCAL_WHISPER_DEVICE", "cpu")
LOCAL_WHISPER_COMPUTE_TYPE = os.getenv("LOCAL_WHISPER_COMPUTE_TYPE", "int8")
PREFERRED_PROVIDER = os.getenv("TRANSCRIPTION_PROVIDER", "auto").lower()
JOB_STORE_DIR = Path(os.getenv("JOB_STORE_DIR", "/tmp/transcription_jobs"))
APP_STARTED_AT_EPOCH = time.time()

client = None
client_lock = threading.Lock()
local_model = None
local_model_lock = threading.Lock()
jobs = {}
jobs_lock = threading.Lock()
JOB_STORE_DIR.mkdir(parents=True, exist_ok=True)


def job_store_path(job_id: str) -> Path:
    return JOB_STORE_DIR / f"{job_id}.json"


def persist_job(job: dict):
    path = job_store_path(job["job_id"])
    path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")


def load_job(job_id: str) -> dict | None:
    path = job_store_path(job_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def mark_stale_processing_job_as_failed(job: dict) -> dict:
    if job.get("status") != "processing":
        return job
    updated_at = float(job.get("updated_at", 0.0) or 0.0)
    if updated_at >= APP_STARTED_AT_EPOCH:
        return job
    job["status"] = "error"
    job["phase"] = "failed"
    job["error"] = "Server restarted during processing. Please re-upload the file."
    job["updated_at"] = time.time()
    return job


def get_provider() -> str:
    has_openai = OPENAI_IMPORT_ERROR is None and bool(os.getenv("OPENAI_API_KEY"))
    has_whisper = WHISPER_IMPORT_ERROR is None

    if PREFERRED_PROVIDER == "openai":
        if has_openai:
            return "openai"
        raise RuntimeError("TRANSCRIPTION_PROVIDER=openai but OpenAI client is unavailable.")

    if PREFERRED_PROVIDER == "local":
        if has_whisper:
            return "local"
        raise RuntimeError("TRANSCRIPTION_PROVIDER=local but faster-whisper is unavailable.")

    if has_openai:
        return "openai"
    if has_whisper:
        return "local"
    raise RuntimeError(
        "No transcription backend available. Install openai + set OPENAI_API_KEY or install faster-whisper."
    )


def get_client():
    global client

    if OPENAI_IMPORT_ERROR is not None:
        raise RuntimeError("Missing Python package 'openai'.")

    if client is not None:
        return client

    with client_lock:
        if client is not None:
            return client

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
        client = OpenAI(api_key=api_key)
        return client


def get_local_model():
    global local_model

    if WHISPER_IMPORT_ERROR is not None:
        raise RuntimeError("Missing Python package 'faster-whisper'.")

    if local_model is not None:
        return local_model

    with local_model_lock:
        if local_model is not None:
            return local_model
        local_model = WhisperModel(
            LOCAL_WHISPER_MODEL,
            device=LOCAL_WHISPER_DEVICE,
            compute_type=LOCAL_WHISPER_COMPUTE_TYPE,
        )
        return local_model


def transcribe_audio_openai(path: str) -> str:
    local_client = get_client()
    with open(path, "rb") as audio_file:
        result = local_client.audio.transcriptions.create(
            model=TRANSCRIBE_MODEL,
            file=audio_file,
            language=SOURCE_LANGUAGE,
            response_format="text",
        )

    if isinstance(result, str):
        return result.strip()
    return str(result).strip()


def translate_text_openai(text: str) -> str:
    if not text:
        return ""
    if SOURCE_LANGUAGE == TARGET_LANGUAGE:
        return text

    local_client = get_client()
    response = local_client.responses.create(
        model=TRANSLATE_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a professional translator. Translate the user's transcript "
                    "faithfully and naturally. Return only the translated text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Translate this transcript from {SOURCE_LANGUAGE} to {TARGET_LANGUAGE}:\n\n{text}"
                ),
            },
        ],
    )

    output_text = getattr(response, "output_text", "")
    return output_text.strip() if output_text else ""


def apply_speaker_labels_openai(text: str, language_name: str) -> str:
    if not text.strip():
        return text

    local_client = get_client()
    response = local_client.responses.create(
        model=ROLE_LABEL_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "Format the transcript with speaker labels only. "
                    "Use exactly these prefixes: "
                    "'Speaker 1 (Interviewer):' and 'Speaker 2 (Interviewee):'. "
                    "Keep wording and language unchanged. "
                    "Do not summarize. Do not add commentary."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Label this {language_name} transcript with speaker turns:\n\n{text}"
                ),
            },
        ],
    )
    output_text = getattr(response, "output_text", "")
    return output_text.strip() if output_text else text


def transcribe_audio_local(path: str, job_id: str | None = None) -> str:
    model = get_local_model()
    segments, info = model.transcribe(path, task="transcribe", language=SOURCE_LANGUAGE, vad_filter=True)
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    text_parts = []

    for seg in segments:
        text_parts.append(seg.text)
        if job_id and duration > 0:
            seg_end = float(getattr(seg, "end", 0.0) or 0.0)
            fraction = max(0.0, min(1.0, seg_end / duration))
            # Keep transcription phase between 10% and 70%.
            progress = 10.0 + (fraction * 60.0)
            update_job_progress(job_id, worked_seconds=progress, total_work_seconds=100.0)

    return "".join(text_parts).strip()


def translate_text_local(path: str, fallback_text: str, job_id: str | None = None) -> str:
    if SOURCE_LANGUAGE == TARGET_LANGUAGE:
        return fallback_text
    if TARGET_LANGUAGE != "en":
        return fallback_text
    model = get_local_model()
    segments, info = model.transcribe(path, task="translate", language=SOURCE_LANGUAGE, vad_filter=True)
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    text_parts = []

    for seg in segments:
        text_parts.append(seg.text)
        if job_id and duration > 0:
            seg_end = float(getattr(seg, "end", 0.0) or 0.0)
            fraction = max(0.0, min(1.0, seg_end / duration))
            # Keep translation phase between 70% and 98%.
            progress = 70.0 + (fraction * 28.0)
            update_job_progress(job_id, worked_seconds=progress, total_work_seconds=100.0)

    return "".join(text_parts).strip()


def update_job(job_id: str, **fields):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job.update(fields)
        job["updated_at"] = time.time()
        persist_job(job)


def update_job_progress(job_id: str, worked_seconds: float, total_work_seconds: float):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        progress = max(0.0, min(100.0, (worked_seconds / total_work_seconds) * 100.0))
        # Never let progress move backwards.
        progress = max(float(job.get("progress_percent", 0.0) or 0.0), progress)
        elapsed = max(0.0, time.monotonic() - job["started_at"])
        eta = None
        if progress > 0:
            eta = max(0.0, elapsed * ((100.0 - progress) / progress))
        job["progress_percent"] = round(progress, 2)
        job["elapsed_seconds"] = round(elapsed, 2)
        job["eta_seconds"] = None if eta is None else round(eta, 2)
        job["worked_seconds"] = round(worked_seconds, 2)
        job["total_work_seconds"] = round(total_work_seconds, 2)
        job["updated_at"] = time.time()
        persist_job(job)


def process_job(job_id: str, audio_path: str):
    try:
        provider = get_provider()
        update_job(job_id, status="processing", phase=f"initializing_{provider}", provider=provider)
        if provider == "openai":
            get_client()
        else:
            get_local_model()
        update_job_progress(job_id, worked_seconds=10.0, total_work_seconds=100.0)

        update_job(job_id, phase="transcribing")
        if provider == "openai":
            transcript_text = transcribe_audio_openai(audio_path)
        else:
            transcript_text = transcribe_audio_local(audio_path, job_id=job_id)
        update_job_progress(job_id, worked_seconds=70.0, total_work_seconds=100.0)

        update_job(job_id, phase="translating")
        if provider == "openai":
            translation_text = translate_text_openai(transcript_text)
        else:
            translation_text = translate_text_local(audio_path, transcript_text, job_id=job_id)
        update_job_progress(job_id, worked_seconds=88.0, total_work_seconds=100.0)

        update_job(job_id, phase="labeling_speakers")
        try:
            if OPENAI_IMPORT_ERROR is None and bool(os.getenv("OPENAI_API_KEY")):
                transcript_text = apply_speaker_labels_openai(transcript_text, SOURCE_LANGUAGE)
                translation_text = apply_speaker_labels_openai(translation_text, TARGET_LANGUAGE)
        except Exception:
            # Keep original text if speaker labeling fails.
            pass
        update_job_progress(job_id, worked_seconds=100.0, total_work_seconds=100.0)

        update_job(
            job_id,
            status="done",
            phase="complete",
            transcript=transcript_text,
            translation=translation_text,
            # Backward-compatible keys for the current frontend payload shape.
            arabic_transcript=transcript_text,
            english_translation=translation_text,
        )
    except RateLimitError:
        update_job(job_id, status="error", phase="failed", error="OpenAI rate limit reached. Please retry shortly.")
    except APITimeoutError:
        update_job(job_id, status="error", phase="failed", error="OpenAI request timed out. Please retry.")
    except APIError as e:
        update_job(job_id, status="error", phase="failed", error=f"OpenAI API error: {str(e)}")
    except Exception as e:
        update_job(job_id, status="error", phase="failed", error=str(e))
    finally:
        try:
            os.remove(audio_path)
        except OSError:
            pass


@app.post("/process")
async def process_audio(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower() or ".bin"
    job_id = str(uuid.uuid4())
    raw_path = f"/tmp/{job_id}{ext}"

    content = await file.read()
    if not content:
        return JSONResponse({"error": "Uploaded file is empty."}, status_code=400)
    if len(content) > MAX_UPLOAD_BYTES:
        return JSONResponse({"error": f"File too large. Max size is {MAX_UPLOAD_MB} MB."}, status_code=413)

    with open(raw_path, "wb") as out:
        out.write(content)

    with jobs_lock:
        new_job = {
            "job_id": job_id,
            "status": "queued",
            "phase": "queued",
            "progress_percent": 0.0,
            "elapsed_seconds": 0.0,
            "eta_seconds": None,
            "worked_seconds": 0.0,
            "total_work_seconds": None,
            "audio_duration_seconds": None,
            "transcript": "",
            "translation": "",
            "arabic_transcript": "",
            "english_translation": "",
            "error": None,
            "model_name": TRANSCRIBE_MODEL,
            "translation_model": TRANSLATE_MODEL,
            "provider": None,
            "started_at": time.monotonic(),
            "updated_at": time.time(),
        }
        jobs[job_id] = new_job
        persist_job(new_job)

    worker = threading.Thread(target=process_job, args=(job_id, raw_path), daemon=True)
    worker.start()
    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.get("/")
def home():
    index_path = Path(__file__).with_name("index.html")
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"status": "ok", "message": "Backend is running."})


@app.get("/progress/{job_id}")
def get_progress(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            loaded = load_job(job_id)
            if loaded:
                jobs[job_id] = loaded
                job = loaded
        if not job:
            return JSONResponse({"error": "job not found"}, status_code=404)
        job = mark_stale_processing_job_as_failed(job)
        jobs[job_id] = job
        persist_job(job)

        now = time.time()
        runtime_elapsed = round(max(0.0, time.monotonic() - job["started_at"]), 2)
        progress_percent = float(job.get("progress_percent", 0.0) or 0.0)
        eta_seconds = job.get("eta_seconds")

        # Heartbeat progress for long local decode gaps so UI does not appear frozen.
        if job.get("status") == "processing":
            phase = job.get("phase")
            phase_caps = {
                "initializing_openai": 15.0,
                "initializing_local": 15.0,
                "transcribing": 69.5,
                "translating": 87.5,
                "labeling_speakers": 98.5,
            }
            cap = phase_caps.get(phase)
            if cap is not None:
                updated_at = float(job.get("updated_at", now) or now)
                idle_seconds = max(0.0, now - updated_at)
                if idle_seconds > 2.0 and progress_percent < cap:
                    progress_percent = min(cap, progress_percent + ((idle_seconds - 2.0) * 0.08))

            # Ensure response progress is never lower than stored progress.
            progress_percent = max(float(job.get("progress_percent", 0.0) or 0.0), progress_percent)

            if progress_percent > 0:
                eta_seconds = round(max(0.0, runtime_elapsed * ((100.0 - progress_percent) / progress_percent)), 2)
            else:
                eta_seconds = None

        payload = {
            "job_id": job["job_id"],
            "status": job["status"],
            "phase": job["phase"],
            "progress_percent": round(progress_percent, 2),
            "elapsed_seconds": runtime_elapsed,
            "eta_seconds": eta_seconds,
            "worked_seconds": job["worked_seconds"],
            "total_work_seconds": job["total_work_seconds"],
            "audio_duration_seconds": job["audio_duration_seconds"],
            "transcript": job["transcript"],
            "translation": job["translation"],
            "arabic_transcript": job["arabic_transcript"],
            "english_translation": job["english_translation"],
            "error": job["error"],
            "model_name": job["model_name"],
            "translation_model": job["translation_model"],
            "provider": job["provider"],
        }
    return JSONResponse(payload)
