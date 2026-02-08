import os
import threading
import time
import uuid
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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe")
TRANSLATE_MODEL = os.getenv("OPENAI_TRANSLATE_MODEL", "gpt-4.1-mini")
SOURCE_LANGUAGE = os.getenv("SOURCE_LANGUAGE", "ar")
TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "en")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

client = None
client_lock = threading.Lock()
jobs = {}
jobs_lock = threading.Lock()


def get_client() -> OpenAI:
    global client

    if OPENAI_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing Python package 'openai'. Run: .venv/bin/python -m pip install -r requirements.txt"
        )

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


def transcribe_audio(path: str) -> str:
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


def translate_text(text: str) -> str:
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


def update_job(job_id: str, **fields):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job.update(fields)


def update_job_progress(job_id: str, worked_seconds: float, total_work_seconds: float):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        progress = max(0.0, min(100.0, (worked_seconds / total_work_seconds) * 100.0))
        elapsed = max(0.0, time.monotonic() - job["started_at"])
        eta = None
        if progress > 0:
            eta = max(0.0, elapsed * ((100.0 - progress) / progress))
        job["progress_percent"] = round(progress, 2)
        job["elapsed_seconds"] = round(elapsed, 2)
        job["eta_seconds"] = None if eta is None else round(eta, 2)
        job["worked_seconds"] = round(worked_seconds, 2)
        job["total_work_seconds"] = round(total_work_seconds, 2)


def process_job(job_id: str, audio_path: str):
    try:
        update_job(job_id, status="processing", phase="checking_api_key")
        get_client()
        update_job_progress(job_id, worked_seconds=5.0, total_work_seconds=100.0)

        update_job(job_id, phase="transcribing")
        transcript_text = transcribe_audio(audio_path)
        update_job_progress(job_id, worked_seconds=70.0, total_work_seconds=100.0)

        update_job(job_id, phase="translating")
        translation_text = translate_text(transcript_text)
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
        jobs[job_id] = {
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
            "started_at": time.monotonic(),
        }

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
            return JSONResponse({"error": "job not found"}, status_code=404)

        payload = {
            "job_id": job["job_id"],
            "status": job["status"],
            "phase": job["phase"],
            "progress_percent": job["progress_percent"],
            "elapsed_seconds": job["elapsed_seconds"],
            "eta_seconds": job["eta_seconds"],
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
        }
    return JSONResponse(payload)
