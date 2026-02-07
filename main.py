import os
import uuid
import time
import threading
import subprocess
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

app = FastAPI()

MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")
model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
jobs = {}
jobs_lock = threading.Lock()

def convert_to_wav(input_path: str, output_path: str):
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", output_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_audio_duration_seconds(input_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())

def run_whisper_with_progress(job_id: str, wav_path: str, task: str, duration_seconds: float, phase_offset: float):
    segments, _info = model.transcribe(
        wav_path,
        task=task,          # "transcribe" or "translate"
        language="ar",      # Arabic (works fine for Egyptian dialect)
        vad_filter=True
    )
    transcript = []
    for seg in segments:
        transcript.append(seg.text)
        seg_end = max(0.0, min(seg.end or 0.0, duration_seconds))
        worked_seconds = phase_offset + seg_end
        update_job_progress(job_id, worked_seconds=worked_seconds, total_work_seconds=max(1.0, duration_seconds * 2.0))
    return "".join(transcript).strip()

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

def process_job(job_id: str, raw_path: str, wav_path: str):
    try:
        update_job(job_id, status="processing", phase="converting")
        convert_to_wav(raw_path, wav_path)
        duration_seconds = get_audio_duration_seconds(wav_path)

        update_job(
            job_id,
            phase="transcribing_arabic",
            audio_duration_seconds=round(duration_seconds, 2),
        )
        arabic_text = run_whisper_with_progress(job_id, wav_path, task="transcribe", duration_seconds=duration_seconds, phase_offset=0.0)

        update_job(job_id, phase="translating_english")
        english_text = run_whisper_with_progress(
            job_id,
            wav_path,
            task="translate",
            duration_seconds=duration_seconds,
            phase_offset=duration_seconds
        )

        update_job_progress(job_id, worked_seconds=duration_seconds * 2.0, total_work_seconds=max(1.0, duration_seconds * 2.0))
        update_job(
            job_id,
            status="done",
            phase="complete",
            arabic_transcript=arabic_text,
            english_translation=english_text
        )
    except subprocess.CalledProcessError:
        update_job(job_id, status="error", phase="failed", error="FFmpeg/ffprobe failed while processing audio.")
    except Exception as e:
        update_job(job_id, status="error", phase="failed", error=str(e))
    finally:
        for p in [raw_path, wav_path]:
            try:
                os.remove(p)
            except:
                pass

@app.post("/process")
async def process_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower() or ".bin"
    job_id = str(uuid.uuid4())
    raw_path = f"/tmp/{job_id}{ext}"
    wav_path = f"/tmp/{job_id}.wav"

    with open(raw_path, "wb") as f:
        f.write(await file.read())

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
            "arabic_transcript": "",
            "english_translation": "",
            "error": None,
            "started_at": time.monotonic(),
        }

    background_tasks.add_task(process_job, job_id, raw_path, wav_path)
    return JSONResponse({"job_id": job_id, "status": "queued"})

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
            "arabic_transcript": job["arabic_transcript"],
            "english_translation": job["english_translation"],
            "error": job["error"],
        }
    return JSONResponse(payload)
