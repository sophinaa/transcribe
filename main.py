import os
import uuid
import subprocess
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel

app = FastAPI()

MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")
model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")

def convert_to_wav(input_path: str, output_path: str):
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", "16000", output_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def run_whisper(wav_path: str, task: str):
    segments, _info = model.transcribe(
        wav_path,
        task=task,          # "transcribe" or "translate"
        language="ar",      # Arabic (works fine for Egyptian dialect)
        vad_filter=True
    )
    return "".join(seg.text for seg in segments).strip()

@app.post("/process")
async def process_audio(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower() or ".bin"
    job_id = str(uuid.uuid4())
    raw_path = f"/tmp/{job_id}{ext}"
    wav_path = f"/tmp/{job_id}.wav"

    with open(raw_path, "wb") as f:
        f.write(await file.read())

    try:
        convert_to_wav(raw_path, wav_path)
        arabic_text = run_whisper(wav_path, task="transcribe")
        english_text = run_whisper(wav_path, task="translate")

        return JSONResponse({
            "job_id": job_id,
            "arabic_transcript": arabic_text,
            "english_translation": english_text
        })

    except subprocess.CalledProcessError:
        return JSONResponse({"error": "FFmpeg failed to convert audio."}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        for p in [raw_path, wav_path]:
            try:
                os.remove(p)
            except:
                pass

