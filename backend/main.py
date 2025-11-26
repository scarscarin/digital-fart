import asyncio
import json
import random
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional, Set

import aiofiles
import numpy as np
import soundfile as sf
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "audio"
STATIC_DIR = BASE_DIR / "static"
GENERATED_AUDIO_PATH = STATIC_DIR / "generated_fart.wav"

AUDIO_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

CPU_POWER_W = 50.0
GRID_INTENSITY_G_PER_KWH = 400.0
CO2_PER_FART_G = 0.0160137
TRAIN_DURATION_SECONDS = 600.0  # 10 minutes of progress updates
PROGRESS_INTERVAL_SECONDS = 1.0  # update clients roughly every second

app = FastAPI(title="digitalfart.net API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

connected_clients: Set[WebSocket] = set()
training_task: Optional[asyncio.Task] = None


def count_audio_files() -> int:
    return len([p for p in AUDIO_DIR.iterdir() if p.is_file()])


def compute_emissions(elapsed_seconds: float):
    energy_wh = CPU_POWER_W * elapsed_seconds / 3600.0
    energy_kwh = energy_wh / 1000.0
    co2_g = energy_kwh * GRID_INTENSITY_G_PER_KWH
    equivalent_farts = co2_g / CO2_PER_FART_G
    return energy_wh, co2_g, equivalent_farts


def convert_webm_to_wav(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "22050",
        str(dst),
    ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src}: {completed.stderr.decode(errors='ignore')}")


def load_farts_to_grains():
    grains: List[np.ndarray] = []
    sr = 22050
    grain_samples = int(0.03 * sr)
    hop = max(1, grain_samples // 2)

    with TemporaryDirectory() as tmpdir:
        for path in AUDIO_DIR.glob("*.webm"):
            wav_path = Path(tmpdir) / f"{path.stem}.wav"
            convert_webm_to_wav(path, wav_path)
            data, sr = sf.read(wav_path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            for start in range(0, max(0, len(data) - grain_samples + 1), hop):
                grain = data[start : start + grain_samples]
                if len(grain) == grain_samples:
                    grains.append(grain.astype(np.float32))
    return np.array(grains), sr


def generate_fart_from_grains(grains: np.ndarray, sr: int = 22050, duration_seconds: float = 5.0):
    total_samples = int(sr * duration_seconds)
    if grains.size == 0:
        return np.zeros(total_samples, dtype=np.float32)

    fade_len = 192
    output = np.zeros(total_samples, dtype=np.float32)
    position = 0

    while position < total_samples:
        grain = random.choice(grains).astype(np.float32)
        if len(grain) < fade_len * 2:
            fade = max(1, len(grain) // 4)
        else:
            fade = fade_len
        fade_in = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        grain[:fade] *= fade_in
        grain[-fade:] *= fade_out

        end = min(position + len(grain), total_samples)
        chunk = grain[: end - position]
        output[position:end] += chunk
        position += len(chunk)

    max_amp = np.max(np.abs(output))
    if max_amp > 1e-6:
        output = output / max_amp * 0.9
    return output


async def broadcast(payload: dict) -> None:
    to_remove: List[WebSocket] = []
    for ws in connected_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            to_remove.append(ws)
    for ws in to_remove:
        connected_clients.discard(ws)

    total = count_audio_files()
    return {"success": True, "filename": filename, "count": total}

@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    filename = f"fart_{int(time.time() * 1000)}.webm"
    save_path = AUDIO_DIR / filename

    try:
        async with aiofiles.open(save_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                await buffer.write(chunk)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save audio: {exc}")

    total = count_audio_files()
    return {"success": True, "filename": filename, "count": total}


async def _run_training():
    start_time = time.perf_counter()
    await broadcast(
        {
            "type": "training-start",
            "video": "https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4",
            "target_seconds": TRAIN_DURATION_SECONDS,
        }
    )

    loop = asyncio.get_running_loop()
    try:
        # Emit a steady stream of progress updates for the full 10-minute window.
        while True:
            elapsed = time.perf_counter() - start_time
            progress = min(elapsed / TRAIN_DURATION_SECONDS, 1.0)
            _, co2_g, equivalent_farts = compute_emissions(elapsed)
            await broadcast(
                {
                    "type": "training-progress",
                    "progress": progress,
                    "co2_g": co2_g,
                    "equivalent_farts": equivalent_farts,
                    "elapsed": elapsed,
                }
            )
            if progress >= 1.0:
                break
            await asyncio.sleep(PROGRESS_INTERVAL_SECONDS)

        # After the timed progress loop, do the actual synthesis work.
        grains, sr = await loop.run_in_executor(None, load_farts_to_grains)
        fart_audio = await loop.run_in_executor(None, generate_fart_from_grains, grains, sr, 5.0)
        await loop.run_in_executor(None, sf.write, GENERATED_AUDIO_PATH, fart_audio, sr)

        elapsed = time.perf_counter() - start_time
        energy_wh, co2_g, equivalent_farts = compute_emissions(elapsed)
        await broadcast(
            {
                "type": "training-complete",
                "runtime_seconds": elapsed,
                "energy_Wh": energy_wh,
                "co2_g": co2_g,
                "equivalent_farts": equivalent_farts,
                "audio": "/static/generated_fart.wav",
            }
        )
    except Exception as exc:  # pragma: no cover - broadcast failures to clients
        await broadcast({"type": "training-error", "message": str(exc)})

@app.post("/start-training")
async def start_training(background_tasks: BackgroundTasks):
    global training_task
    if training_task and not training_task.done():
        raise HTTPException(status_code=400, detail="Training already running")

@app.post("/start-training")
async def start_training(background_tasks: BackgroundTasks):
    global training_task
    if training_task and not training_task.done():
        raise HTTPException(status_code=400, detail="Training already running")

    training_task = asyncio.create_task(_run_training())
    return {"status": "started"}


@app.get("/audio-count")
async def audio_count():
    return {"count": count_audio_files()}


@app.websocket("/ws/events")
async def events(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data:
                # Clients can ping to keep connection alive
                await websocket.send_text(json.dumps({"echo": data}))
    except WebSocketDisconnect:
        connected_clients.discard(websocket)
    except Exception:
        connected_clients.discard(websocket)


@app.get("/")
async def root():
    return JSONResponse({"status": "ok", "audio_files": count_audio_files()})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
