import asyncio
from datetime import datetime, timedelta
import json
import math
import os
import struct
import uuid
from typing import AsyncGenerator, List, Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, desc
from sqlalchemy.orm import Session, declarative_base, sessionmaker
import uvicorn
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
GENERATED_DIR = os.path.join(BASE_DIR, "generated")
DB_PATH = os.path.join(BASE_DIR, "db.sqlite3")

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Fart(Base):
    __tablename__ = "farts"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    duration_seconds = Column(Float, nullable=True)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    status = Column(String, nullable=False)
    output_filename = Column(String, nullable=True)


Base.metadata.create_all(bind=engine)

app = FastAPI(title="digitalfart.net API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories for audio playback
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")
app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# In-memory queues for SSE subscribers
subscriber_queues: List[asyncio.Queue] = []


class TrainRequest(BaseModel):
    duration_minutes: Optional[int] = Field(10, ge=1, le=120, description="Length of training in minutes")


async def broadcast_message(message: dict) -> None:
    for queue in list(subscriber_queues):
        await queue.put(message)


def estimate_energy_co2(elapsed_seconds: int, cpu_power_watts: float = 35.0, emission_factor_kg_per_kwh: float = 0.4):
    """Return tuple of (energy_kwh, co2_grams) using a simple linear estimate."""
    energy_kwh = (cpu_power_watts / 1000.0) * (elapsed_seconds / 3600.0)
    co2_grams = energy_kwh * emission_factor_kg_per_kwh * 1000
    return energy_kwh, co2_grams


async def simulate_training(run_id: int, duration_minutes: int) -> None:
    db: Session = SessionLocal()
    try:
        # Real neural network: tiny 1D autoencoder trained on synthetic sine waves
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=9, padding=4),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        total_seconds = duration_minutes * 60
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(seconds=total_seconds)
        step = 0
        last_log = start_time
        await broadcast_message(
            {
                "type": "log",
                "message": (
                    f"Bootstrapping real neural net on {device.type.upper()} for {duration_minutes} minute(s). "
                    f"Target duration: {total_seconds} seconds."
                ),
            }
        )
        last_energy = last_co2 = 0.0

        def make_batch(batch_size: int = 8, sample_len: int = 1024):
            t = torch.linspace(0, 1, sample_len, device=device).unsqueeze(0).unsqueeze(0)
            freq = torch.rand(batch_size, 1, 1, device=device) * 6 + 1
            signal = torch.sin(2 * math.pi * freq * t)
            noise = 0.05 * torch.randn(batch_size, 1, sample_len, device=device)
            data = signal + noise
            return data, data  # autoencoder target is the input itself

        while datetime.utcnow() < end_time:
            inputs, targets = make_batch()
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            step += 1

            now = datetime.utcnow()
            elapsed = (now - start_time).total_seconds()
            remaining = max(0.0, total_seconds - elapsed)
            # Estimate resource usage: assume 35W baseline CPU, 100W if GPU
            cpu_power = 100.0 if device.type == "cuda" else 35.0
            energy_kwh, co2_g = estimate_energy_co2(int(elapsed), cpu_power)
            last_energy, last_co2 = energy_kwh, co2_g

            if (now - last_log).total_seconds() >= 1:
                progress = min(100.0, (elapsed / total_seconds) * 100)
                await broadcast_message(
                    {
                        "type": "log",
                        "message": (
                            f"Step {step} | loss={loss.item():.5f} | progress={progress:.1f}% | "
                            f"remaining={remaining:.1f}s | power~{cpu_power:.1f}W | "
                            f"energy~{energy_kwh:.4f} kWh | co2~{co2_g:.2f} g"
                        ),
                    }
                )
                last_log = now

            # Yield control very briefly to keep the event loop responsive
            await asyncio.sleep(0)

        # If we exit slightly early due to loop timing, top up with sleep to hit the target duration
        final_elapsed = (datetime.utcnow() - start_time).total_seconds()
        if final_elapsed < total_seconds:
            await asyncio.sleep(total_seconds - final_elapsed)

        # Build output file from a trained sample
        model.eval()
        with torch.no_grad():
            sample_in, _ = make_batch(batch_size=1)
            generated = model(sample_in).cpu().squeeze().numpy()

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_filename = f"normal_fart_{timestamp}.wav"
        output_path = os.path.join(GENERATED_DIR, output_filename)

        # Convert float waveform (-1..1) to 16-bit PCM WAV
        int_data = (generated * 32767.0).clip(-32768, 32767).astype("int16")
        with open(output_path, "wb") as f:
            # Minimal RIFF/WAV header
            num_channels = 1
            sample_rate = 16000
            byte_rate = sample_rate * num_channels * 2
            block_align = num_channels * 2
            bits_per_sample = 16
            data_bytes = int_data.tobytes()
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + len(data_bytes)))
            f.write(b"WAVEfmt ")
            f.write(struct.pack("<IHHIIHH", 16, 1, num_channels, sample_rate, byte_rate, block_align, bits_per_sample))
            f.write(b"data")
            f.write(struct.pack("<I", len(data_bytes)))
            f.write(data_bytes)

        # Update training run status
        run: TrainingRun = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if run:
            run.status = "completed"
            run.finished_at = datetime.utcnow()
            run.output_filename = output_filename
            db.commit()
        await broadcast_message(
            {
                "type": "log",
                "message": (
                    f"Training finished after {duration_minutes} minute(s). Total energy ~{last_energy:.4f} kWh; "
                    f"estimated CO2 ~{last_co2:.2f} g. Writing output {output_filename}."
                ),
            }
        )
        await broadcast_message({"type": "done", "url": f"/generated/{output_filename}"})
    except Exception as exc:  # pragma: no cover - safety
        run: TrainingRun = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if run:
            run.status = "failed"
            run.finished_at = datetime.utcnow()
            db.commit()
        await broadcast_message({"type": "log", "message": f"Training failed: {exc}"})
    finally:
        db.close()


@app.post("/api/upload")
async def upload_fart(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    ext = os.path.splitext(file.filename)[1] or ".webm"
    unique_name = f"{uuid.uuid4()}{ext}"
    save_path = os.path.join(AUDIO_DIR, unique_name)

    with open(save_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    fart = Fart(filename=unique_name, created_at=datetime.utcnow())
    db.add(fart)
    db.commit()
    db.refresh(fart)

    return {
        "id": fart.id,
        "filename": fart.filename,
        "url": f"/audio/{fart.filename}",
        "created_at": fart.created_at.isoformat() + "Z",
    }


@app.get("/api/archive")
async def list_archive(db: Session = Depends(get_db)):
    farts = db.query(Fart).order_by(desc(Fart.created_at)).all()
    return [
        {
            "id": fart.id,
            "url": f"/audio/{fart.filename}",
            "created_at": fart.created_at.isoformat() + "Z",
            "duration_seconds": fart.duration_seconds,
        }
        for fart in farts
    ]


@app.post("/api/admin/train")
async def start_training(
    payload: TrainRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    existing = db.query(TrainingRun).filter(TrainingRun.status == "running").first()
    if existing:
        return JSONResponse(status_code=409, content={"error": "Training already running"})

    duration_minutes = payload.duration_minutes or 10
    run = TrainingRun(status="running", started_at=datetime.utcnow())
    db.add(run)
    db.commit()
    db.refresh(run)

    background_tasks.add_task(simulate_training, run.id, duration_minutes)

    return {"started": True, "training_id": run.id, "duration_minutes": duration_minutes}


@app.get("/api/training/status")
async def training_status(db: Session = Depends(get_db)):
    running = db.query(TrainingRun).filter(TrainingRun.status == "running").order_by(desc(TrainingRun.started_at)).first()
    last_run = db.query(TrainingRun).order_by(desc(TrainingRun.started_at)).first()

    if running:
        target = running
        is_running = True
    else:
        target = last_run
        is_running = False

    if not target:
        return {
            "running": False,
            "training_id": None,
            "started_at": None,
            "finished_at": None,
            "status": None,
            "output_url": None,
        }

    output_url = f"/generated/{target.output_filename}" if target.output_filename else None

    return {
        "running": is_running,
        "training_id": target.id,
        "started_at": target.started_at.isoformat() + "Z" if target.started_at else None,
        "finished_at": target.finished_at.isoformat() + "Z" if target.finished_at else None,
        "status": target.status,
        "output_url": output_url,
    }


@app.get("/api/normal_fart/current")
async def current_normal_fart(db: Session = Depends(get_db)):
    run = (
        db.query(TrainingRun)
        .filter(TrainingRun.status == "completed", TrainingRun.output_filename.isnot(None))
        .order_by(desc(TrainingRun.finished_at))
        .first()
    )
    if not run:
        raise HTTPException(status_code=404, detail={"output_url": None})

    return {
        "training_id": run.id,
        "output_url": f"/generated/{run.output_filename}",
    }


@app.get("/api/training/stream")
async def training_stream() -> StreamingResponse:
    queue: asyncio.Queue = asyncio.Queue()
    subscriber_queues.append(queue)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                message = await queue.get()
                yield f"data: {json.dumps(message)}\n\n"
        except asyncio.CancelledError:
            raise
        finally:
            if queue in subscriber_queues:
                subscriber_queues.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
