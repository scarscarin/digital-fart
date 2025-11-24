import asyncio
from datetime import datetime
import json
import os
import shutil
import uuid
from typing import AsyncGenerator, List

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine, desc
from sqlalchemy.orm import Session, declarative_base, sessionmaker
import uvicorn

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


async def broadcast_message(message: dict) -> None:
    for queue in list(subscriber_queues):
        await queue.put(message)


async def simulate_training(run_id: int) -> None:
    db: Session = SessionLocal()
    try:
        steps = 60  # simulate ~10 minutes with 60 short steps
        for i in range(1, steps + 1):
            await asyncio.sleep(1)
            log_message = {"type": "log", "message": f"Step {i}/{steps}: crunching numbers"}
            await broadcast_message(log_message)

        # Build dummy output file
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_filename = f"normal_fart_{timestamp}.wav"
        output_path = os.path.join(GENERATED_DIR, output_filename)

        # Copy a random audio file if exists; otherwise, write placeholder bytes
        existing_files = [f for f in os.listdir(AUDIO_DIR) if os.path.isfile(os.path.join(AUDIO_DIR, f))]
        if existing_files:
            shutil.copy(os.path.join(AUDIO_DIR, existing_files[0]), output_path)
        else:
            with open(output_path, "wb") as f:
                f.write(b"RIFF\x00\x00\x00\x00WAVEfmt ")

        # Update training run status
        run: TrainingRun = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
        if run:
            run.status = "completed"
            run.finished_at = datetime.utcnow()
            run.output_filename = output_filename
            db.commit()

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
async def start_training(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    existing = db.query(TrainingRun).filter(TrainingRun.status == "running").first()
    if existing:
        return JSONResponse(status_code=409, content={"error": "Training already running"})

    run = TrainingRun(status="running", started_at=datetime.utcnow())
    db.add(run)
    db.commit()
    db.refresh(run)

    background_tasks.add_task(simulate_training, run.id)

    return {"started": True, "training_id": run.id}


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
