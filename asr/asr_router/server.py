from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
import io
import tempfile
from typing import Annotated

import soundfile as sf
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from asr_router.config import Settings, load_pipelines, load_routing
from asr_router.im.router import IMRouter
from asr_router.jobs import JobStore
from asr_router.meeting.pipeline import Worker
from asr_router.models.omlx_client import OMLXClient
from asr_router.models.sense_voice import SenseVoiceTranscriber


_settings: Settings | None = None
_router: IMRouter | None = None
_omlx: OMLXClient | None = None
_sv: SenseVoiceTranscriber | None = None
_store: JobStore | None = None
_worker: Worker | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def get_router() -> IMRouter:
    global _router
    if _router is None:
        _router = IMRouter(load_routing(get_settings().routing_yaml))
    return _router


def get_omlx() -> OMLXClient:
    global _omlx
    if _omlx is None:
        _omlx = OMLXClient.from_settings()
    return _omlx


def get_sv() -> SenseVoiceTranscriber:
    global _sv
    if _sv is None:
        _sv = SenseVoiceTranscriber.get()
    return _sv


def get_store() -> JobStore:
    global _store
    if _store is None:
        _store = JobStore(get_settings().storage_dir / "jobs.db")
    return _store


def require_auth(
    authorization: Annotated[str | None, Header()] = None,
    settings: Settings = Depends(get_settings),
) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing API key")
    if authorization.removeprefix("Bearer ").strip() != settings.api_key:
        raise HTTPException(401, "Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _worker
    settings = get_settings()
    store = get_store()
    omlx = get_omlx()
    pipelines_cfg = load_pipelines(settings.pipelines_yaml)
    _worker = Worker(store, settings, omlx, pipelines_cfg)
    _worker.start()
    try:
        yield
    finally:
        if _worker is not None:
            _worker.stop()


app = FastAPI(title="asr-router", version="0.1.0", lifespan=lifespan)


@app.get("/v1/models")
async def list_models(_: None = Depends(require_auth)):
    return {
        "object": "list",
        "data": [
            {"id": "auto", "object": "model", "owned_by": "asr-router"},
            {"id": "sense_voice", "object": "model", "owned_by": "asr-router"},
            {"id": "Qwen3-ASR-1.7B-8bit", "object": "model", "owned_by": "omlx"},
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    _: None = Depends(require_auth),
    file: UploadFile = File(...),
    model: Annotated[str, Form()] = "auto",
    quality: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "json",
    sv: SenseVoiceTranscriber = Depends(get_sv),
    omlx: OMLXClient = Depends(get_omlx),
    router: IMRouter = Depends(get_router),
):
    raw = await file.read()
    samples, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    duration = len(samples) / sr

    sv_result = sv.transcribe(samples=samples, sr=sr)
    decision = router.decide(
        duration_sec=duration,
        event=sv_result.event,
        lang=sv_result.lang,
        request_params={"quality": quality} if quality else {},
    )

    if decision.upstream == "sense_voice":
        return {
            "text": sv_result.text,
            "language": sv_result.lang,
            "duration": sv_result.duration_sec,
            "x_route": {
                "upstream": "sense_voice",
                "reason": decision.reason,
                "decode_ms": sv_result.decode_ms,
            },
            "x_tags": {
                "emotion": sv_result.emotion,
                "event": sv_result.event,
            },
        }

    # oMLX path: persist samples to a temp wav so we can multipart-upload it.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, samples, sr, subtype="PCM_16")
        omlx_resp = omlx.transcribe(tmp.name, model="Qwen3-ASR-1.7B-8bit")
    return {
        **omlx_resp,
        "x_route": {"upstream": "omlx", "reason": decision.reason},
        "x_tags": {
            "emotion": sv_result.emotion,
            "event": sv_result.event,
            "lid": sv_result.lang,
        },
    }


@app.post("/v1/audio/jobs")
async def submit_job(
    _: None = Depends(require_auth),
    file: UploadFile = File(...),
    glossary: Annotated[str, Form()] = "",
    settings: Settings = Depends(get_settings),
    store: JobStore = Depends(get_store),
):
    job_id = store.create(audio_path="<pending>", glossary_yaml=glossary)
    job_dir = settings.storage_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    audio_path = job_dir / "input.wav"

    raw = await file.read()
    samples, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    sf.write(str(audio_path), samples, sr, subtype="PCM_16")

    store.update(job_id, audio_path=str(audio_path))
    return {"id": job_id, "status": "queued"}


@app.get("/v1/audio/jobs/{job_id}")
async def get_job(
    job_id: str,
    _: None = Depends(require_auth),
    store: JobStore = Depends(get_store),
):
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    artifacts: list[str] = []
    if job.artifact_dir and Path(job.artifact_dir).exists():
        artifacts = sorted(p.name for p in Path(job.artifact_dir).iterdir() if p.is_file())
    return {
        "id": job.id,
        "status": job.status.value,
        "audio_path": job.audio_path,
        "artifact_dir": job.artifact_dir,
        "artifacts": artifacts,
        "error": job.error,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


@app.get("/v1/audio/jobs/{job_id}/artifact/{name}")
async def get_artifact(
    job_id: str,
    name: str,
    _: None = Depends(require_auth),
    store: JobStore = Depends(get_store),
):
    job = store.get(job_id)
    if job is None or job.artifact_dir is None:
        raise HTTPException(404, "job or artifact dir not found")
    if "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(400, "invalid artifact name")
    artifact_dir = Path(job.artifact_dir).resolve()
    p = (artifact_dir / name).resolve()
    if artifact_dir != p.parent:
        raise HTTPException(400, "invalid artifact name")
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "artifact not found")
    return FileResponse(p)
