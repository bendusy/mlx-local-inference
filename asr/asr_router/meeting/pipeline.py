from __future__ import annotations
from pathlib import Path
import threading
import time
import traceback

import yaml

from asr_router.config import Settings
from asr_router.glossary import Glossary
from asr_router.jobs import Job, JobStatus, JobStore
from asr_router.meeting.vad_diarize import vad_diarize, release_diarizer_cache
from asr_router.meeting.transcribe import transcribe_segments, merge_consecutive_same_speaker
from asr_router.meeting.review import review_segments
from asr_router.meeting.render import render_artifacts
from asr_router.models.omlx_client import OMLXClient
from asr_router.models.sense_voice import SenseVoiceTranscriber


def run_job(
    job: Job,
    *,
    settings: Settings,
    store: JobStore,
    omlx: OMLXClient,
    pipelines_cfg: dict,
) -> None:
    """Run the full meeting pipeline for a single job. State transitions are
    persisted via `store.update`. Failures set status=FAILED with a traceback.
    """
    try:
        cfg = pipelines_cfg["meeting"]
        audio = Path(job.audio_path)
        out_dir = settings.storage_dir / "jobs" / job.id
        out_dir.mkdir(parents=True, exist_ok=True)
        store.update(job.id, artifact_dir=str(out_dir))

        # Pass 1: VAD + diarize
        store.update(job.id, status=JobStatus.VAD_DIARIZE)
        diar_cfg = cfg["diarize"]
        diarized = vad_diarize(
            audio,
            num_clusters=int(diar_cfg.get("num_clusters", -1)),
            cluster_threshold=float(diar_cfg.get("cluster_threshold", 0.5)),
            min_duration_on=float(diar_cfg.get("min_duration_on", 0.3)),
            min_duration_off=float(diar_cfg.get("min_duration_off", 0.5)),
        )
        # Diarize done — release the pyannote+3D-Speaker ONNX sessions.
        # SenseVoice is still needed for Pass 2; keep that loaded.
        release_diarizer_cache()
        print("[pipeline] released diarizer models after Pass 1", flush=True)

        # Pass 2: Transcribe per segment
        store.update(job.id, status=JobStatus.TRANSCRIBING)
        raw = transcribe_segments(
            audio, diarized, chunk_max_sec=cfg["transcribe"]["chunk_max_sec"]
        )
        # Optional same-speaker merge to reduce reviewer batch count
        merge_cfg = cfg["transcribe"].get("merge", {})
        if merge_cfg.get("enabled", True):
            before = len(raw)
            raw = merge_consecutive_same_speaker(
                raw,
                max_gap_sec=float(merge_cfg.get("max_gap_sec", 2.0)),
                max_merged_sec=float(merge_cfg.get("max_merged_sec", 60.0)),
            )
            print(
                f"[pipeline] same-speaker merge: {before} → {len(raw)} segments",
                flush=True,
            )
        # Transcribe done — release SenseVoice before kicking off gemma-4
        # review in oMLX. This frees ~228 MB so oMLX can fully utilise
        # GPU/memory for the LLM. Cold reload cost (~500 ms) hits the
        # next job, not the current pipeline.
        SenseVoiceTranscriber.release()
        print("[pipeline] released SenseVoice after Pass 2", flush=True)

        # Pass 3: gemma-4 review
        store.update(job.id, status=JobStatus.REVIEWING)
        default_yaml = (
            yaml.safe_load(settings.glossary_default.read_text(encoding="utf-8"))
            if settings.glossary_default.exists()
            else {}
        )
        perjob_yaml = yaml.safe_load(job.glossary_yaml) if job.glossary_yaml else {}
        glossary = Glossary.merged(default_yaml, perjob_yaml)
        reviewed, role_map = review_segments(
            raw,
            glossary=glossary,
            omlx=omlx,
            model=cfg["review"]["model"],
            window=cfg["review"]["context_window_segments"],
            batch=cfg["review"]["max_segments_per_call"],
            timeout_sec=float(cfg["review"].get("timeout_sec", 300)),
            parallel_batches=int(cfg["review"].get("parallel_batches", 1)),
        )

        # Pass 4: Render
        store.update(job.id, status=JobStatus.RENDERING)
        render_artifacts(
            out_dir=out_dir,
            stem=audio.stem,
            raw=raw,
            reviewed=reviewed,
            role_map=role_map,
            omlx=omlx,
            summary_model=cfg["summary"]["model"],
            summary_timeout_sec=float(cfg["summary"].get("timeout_sec", 180)),
        )

        store.update(job.id, status=JobStatus.DONE)
    except Exception as e:  # pragma: no cover — covered by Task 14 e2e
        tb = traceback.format_exc()
        store.update(
            job.id,
            status=JobStatus.FAILED,
            error=f"{type(e).__name__}: {e}\n{tb}",
        )


class Worker:
    """Background thread polling the JobStore for QUEUED jobs and running them."""

    def __init__(
        self,
        store: JobStore,
        settings: Settings,
        omlx: OMLXClient,
        pipelines_cfg: dict,
        poll_interval_sec: float = 2.0,
    ):
        self._store = store
        self._settings = settings
        self._omlx = omlx
        self._pipelines = pipelines_cfg
        self._poll = poll_interval_sec
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="asr-worker")

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self._store.pop_next_queued()
            if job is None:
                # Wait for next poll, but wake up early if asked to stop.
                self._stop.wait(self._poll)
                continue
            run_job(
                job,
                settings=self._settings,
                store=self._store,
                omlx=self._omlx,
                pipelines_cfg=self._pipelines,
            )
