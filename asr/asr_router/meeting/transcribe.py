from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from asr_router.meeting.vad_diarize import DiarizedSegment
from asr_router.models.sense_voice import SenseVoiceTranscriber


@dataclass(frozen=True)
class SegmentTranscript:
    speaker: str
    start: float
    end: float
    text: str
    lang: str
    emotion: str
    event: str


_MIN_SEG_SEC = 0.2  # discard segments shorter than 200 ms


def transcribe_segments(
    wav_path: Path,
    segments: list[DiarizedSegment],
    chunk_max_sec: float = 30.0,
) -> list[SegmentTranscript]:
    """Run SenseVoice on each diarized segment.

    Segments shorter than _MIN_SEG_SEC are skipped silently.
    Segments longer than chunk_max_sec are split into chunk_max_sec pieces;
    timestamps remain absolute relative to the source audio.
    """
    samples, sr = sf.read(str(wav_path), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    sv = SenseVoiceTranscriber.get()
    out: list[SegmentTranscript] = []

    for seg in segments:
        s_idx = max(0, int(seg.start * sr))
        e_idx = min(len(samples), int(seg.end * sr))
        if e_idx - s_idx < int(_MIN_SEG_SEC * sr):
            continue
        chunk = samples[s_idx:e_idx]
        chunk_dur = (e_idx - s_idx) / sr

        if chunk_dur > chunk_max_sec:
            # Split into chunk_max_sec slices, preserving absolute timestamps
            step = int(chunk_max_sec * sr)
            for off in range(0, len(chunk), step):
                sub = chunk[off:off + step]
                if len(sub) < int(_MIN_SEG_SEC * sr):
                    continue
                r = sv.transcribe(samples=sub, sr=sr)
                out.append(
                    SegmentTranscript(
                        speaker=seg.speaker,
                        start=seg.start + off / sr,
                        end=seg.start + (off + len(sub)) / sr,
                        text=r.text,
                        lang=r.lang,
                        emotion=r.emotion,
                        event=r.event,
                    )
                )
        else:
            r = sv.transcribe(samples=chunk, sr=sr)
            out.append(
                SegmentTranscript(
                    speaker=seg.speaker,
                    start=seg.start,
                    end=seg.end,
                    text=r.text,
                    lang=r.lang,
                    emotion=r.emotion,
                    event=r.event,
                )
            )

    return out
