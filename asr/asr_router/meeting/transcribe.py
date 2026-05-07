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


def merge_consecutive_same_speaker(
    segments: list[SegmentTranscript],
    *,
    max_gap_sec: float = 2.0,
    max_merged_sec: float = 60.0,
) -> list[SegmentTranscript]:
    """Collapse adjacent SegmentTranscript chunks that share a speaker label
    and are separated by at most `max_gap_sec` of silence, up to a total
    `max_merged_sec` window.

    The intent is to reduce the number of items the LLM reviewer has to
    process. Diarization often emits many sub-second slivers for the same
    speaker (esp. with conservative thresholds); the reviewer doesn't need
    them as separate units. Texts are joined with a single space; lang /
    emotion / event are inherited from the FIRST chunk in the merged group
    (they are essentially per-utterance signals and the merged unit
    represents one continuous speaker turn).
    """
    if not segments:
        return []
    out: list[SegmentTranscript] = []
    cur: SegmentTranscript = segments[0]
    for nxt in segments[1:]:
        same_speaker = nxt.speaker == cur.speaker
        gap = nxt.start - cur.end
        merged_dur = nxt.end - cur.start
        if (
            same_speaker
            and gap <= max_gap_sec
            and merged_dur <= max_merged_sec
        ):
            cur = SegmentTranscript(
                speaker=cur.speaker,
                start=cur.start,
                end=nxt.end,
                text=f"{cur.text} {nxt.text}".strip(),
                lang=cur.lang,
                emotion=cur.emotion,
                event=cur.event,
            )
        else:
            out.append(cur)
            cur = nxt
    out.append(cur)
    return out


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
