from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
import time

import numpy as np
import sherpa_onnx
import soundfile as sf

from asr_router.config import Settings


_LANG_MAP = {
    "<|zh|>": "zh",
    "<|en|>": "en",
    "<|yue|>": "yue",
    "<|ja|>": "ja",
    "<|ko|>": "ko",
    "<|nospeech|>": "nospeech",
}

_EVENT_MAP = {
    "<|Speech|>": "Speech",
    "<|BGM|>": "BGM",
    "<|Applause|>": "Applause",
    "<|Laughter|>": "Laughter",
    "<|Cry|>": "Cry",
    "<|Sneeze|>": "Sneeze",
    "<|Breath|>": "Breath",
    "<|Cough|>": "Cough",
}


@dataclass(frozen=True)
class TranscribeResult:
    text: str
    lang: str           # zh / en / yue / ja / ko / nospeech / unknown
    emotion: str        # NEUTRAL / HAPPY / SAD / ANGRY / FEARFUL / DISGUSTED / SURPRISED / ""
    event: str          # Speech / BGM / Applause / ... / unknown
    duration_sec: float
    decode_ms: float
    timestamps: list[float]
    tokens: list[str]


class SenseVoiceTranscriber:
    _instance: ClassVar["SenseVoiceTranscriber | None"] = None

    def __init__(self, model_dir: Path):
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_dir / "model.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            use_itn=True,
            language="auto",
            num_threads=4,
        )

    @classmethod
    def get(cls) -> "SenseVoiceTranscriber":
        if cls._instance is None:
            cls._instance = cls(Settings.load().sense_voice_dir)
        return cls._instance

    def transcribe(
        self,
        wav_path: Path | str | None = None,
        *,
        samples: np.ndarray | None = None,
        sr: int | None = None,
    ) -> TranscribeResult:
        if samples is None:
            if wav_path is None:
                raise ValueError("Either wav_path or (samples, sr) must be provided")
            samples, sr = sf.read(str(wav_path), dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
        if sr is None:
            raise ValueError("sr is required when samples is provided")

        duration_sec = len(samples) / sr
        s = self._recognizer.create_stream()
        t0 = time.perf_counter()
        s.accept_waveform(sr, samples)
        self._recognizer.decode_stream(s)
        decode_ms = (time.perf_counter() - t0) * 1000
        r = s.result
        emotion_raw = r.emotion or ""
        emotion = emotion_raw.strip("<|>") if emotion_raw else ""
        return TranscribeResult(
            text=r.text,
            lang=_LANG_MAP.get(r.lang, "unknown"),
            emotion=emotion,
            event=_EVENT_MAP.get(r.event, "unknown"),
            duration_sec=duration_sec,
            decode_ms=decode_ms,
            timestamps=list(r.timestamps),
            tokens=list(r.tokens),
        )
