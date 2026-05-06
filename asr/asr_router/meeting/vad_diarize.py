from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sherpa_onnx
import soundfile as sf

from asr_router.config import Settings


@dataclass(frozen=True)
class DiarizedSegment:
    start: float            # seconds
    end: float              # seconds
    speaker: str            # "Speaker_0", "Speaker_1", ...


_diarizer: Optional["sherpa_onnx.OfflineSpeakerDiarization"] = None


def _get_diarizer() -> "sherpa_onnx.OfflineSpeakerDiarization":
    global _diarizer
    if _diarizer is not None:
        return _diarizer
    s = Settings.load()
    seg_cfg = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=str(s.diarize_dir / "model.int8.onnx"),
        ),
    )
    emb_cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(s.speaker_embed_path),
    )
    cluster_cfg = sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5)
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=seg_cfg,
        embedding=emb_cfg,
        clustering=cluster_cfg,
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    _diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    return _diarizer


def _resample_to_16k(samples: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Resample mono float32 audio to 16 kHz via numpy linear interpolation.

    Sufficient for ASR; if higher quality is needed later, switch to scipy.signal.resample_poly.
    """
    if sr == 16000:
        return samples, sr
    target_sr = 16000
    new_len = int(round(len(samples) * target_sr / sr))
    x_old = np.linspace(0.0, 1.0, len(samples), dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, new_len, dtype=np.float64)
    out = np.interp(x_new, x_old, samples).astype(np.float32)
    return out, target_sr


def vad_diarize(wav_path: Path, max_speakers: int = 0) -> list[DiarizedSegment]:
    """Run VAD + speaker diarization on the given audio file.

    `max_speakers` is currently informational only — sherpa-onnx's threshold-based
    clustering auto-detects speaker count. The argument is kept for forward
    compatibility with hard-cap clustering modes.
    """
    samples, sr = sf.read(str(wav_path), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples, sr = _resample_to_16k(samples, sr)
    diarizer = _get_diarizer()
    if diarizer.sample_rate != sr:
        raise RuntimeError(
            f"Resampler produced {sr} Hz but diarizer wants {diarizer.sample_rate} Hz"
        )
    result = diarizer.process(samples.tolist())
    return [
        DiarizedSegment(start=s.start, end=s.end, speaker=f"Speaker_{s.speaker}")
        for s in result.sort_by_start_time()
    ]
