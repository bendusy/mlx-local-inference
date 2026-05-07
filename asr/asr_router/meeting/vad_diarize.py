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


# Cached diarizer keyed by (num_clusters, threshold, min_on, min_off) so
# tuning these in pipelines.yaml does not require a daemon restart for
# fresh jobs to pick up the new params.
_diarizer_cache: dict[tuple, "sherpa_onnx.OfflineSpeakerDiarization"] = {}


def _get_diarizer(
    *,
    num_clusters: int = -1,
    cluster_threshold: float = 0.5,
    min_duration_on: float = 0.3,
    min_duration_off: float = 0.5,
) -> "sherpa_onnx.OfflineSpeakerDiarization":
    key = (num_clusters, cluster_threshold, min_duration_on, min_duration_off)
    if key in _diarizer_cache:
        return _diarizer_cache[key]
    s = Settings.load()
    seg_cfg = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
        pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
            model=str(s.diarize_dir / "model.int8.onnx"),
        ),
    )
    emb_cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(s.speaker_embed_path),
    )
    cluster_cfg = sherpa_onnx.FastClusteringConfig(
        num_clusters=num_clusters, threshold=cluster_threshold
    )
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=seg_cfg,
        embedding=emb_cfg,
        clustering=cluster_cfg,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
    )
    diar = sherpa_onnx.OfflineSpeakerDiarization(config)
    _diarizer_cache[key] = diar
    return diar


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


def vad_diarize(
    wav_path: Path,
    *,
    num_clusters: int = -1,
    cluster_threshold: float = 0.5,
    min_duration_on: float = 0.3,
    min_duration_off: float = 0.5,
) -> list[DiarizedSegment]:
    """Run VAD + speaker diarization on the given audio file.

    Tunables (all reflected in pipelines.yaml meeting.diarize.*):
    - num_clusters: -1 lets the threshold path auto-detect speaker count.
      Set to a positive integer to force exactly N clusters (best when you
      know the meeting size).
    - cluster_threshold: similarity bound for FastClustering. Lower values
      merge more aggressively (FEWER clusters); higher values split more
      eagerly (MORE clusters). Empirical sweet spot for 30+ min mixed-language
      meetings: 0.3.
    - min_duration_on/off: minimum speech / silence durations passed to the
      pyannote segmentation step.
    """
    samples, sr = sf.read(str(wav_path), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples, sr = _resample_to_16k(samples, sr)
    diarizer = _get_diarizer(
        num_clusters=num_clusters,
        cluster_threshold=cluster_threshold,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
    )
    if diarizer.sample_rate != sr:
        raise RuntimeError(
            f"Resampler produced {sr} Hz but diarizer wants {diarizer.sample_rate} Hz"
        )
    result = diarizer.process(samples.tolist())
    return [
        DiarizedSegment(start=s.start, end=s.end, speaker=f"Speaker_{s.speaker}")
        for s in result.sort_by_start_time()
    ]
