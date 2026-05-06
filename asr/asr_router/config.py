from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os
import yaml


def _default_sense_voice_dir() -> Path:
    return Path.home() / "models/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"


def _default_silero_vad() -> Path:
    return Path.home() / "models/sherpa-onnx/silero-vad/silero_vad.onnx"


def _default_diarize_dir() -> Path:
    return Path.home() / "models/sherpa-onnx/sherpa-onnx-pyannote-segmentation-3-0"


def _default_speaker_embed() -> Path:
    return Path.home() / "models/sherpa-onnx/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"


def _default_storage_dir() -> Path:
    return Path.home() / ".asr-router"


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 18081
    api_key: str = "sk-mlx"
    omlx_base_url: str = "http://localhost:18080/v1"
    omlx_api_key: str = "sk-mlx"
    sense_voice_dir: Path = field(default_factory=_default_sense_voice_dir)
    silero_vad_path: Path = field(default_factory=_default_silero_vad)
    diarize_dir: Path = field(default_factory=_default_diarize_dir)
    speaker_embed_path: Path = field(default_factory=_default_speaker_embed)
    storage_dir: Path = field(default_factory=_default_storage_dir)
    # NOTE: review/summary model selection lives in pipelines.yaml, NOT here.
    # See `meeting.review.model` and `meeting.summary.model` in pipelines.yaml.
    glossary_default: Path = field(default_factory=lambda: Path(__file__).parent.parent / "glossary/default.yaml")
    routing_yaml: Path = field(default_factory=lambda: Path(__file__).parent.parent / "routing.yaml")
    pipelines_yaml: Path = field(default_factory=lambda: Path(__file__).parent.parent / "pipelines.yaml")

    @classmethod
    def load(cls) -> "Settings":
        return cls(
            port=int(os.environ.get("ASR_PORT", 18081)),
            api_key=os.environ.get("ASR_API_KEY", "sk-mlx"),
            omlx_base_url=os.environ.get("OMLX_BASE_URL", "http://localhost:18080/v1"),
            omlx_api_key=os.environ.get("OMLX_API_KEY", "sk-mlx"),
        )


def load_routing(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def load_pipelines(path: Path) -> dict:
    return yaml.safe_load(path.read_text())
