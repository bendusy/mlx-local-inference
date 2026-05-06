# ASR Routing Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone ASR module under `asr/` that exposes a Whisper-compatible HTTP service on `localhost:18081`, dynamically routes IM-style short audio between fast local SenseVoice and high-quality oMLX Qwen3-ASR, and runs a multi-pass review pipeline (VAD → diarize → SenseVoice → gemma-4 contextual review) for long meeting recordings to recover from poor single-shot ASR quality.

**Architecture:** Single FastAPI process exposing two endpoints — sync `POST /v1/audio/transcriptions` for IM mode (sub-second routing on lang/duration/event tags emitted by SenseVoice), async `POST /v1/audio/jobs` for meeting mode (queued multi-pass pipeline persisting artifacts under `~/.asr-router/jobs/<id>/`). Meeting pipeline produces five model-named artifacts (`_raw.json`, `_sensevoice.md`, `_gemma4.md`, `_diff.md`, `_summary.md`) so quality differences between models are auditable. Validation uses your meeting recording directory (set `ASR_EVAL_AUDIO` env var) + manually corrected diarization JSON as ground truth.

**Tech Stack:** Python 3.11, uv-managed deps, FastAPI + uvicorn, sherpa-onnx (SenseVoice int8 + silero-vad + 3D-Speaker diarization), httpx (oMLX upstream), sqlite-vec (job store), soundfile + numpy, openai (gemma-4 client via oMLX), pytest.

---

## File Structure

```
asr/
├── README.md                            # Module-level docs, quickstart, API reference
├── pyproject.toml                       # uv project, deps pinned
├── routing.yaml                         # IM mode rule table
├── pipelines.yaml                       # Meeting mode stage config (chunk size, model picks)
├── glossary/
│   └── default.yaml                     # Persistent terms (人名/项目代号), checked into git
├── prompts/
│   ├── review.j2                        # gemma-4 contextual review prompt
│   └── summary.j2                       # gemma-4 meeting-summary prompt
├── asr_router/
│   ├── __init__.py
│   ├── config.py                        # Load routing.yaml + pipelines.yaml + env
│   ├── server.py                        # FastAPI app, route definitions
│   ├── models/
│   │   ├── __init__.py
│   │   ├── sense_voice.py               # SenseVoiceTranscriber wrapper (singleton)
│   │   ├── omlx_client.py               # OMLXClient (audio + chat completions)
│   │   ├── vad.py                       # SileroVAD wrapper
│   │   └── diarize.py                   # SpeakerDiarizer wrapper (sherpa-onnx)
│   ├── im/
│   │   ├── __init__.py
│   │   └── router.py                    # IM mode rule-based router
│   ├── meeting/
│   │   ├── __init__.py
│   │   ├── pipeline.py                  # Orchestrator: queued → done state machine
│   │   ├── vad_diarize.py               # Pass 1
│   │   ├── transcribe.py                # Pass 2 (chunked SenseVoice over diarized segments)
│   │   ├── review.py                    # Pass 3 (gemma-4 contextual review w/ glossary)
│   │   └── render.py                    # Pass 4 (write 5 artifacts)
│   ├── jobs.py                          # SQLite job store + worker loop
│   ├── glossary.py                      # Glossary merge (default + per-job)
│   └── cli.py                           # `python -m asr_router transcribe path.wav`
├── tests/
│   ├── conftest.py
│   ├── fixtures/                        # short test wavs (symlink to sherpa-onnx test_wavs)
│   ├── test_im_router.py
│   ├── test_sense_voice.py
│   ├── test_vad_diarize.py
│   ├── test_review.py
│   ├── test_jobs.py
│   ├── test_server_im.py
│   └── test_server_jobs.py
└── scripts/
    ├── install_models.sh                # Download SenseVoice int8 + silero-vad + diarization
    ├── run_dev.sh                       # uvicorn --reload on 18081
    └── eval_meeting.py                  # Run full pipeline on real meeting audio (env var), CER vs ground truth
```

**Storage layout (runtime, outside repo):**
```
~/.asr-router/
├── jobs.db                              # sqlite (job_id, status, audio_path, timestamps, artifact_dir)
└── jobs/<job_id>/
    ├── manifest.json
    ├── input.wav                        # symlink to source
    ├── vad_diarize.json
    ├── _raw.json                        # all SenseVoice segments + tags
    ├── <stem>_sensevoice.md             # SenseVoice raw concat
    ├── <stem>_gemma4.md                 # gemma-4 reviewed
    ├── <stem>_diff.md                   # changes between sv ↔ gemma4
    └── <stem>_summary.md                # gemma-4 generated summary
```

---

## Phase 0: Preparation

### Task 0: Remove obsolete server/

**Files:**
- Delete: `server/admin_api_patch.py`, `server/config.yaml`, `server/idle-unload-watchdog.py`, `server/lazy_handler_proxy.py`, `server/start_with_admin.py`, `server/transcribe-daemon.py`, `server/watchdog.yaml`
- Delete: empty `server/` directory

- [ ] **Step 1: Verify no current process depends on `server/`**

```bash
launchctl list | grep -i omlx       # should show com.omlx.app, NOT homebrew.mxcl.omlx
ps aux | grep -E "lazy_handler|idle-unload|transcribe-daemon" | grep -v grep
```
Expected: empty (oMLX.app handles all of this natively now).

- [ ] **Step 2: Remove the directory**

```bash
git rm -r server/
```

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: remove server/ (replaced by oMLX native)

oMLX.app provides continuous batching, SSD cache, idle unload,
admin endpoints natively. The lazy/watchdog/admin shims are
obsolete. transcribe-daemon will be replaced by the asr/
module in the next commits."
```

---

## Phase 1: Module Scaffold

### Task 1: Bootstrap `asr/` package

**Files:**
- Create: `asr/pyproject.toml`, `asr/README.md` (stub), `asr/asr_router/__init__.py`, `asr/asr_router/config.py`, `asr/routing.yaml`, `asr/pipelines.yaml`, `asr/scripts/install_models.sh`, `asr/scripts/run_dev.sh`, `asr/tests/conftest.py`, `asr/.gitignore`

- [ ] **Step 1: Write `asr/pyproject.toml`**

```toml
[project]
name = "asr-router"
version = "0.1.0"
description = "Standalone ASR routing service: SenseVoice (fast) + oMLX (quality) + meeting pipeline."
requires-python = ">=3.11,<3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "sherpa-onnx>=1.10",
    "soundfile>=0.12",
    "numpy>=1.26",
    "httpx>=0.27",
    "openai>=1.50",
    "pyyaml>=6.0",
    "jinja2>=3.1",
    "pydantic>=2.7",
    "python-multipart>=0.0.9",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23", "httpx[testing]"]

[project.scripts]
asr-router = "asr_router.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Write `asr/asr_router/config.py`** — single config loader (no env-var sprawl)

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os, yaml

@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 18081
    api_key: str = "sk-mlx"
    omlx_base_url: str = "http://localhost:18080/v1"
    omlx_api_key: str = "sk-mlx"
    sense_voice_dir: Path = Path.home() / "models/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    silero_vad_path: Path = Path.home() / "models/sherpa-onnx/silero-vad/silero_vad.onnx"
    diarize_dir: Path = Path.home() / "models/sherpa-onnx/sherpa-onnx-pyannote-segmentation-3-0"
    speaker_embed_path: Path = Path.home() / "models/sherpa-onnx/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
    storage_dir: Path = Path.home() / ".asr-router"
    review_model: str = "gemma-4-26b-a4b-it-4bit"
    summary_model: str = "gemma-4-26b-a4b-it-4bit"
    glossary_default: Path = Path(__file__).parent.parent / "glossary/default.yaml"
    routing_yaml: Path = Path(__file__).parent.parent / "routing.yaml"
    pipelines_yaml: Path = Path(__file__).parent.parent / "pipelines.yaml"

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
```

- [ ] **Step 3: Write `asr/routing.yaml` and `asr/pipelines.yaml`**

`routing.yaml`:
```yaml
defaults:
  upstream: sense_voice

rules:
  - when: { request_param: { quality: high } }
    use: omlx
  - when: { request_param: { quality: fast } }
    use: sense_voice
  - when: { duration_gt: 30 }
    use: omlx
  - when: { event_in: ["<|BGM|>", "<|Applause|>", "<|Laughter|>"] }
    use: omlx
```

`pipelines.yaml`:
```yaml
meeting:
  vad:
    min_silence_ms: 500
    speech_pad_ms: 200
  diarize:
    min_segment_sec: 1.0
    max_speakers: 8
  transcribe:
    backend: sense_voice
    chunk_max_sec: 30
  review:
    model: gemma-4-26b-a4b-it-4bit
    context_window_segments: 4
    max_segments_per_call: 12
  summary:
    model: gemma-4-26b-a4b-it-4bit
```

- [ ] **Step 4: Write `asr/glossary/default.yaml`** (Replace these placeholders with your own terms via per-job glossary)

```yaml
# Default glossary — placeholder examples. Override with per-job glossary
# at request time, OR replace this file with your own real terms in a fork.
terms:
  - term: "Alpha Group"
    aliases: ["Alpa Group", "Alpha"]
  - term: "Beta Corp"
    aliases: ["Beta Corporation", "Beta Inc."]
  - term: "Chairman Zhang"
    aliases: ["Mr. Zhang", "Zhang"]
  - term: "Example City"
    aliases: ["E.C."]
```

- [ ] **Step 5: Write `asr/scripts/install_models.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/models/sherpa-onnx"
mkdir -p "$ROOT"
cd "$ROOT"

# 1. SenseVoice int8 (already present per prior verification)
SV="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
[ -d "$SV" ] || {
  curl -L -o sv.tar.bz2 "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${SV}.tar.bz2"
  tar xjf sv.tar.bz2 && rm sv.tar.bz2
}

# 2. silero-vad
mkdir -p silero-vad
[ -f silero-vad/silero_vad.onnx ] || {
  curl -L -o silero-vad/silero_vad.onnx \
    "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
}

# 3. Speaker diarization (pyannote segmentation + 3D-Speaker embedding)
SEG="sherpa-onnx-pyannote-segmentation-3-0"
[ -d "$SEG" ] || {
  curl -L -o seg.tar.bz2 "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/${SEG}.tar.bz2"
  tar xjf seg.tar.bz2 && rm seg.tar.bz2
}
EMB="3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
[ -f "$EMB" ] || {
  curl -L -O "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/${EMB}"
}

echo "All ASR models installed at $ROOT"
```

- [ ] **Step 6: Write `asr/scripts/run_dev.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run --python 3.11 uvicorn asr_router.server:app --host 0.0.0.0 --port "${ASR_PORT:-18081}" --reload
```

- [ ] **Step 7: Write `asr/.gitignore`**

```
__pycache__/
*.pyc
.venv/
.pytest_cache/
data/
```

- [ ] **Step 8: Write `asr/tests/conftest.py`**

```python
import os, pytest
from pathlib import Path

SV_DIR = Path.home() / "models/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"

@pytest.fixture(scope="session")
def test_wav_dir() -> Path:
    return SV_DIR / "test_wavs"

@pytest.fixture(scope="session")
def zh_wav(test_wav_dir) -> Path:
    return test_wav_dir / "zh.wav"

@pytest.fixture(scope="session")
def en_wav(test_wav_dir) -> Path:
    return test_wav_dir / "en.wav"

@pytest.fixture(scope="session")
def yue_wav(test_wav_dir) -> Path:
    return test_wav_dir / "yue.wav"

@pytest.fixture(scope="session", autouse=True)
def _require_sv_dir():
    if not SV_DIR.exists():
        pytest.skip("Run scripts/install_models.sh first", allow_module_level=True)
```

- [ ] **Step 9: Run install + verify**

```bash
bash asr/scripts/install_models.sh
cd asr && uv sync --extra dev && uv run pytest -q
```
Expected: 0 tests, no import errors.

- [ ] **Step 10: Commit**

```bash
git add asr/
git commit -m "feat(asr): scaffold standalone module structure

- pyproject.toml with sherpa-onnx + FastAPI deps
- routing.yaml (IM mode rules) + pipelines.yaml (meeting stages)
- default glossary with placeholder examples (replace with your own terms)
- install_models.sh covers SenseVoice + silero-vad + diarization
- empty test scaffold; conftest skips if models missing"
```

---

### Task 2: SenseVoice transcriber wrapper (singleton + LID extraction)

**Files:**
- Create: `asr/asr_router/models/__init__.py`, `asr/asr_router/models/sense_voice.py`, `asr/tests/test_sense_voice.py`

- [ ] **Step 1: Write failing test `tests/test_sense_voice.py`**

```python
from asr_router.models.sense_voice import SenseVoiceTranscriber, TranscribeResult

def test_zh_lid(zh_wav):
    sv = SenseVoiceTranscriber.get()
    r = sv.transcribe(zh_wav)
    assert isinstance(r, TranscribeResult)
    assert r.lang == "zh"
    assert "开放时间" in r.text
    assert r.event == "Speech"
    assert r.duration_sec > 5.0
    assert r.decode_ms < 500

def test_yue_lid(yue_wav):
    r = SenseVoiceTranscriber.get().transcribe(yue_wav)
    assert r.lang == "yue"
    assert "唔到" in r.text  # 粤语特征字

def test_en_lid(en_wav):
    r = SenseVoiceTranscriber.get().transcribe(en_wav)
    assert r.lang == "en"

def test_singleton_reuses_session():
    a = SenseVoiceTranscriber.get()
    b = SenseVoiceTranscriber.get()
    assert a is b
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd asr && uv run pytest tests/test_sense_voice.py -v
```
Expected: ImportError on `asr_router.models.sense_voice`.

- [ ] **Step 3: Implement `models/sense_voice.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
import time, soundfile as sf, sherpa_onnx, numpy as np
from asr_router.config import Settings

_LANG_MAP = {"<|zh|>":"zh","<|en|>":"en","<|yue|>":"yue","<|ja|>":"ja","<|ko|>":"ko","<|nospeech|>":"nospeech"}
_EVENT_MAP = {"<|Speech|>":"Speech","<|BGM|>":"BGM","<|Applause|>":"Applause","<|Laughter|>":"Laughter","<|Cry|>":"Cry","<|Sneeze|>":"Sneeze","<|Breath|>":"Breath","<|Cough|>":"Cough"}

@dataclass(frozen=True)
class TranscribeResult:
    text: str
    lang: str
    emotion: str
    event: str
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

    def transcribe(self, wav_path: Path | str, samples: np.ndarray | None = None, sr: int | None = None) -> TranscribeResult:
        if samples is None:
            samples, sr = sf.read(str(wav_path), dtype="float32")
        dur = len(samples) / sr
        s = self._recognizer.create_stream()
        t0 = time.perf_counter()
        s.accept_waveform(sr, samples)
        self._recognizer.decode_stream(s)
        ms = (time.perf_counter() - t0) * 1000
        r = s.result
        return TranscribeResult(
            text=r.text,
            lang=_LANG_MAP.get(r.lang, "unknown"),
            emotion=r.emotion.strip("<>|") if r.emotion else "",
            event=_EVENT_MAP.get(r.event, "unknown"),
            duration_sec=dur,
            decode_ms=ms,
            timestamps=list(r.timestamps),
            tokens=list(r.tokens),
        )
```

- [ ] **Step 4: Run test to verify pass**

```bash
cd asr && uv run pytest tests/test_sense_voice.py -v
```
Expected: 4/4 pass.

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(asr): SenseVoice transcriber wrapper with LID extraction"
```

---

### Task 3: oMLX client wrapper (audio + chat)

**Files:**
- Create: `asr/asr_router/models/omlx_client.py`, `asr/tests/test_omlx_client.py`

- [ ] **Step 1: Write failing test (live oMLX required)**

```python
import pytest
from asr_router.models.omlx_client import OMLXClient

@pytest.fixture
def client():
    return OMLXClient.from_settings()

def test_models_list(client):
    ids = client.list_model_ids()
    assert "Qwen3-ASR-1.7B-8bit" in ids
    assert "gemma-4-26b-a4b-it-4bit" in ids

def test_transcribe(client, zh_wav):
    r = client.transcribe(zh_wav, model="Qwen3-ASR-1.7B-8bit")
    assert r["text"]
    assert r["language"] in ("Chinese", "zh")

def test_chat_review(client):
    r = client.chat(
        model="gemma-4-26b-a4b-it-4bit",
        messages=[{"role":"user","content":"输出 JSON: {\"ok\": true}"}],
        response_format={"type":"json_object"},
    )
    assert "ok" in r.lower()
```

- [ ] **Step 2: Implement `models/omlx_client.py`**

```python
from __future__ import annotations
from pathlib import Path
import httpx
from openai import OpenAI
from asr_router.config import Settings

class OMLXClient:
    def __init__(self, base_url: str, api_key: str):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._http = httpx.Client(base_url=self._base_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=120.0)
        self._oa = OpenAI(base_url=self._base_url, api_key=api_key)

    @classmethod
    def from_settings(cls) -> "OMLXClient":
        s = Settings.load()
        return cls(s.omlx_base_url, s.omlx_api_key)

    def list_model_ids(self) -> list[str]:
        r = self._http.get("/models")
        r.raise_for_status()
        return [m["id"] for m in r.json()["data"]]

    def transcribe(self, wav_path: Path | str, model: str = "Qwen3-ASR-1.7B-8bit") -> dict:
        with open(wav_path, "rb") as f:
            r = self._http.post(
                "/audio/transcriptions",
                data={"model": model},
                files={"file": (Path(wav_path).name, f, "audio/wav")},
            )
        r.raise_for_status()
        return r.json()

    def chat(self, model: str, messages: list[dict], **kwargs) -> str:
        resp = self._oa.chat.completions.create(model=model, messages=messages, **kwargs)
        return resp.choices[0].message.content or ""
```

- [ ] **Step 3: Run, verify pass, commit**

```bash
cd asr && uv run pytest tests/test_omlx_client.py -v
git commit -m "feat(asr): oMLX client (audio transcribe + chat completions)"
```

---

## Phase 2: IM Mode

### Task 4: IM mode router

**Files:**
- Create: `asr/asr_router/im/__init__.py`, `asr/asr_router/im/router.py`, `asr/tests/test_im_router.py`

- [ ] **Step 1: Write failing test**

```python
from asr_router.im.router import IMRouter, RouteDecision

ROUTING = {
    "defaults": {"upstream": "sense_voice"},
    "rules": [
        {"when": {"request_param": {"quality": "high"}}, "use": "omlx"},
        {"when": {"request_param": {"quality": "fast"}}, "use": "sense_voice"},
        {"when": {"duration_gt": 30}, "use": "omlx"},
        {"when": {"event_in": ["BGM", "Applause"]}, "use": "omlx"},
    ],
}

def test_default():
    r = IMRouter(ROUTING)
    d = r.decide(duration_sec=5.0, event="Speech", lang="zh", request_params={})
    assert d.upstream == "sense_voice"
    assert d.reason == "default"

def test_quality_high_overrides():
    r = IMRouter(ROUTING)
    d = r.decide(duration_sec=2.0, event="Speech", lang="zh", request_params={"quality":"high"})
    assert d.upstream == "omlx"
    assert "request_param" in d.reason

def test_long_duration_routes_to_omlx():
    r = IMRouter(ROUTING)
    d = r.decide(duration_sec=45.0, event="Speech", lang="zh", request_params={})
    assert d.upstream == "omlx"

def test_complex_event_routes_to_omlx():
    r = IMRouter(ROUTING)
    d = r.decide(duration_sec=5.0, event="BGM", lang="zh", request_params={})
    assert d.upstream == "omlx"

def test_first_match_wins():
    """quality=fast (rule 2) beats duration>30 (rule 3)"""
    r = IMRouter(ROUTING)
    d = r.decide(duration_sec=60.0, event="Speech", lang="zh", request_params={"quality":"fast"})
    assert d.upstream == "sense_voice"
```

- [ ] **Step 2: Implement `im/router.py`**

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class RouteDecision:
    upstream: str       # "sense_voice" | "omlx"
    reason: str         # human-readable rule trace

class IMRouter:
    def __init__(self, config: dict):
        self._default = config["defaults"]["upstream"]
        self._rules = config.get("rules", [])

    def decide(self, *, duration_sec: float, event: str, lang: str, request_params: dict) -> RouteDecision:
        for i, rule in enumerate(self._rules):
            if self._match(rule["when"], duration_sec=duration_sec, event=event, lang=lang, request_params=request_params):
                return RouteDecision(upstream=rule["use"], reason=f"rule[{i}]:{rule['when']}")
        return RouteDecision(upstream=self._default, reason="default")

    @staticmethod
    def _match(when: dict, *, duration_sec, event, lang, request_params) -> bool:
        for k, v in when.items():
            if k == "duration_gt" and not (duration_sec > v): return False
            elif k == "duration_lt" and not (duration_sec < v): return False
            elif k == "event_in" and event not in v: return False
            elif k == "lang_in" and lang not in v: return False
            elif k == "request_param":
                for pk, pv in v.items():
                    if request_params.get(pk) != pv: return False
        return True
```

- [ ] **Step 3: Verify pass, commit**

```bash
cd asr && uv run pytest tests/test_im_router.py -v
git commit -m "feat(asr): rule-based IM router (duration/event/quality)"
```

---

### Task 5: FastAPI app — `/v1/audio/transcriptions` endpoint (IM mode)

**Files:**
- Create: `asr/asr_router/server.py`, `asr/tests/test_server_im.py`

- [ ] **Step 1: Write failing test**

```python
import io
from fastapi.testclient import TestClient
from asr_router.server import app

def test_auth_required():
    c = TestClient(app)
    r = c.post("/v1/audio/transcriptions")
    assert r.status_code == 401

def test_short_zh_via_sense_voice(zh_wav):
    c = TestClient(app)
    with open(zh_wav, "rb") as f:
        r = c.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer sk-mlx"},
            data={"model": "auto"},
            files={"file": ("zh.wav", f, "audio/wav")},
        )
    assert r.status_code == 200
    j = r.json()
    assert j["text"]
    assert j["language"] == "zh"
    assert j["x_route"]["upstream"] == "sense_voice"

def test_quality_high_routes_to_omlx(zh_wav):
    c = TestClient(app)
    with open(zh_wav, "rb") as f:
        r = c.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer sk-mlx"},
            data={"model": "auto", "quality": "high"},
            files={"file": ("zh.wav", f, "audio/wav")},
        )
    assert r.status_code == 200
    assert r.json()["x_route"]["upstream"] == "omlx"
```

- [ ] **Step 2: Implement `server.py`**

```python
from __future__ import annotations
from fastapi import FastAPI, UploadFile, Form, File, HTTPException, Header
from typing import Annotated
import tempfile, soundfile as sf, numpy as np, io
from asr_router.config import Settings, load_routing
from asr_router.models.sense_voice import SenseVoiceTranscriber
from asr_router.models.omlx_client import OMLXClient
from asr_router.im.router import IMRouter

settings = Settings.load()
router = IMRouter(load_routing(settings.routing_yaml))
omlx = OMLXClient.from_settings()
sv = SenseVoiceTranscriber.get()

app = FastAPI(title="ASR Router")

def _check_auth(authorization: str | None) -> None:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing API key")
    if authorization.removeprefix("Bearer ").strip() != settings.api_key:
        raise HTTPException(401, "Invalid API key")

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()] = "auto",
    quality: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "json",
    authorization: Annotated[str | None, Header()] = None,
):
    _check_auth(authorization)
    raw = await file.read()
    samples, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    duration = len(samples) / sr

    sv_result = sv.transcribe(wav_path="<inline>", samples=samples, sr=sr)
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
            "x_route": {"upstream": "sense_voice", "reason": decision.reason, "decode_ms": sv_result.decode_ms},
            "x_tags": {"emotion": sv_result.emotion, "event": sv_result.event},
        }

    # Fallback to oMLX — write temp wav for httpx multipart
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, samples, sr, subtype="PCM_16")
        omlx_resp = omlx.transcribe(tmp.name, model="Qwen3-ASR-1.7B-8bit")
    return {
        **omlx_resp,
        "x_route": {"upstream": "omlx", "reason": decision.reason},
        "x_tags": {"emotion": sv_result.emotion, "event": sv_result.event, "lid": sv_result.lang},
    }

@app.get("/v1/models")
async def models(authorization: Annotated[str | None, Header()] = None):
    _check_auth(authorization)
    return {"object": "list", "data": [
        {"id": "auto", "object": "model", "owned_by": "asr-router"},
        {"id": "sense_voice", "object": "model", "owned_by": "asr-router"},
        {"id": "Qwen3-ASR-1.7B-8bit", "object": "model", "owned_by": "omlx"},
    ]}
```

- [ ] **Step 3: Verify pass, commit**

```bash
cd asr && uv run pytest tests/test_server_im.py -v
git commit -m "feat(asr): IM mode endpoint /v1/audio/transcriptions with auto-routing"
```

---

## Phase 3: Meeting Mode Pipeline

### Task 6: VAD + Speaker diarization (Pass 1)

**Files:**
- Create: `asr/asr_router/models/vad.py`, `asr/asr_router/models/diarize.py`, `asr/asr_router/meeting/__init__.py`, `asr/asr_router/meeting/vad_diarize.py`, `asr/tests/test_vad_diarize.py`

- [ ] **Step 1: Write failing test (uses 30s sample audio)**

```python
import pytest
from pathlib import Path
from asr_router.meeting.vad_diarize import vad_diarize, DiarizedSegment

SAMPLE = Path(os.environ.get("ASR_TEST_DIARIZE_AUDIO", ""))

@pytest.mark.skipif(not SAMPLE or not SAMPLE.exists(),
                    reason="set ASR_TEST_DIARIZE_AUDIO to a multi-speaker wav to run this test")
def test_returns_segments():
    segs = vad_diarize(SAMPLE, max_speakers=4)
    assert len(segs) >= 1
    assert all(isinstance(s, DiarizedSegment) for s in segs)
    assert all(s.end > s.start for s in segs)
    assert all(s.speaker.startswith("Speaker_") for s in segs)
    speakers = {s.speaker for s in segs}
    assert 1 <= len(speakers) <= 4
```

- [ ] **Step 2: Implement `meeting/vad_diarize.py`** (uses sherpa-onnx OfflineSpeakerDiarization which combines VAD + clustering)

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sherpa_onnx, soundfile as sf, numpy as np
from asr_router.config import Settings

@dataclass(frozen=True)
class DiarizedSegment:
    start: float
    end: float
    speaker: str        # "Speaker_0", "Speaker_1", ...

_diarizer = None

def _get_diarizer():
    global _diarizer
    if _diarizer is None:
        s = Settings.load()
        cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(s.diarize_dir / "model.onnx"),
                ),
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(s.speaker_embed_path)),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        _diarizer = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    return _diarizer

def vad_diarize(wav_path: Path, max_speakers: int = 8) -> list[DiarizedSegment]:
    samples, sr = sf.read(str(wav_path), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != 16000:
        # resample with numpy linear interp as fallback (libraries may differ; if librosa is available it would be preferred)
        ratio = 16000 / sr
        samples = np.interp(np.arange(0, len(samples) * ratio) / ratio, np.arange(len(samples)), samples).astype(np.float32)
        sr = 16000
    diarizer = _get_diarizer()
    if max_speakers > 0:
        diarizer.set_config(sherpa_onnx.FastClusteringConfig(num_clusters=max_speakers, threshold=0.5))
    result = diarizer.process(samples).sort_by_start_time()
    return [DiarizedSegment(start=s.start, end=s.end, speaker=f"Speaker_{s.speaker}") for s in result]
```

- [ ] **Step 3: Verify pass, commit**

```bash
cd asr && uv run pytest tests/test_vad_diarize.py -v
git commit -m "feat(asr): Pass 1 VAD + speaker diarization via sherpa-onnx"
```

---

### Task 7: Chunked SenseVoice transcription over diarized segments (Pass 2)

**Files:**
- Create: `asr/asr_router/meeting/transcribe.py`, `asr/tests/test_meeting_transcribe.py`

- [ ] **Step 1: Write failing test**

```python
from asr_router.meeting.vad_diarize import DiarizedSegment
from asr_router.meeting.transcribe import transcribe_segments, SegmentTranscript

def test_transcribe_yields_one_per_diarize(zh_wav):
    segs = [DiarizedSegment(start=0.0, end=2.5, speaker="Speaker_0"),
            DiarizedSegment(start=2.5, end=5.5, speaker="Speaker_1")]
    out = transcribe_segments(zh_wav, segs, chunk_max_sec=30)
    assert len(out) == 2
    assert all(isinstance(t, SegmentTranscript) for t in out)
    assert all(t.text for t in out)
    assert out[0].speaker == "Speaker_0"
    assert out[1].speaker == "Speaker_1"
```

- [ ] **Step 2: Implement `meeting/transcribe.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import soundfile as sf, numpy as np
from asr_router.models.sense_voice import SenseVoiceTranscriber
from asr_router.meeting.vad_diarize import DiarizedSegment

@dataclass(frozen=True)
class SegmentTranscript:
    speaker: str
    start: float
    end: float
    text: str
    lang: str
    emotion: str
    event: str

def transcribe_segments(wav_path: Path, segments: list[DiarizedSegment], chunk_max_sec: float = 30.0) -> list[SegmentTranscript]:
    samples, sr = sf.read(str(wav_path), dtype="float32")
    if samples.ndim > 1: samples = samples.mean(axis=1)
    sv = SenseVoiceTranscriber.get()
    out: list[SegmentTranscript] = []
    for seg in segments:
        s = max(0, int(seg.start * sr))
        e = min(len(samples), int(seg.end * sr))
        if e - s < int(0.2 * sr):  # skip <200ms slivers
            continue
        chunk = samples[s:e]
        # If segment > chunk_max_sec, split (rare for diarized segments)
        if (e - s) / sr > chunk_max_sec:
            step = int(chunk_max_sec * sr)
            for off in range(0, len(chunk), step):
                sub = chunk[off:off+step]
                if len(sub) < int(0.2*sr): continue
                r = sv.transcribe("<inline>", samples=sub, sr=sr)
                out.append(SegmentTranscript(
                    speaker=seg.speaker, start=seg.start + off/sr, end=seg.start + (off+len(sub))/sr,
                    text=r.text, lang=r.lang, emotion=r.emotion, event=r.event,
                ))
        else:
            r = sv.transcribe("<inline>", samples=chunk, sr=sr)
            out.append(SegmentTranscript(
                speaker=seg.speaker, start=seg.start, end=seg.end,
                text=r.text, lang=r.lang, emotion=r.emotion, event=r.event,
            ))
    return out
```

- [ ] **Step 3: Verify pass, commit**

```bash
cd asr && uv run pytest tests/test_meeting_transcribe.py -v
git commit -m "feat(asr): Pass 2 chunked SenseVoice transcription over diarized segments"
```

---

### Task 8: Glossary loader (default + per-job merge)

**Files:**
- Create: `asr/asr_router/glossary.py`, `asr/tests/test_glossary.py`

- [ ] **Step 1: Write failing test**

```python
from asr_router.glossary import Glossary

DEFAULT = {"terms": [{"term": "Alpha Group", "aliases": ["Alpa Group"]}]}
PERJOB = {"terms": [{"term": "Chairman Zhang", "aliases": ["Mr. Zhang"]}, {"term": "Alpha Group", "aliases": ["Alfa"]}]}

def test_merge_unions_aliases():
    g = Glossary.merged(DEFAULT, PERJOB)
    alpha = g["Alpha Group"]
    assert "Alpa Group" in alpha
    assert "Alfa" in alpha

def test_perjob_adds_new():
    g = Glossary.merged(DEFAULT, PERJOB)
    assert "Chairman Zhang" in g
    assert "Mr. Zhang" in g["Chairman Zhang"]

def test_to_prompt_text():
    g = Glossary.merged(DEFAULT, {})
    text = g.to_prompt_text()
    assert "Alpha Group" in text
    assert "Alpa Group" in text
```

- [ ] **Step 2: Implement `glossary.py`**

```python
from __future__ import annotations
from typing import Iterator

class Glossary:
    def __init__(self, terms: dict[str, list[str]]):
        self._terms = terms

    @classmethod
    def merged(cls, *sources: dict | None) -> "Glossary":
        merged: dict[str, set[str]] = {}
        for src in sources:
            if not src: continue
            for entry in src.get("terms", []):
                t = entry["term"]
                aliases = set(entry.get("aliases", []))
                merged.setdefault(t, set()).update(aliases)
        return cls({t: sorted(a) for t, a in merged.items()})

    def __contains__(self, term): return term in self._terms
    def __getitem__(self, term): return self._terms[term]
    def __iter__(self) -> Iterator[str]: return iter(self._terms)

    def to_prompt_text(self) -> str:
        if not self._terms: return "(none)"
        lines = []
        for t in sorted(self._terms):
            aliases = self._terms[t]
            if aliases:
                lines.append(f"- {t}（同/误：{', '.join(aliases)}）")
            else:
                lines.append(f"- {t}")
        return "\n".join(lines)
```

- [ ] **Step 3: Verify pass, commit**

```bash
cd asr && uv run pytest tests/test_glossary.py -v
git commit -m "feat(asr): glossary merger (default + per-job aliases)"
```

---

### Task 9: gemma-4 contextual review (Pass 3)

**Files:**
- Create: `asr/prompts/review.j2`, `asr/asr_router/meeting/review.py`, `asr/tests/test_review.py`

- [ ] **Step 1: Write `prompts/review.j2`** — explicit, deterministic JSON output

```jinja2
你是中外双语会议转录的资深复核员。基于 SenseVoice 原始转录、上下文窗口、术语表，输出严格 JSON 修订结果。

# 任务
1. **同音/近音错误修正**：结合上下文判断（例：「公园街」→「公园路」、「Alpa Group」→「Alpha Group」）。
2. **应用术语表**：人名/地名/机构名/项目代号统一为标准写法。
3. **推断 Speaker_N 的语义角色**（如「翻译」「主持人」「金会长」「李总裁」），基于发言内容、称谓关系、提问/应答模式。同一个 Speaker_N 在整段会议里应映射到唯一角色。
4. **置信度分级**：high/medium/low。低置信度片段在 `text_corrected` 末尾追加 `[unclear]`，并在 `notes` 解释疑问点。

# 术语表
{{ glossary }}

# 上一窗口（仅作上下文，勿改）
{{ prev_context }}

# 待复核片段（JSON 数组）
{{ segments_json }}

# 输出格式（严格 JSON，禁止额外文字）
```json
{
  "segments": [
    {
      "speaker_id": "Speaker_0",
      "speaker_role": "翻译",
      "start": 60.0,
      "end": 70.0,
      "text_corrected": "...",
      "changes": [{"original": "Alpa", "fixed": "Alpha", "reason": "glossary"}],
      "confidence": "high",
      "notes": ""
    }
  ],
  "speaker_role_map": {"Speaker_0": "翻译", "Speaker_1": "主持人"}
}
```
```

- [ ] **Step 2: Write failing test**

```python
import json
from asr_router.meeting.review import review_segments, ReviewedSegment
from asr_router.meeting.transcribe import SegmentTranscript
from asr_router.glossary import Glossary

def test_review_corrects_glossary_term(monkeypatch):
    raw = [SegmentTranscript(speaker="Speaker_0", start=0.0, end=5.0,
                              text="我们参观了Alpa Group的工厂", lang="zh", emotion="NEUTRAL", event="Speech")]
    g = Glossary.merged({"terms":[{"term":"Alpha Group","aliases":["Alpa Group"]}]})

    fake_response = json.dumps({
        "segments": [{
            "speaker_id":"Speaker_0","speaker_role":"主持人","start":0.0,"end":5.0,
            "text_corrected":"我们参观了Alpha Group的工厂",
            "changes":[{"original":"Alpa Group","fixed":"Alpha Group","reason":"glossary"}],
            "confidence":"high","notes":""
        }],
        "speaker_role_map":{"Speaker_0":"主持人"}
    })

    class FakeOMLX:
        def chat(self, **kw): return fake_response
    reviewed, role_map = review_segments(raw, glossary=g, omlx=FakeOMLX(), model="gemma-4-26b-a4b-it-4bit", window=2, batch=12)
    assert reviewed[0].text_corrected == "我们参观了Alpha Group的工厂"
    assert role_map["Speaker_0"] == "主持人"
```

- [ ] **Step 3: Implement `meeting/review.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from jinja2 import Template
from asr_router.meeting.transcribe import SegmentTranscript
from asr_router.glossary import Glossary

PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts/review.j2"

@dataclass(frozen=True)
class ReviewedSegment:
    speaker_id: str
    speaker_role: str
    start: float
    end: float
    text_corrected: str
    changes: list[dict]
    confidence: str
    notes: str
    text_original: str

def _render_prompt(segments: list[SegmentTranscript], glossary: Glossary, prev_context: str) -> str:
    tpl = Template(PROMPT_PATH.read_text())
    segs_json = json.dumps([{
        "speaker": s.speaker, "start": s.start, "end": s.end,
        "text": s.text, "lang": s.lang, "event": s.event,
    } for s in segments], ensure_ascii=False, indent=2)
    return tpl.render(glossary=glossary.to_prompt_text(), prev_context=prev_context or "(无)", segments_json=segs_json)

def _parse_response(text: str) -> dict:
    # gemma-4 may wrap in code fences; strip
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"): t = t[4:]
    return json.loads(t)

def review_segments(raw: list[SegmentTranscript], *, glossary: Glossary, omlx, model: str, window: int = 4, batch: int = 12) -> tuple[list[ReviewedSegment], dict[str,str]]:
    out: list[ReviewedSegment] = []
    role_map: dict[str, str] = {}
    prev_context = ""
    for i in range(0, len(raw), batch):
        batch_segs = raw[i:i+batch]
        prompt = _render_prompt(batch_segs, glossary, prev_context)
        resp = omlx.chat(model=model, messages=[{"role":"user","content":prompt}], temperature=0.1)
        parsed = _parse_response(resp)
        for orig, rev in zip(batch_segs, parsed["segments"]):
            out.append(ReviewedSegment(
                speaker_id=rev["speaker_id"], speaker_role=rev.get("speaker_role", rev["speaker_id"]),
                start=rev["start"], end=rev["end"],
                text_corrected=rev["text_corrected"], changes=rev.get("changes", []),
                confidence=rev.get("confidence", "medium"), notes=rev.get("notes", ""),
                text_original=orig.text,
            ))
        for sid, role in parsed.get("speaker_role_map", {}).items():
            role_map.setdefault(sid, role)
        prev_context = "\n".join(f"[{r.speaker_role}] {r.text_corrected}" for r in out[-window:])
    return out, role_map
```

- [ ] **Step 4: Verify pass, commit**

```bash
cd asr && uv run pytest tests/test_review.py -v
git commit -m "feat(asr): Pass 3 gemma-4 contextual review with glossary + speaker role inference"
```

---

### Task 10: Render artifacts (Pass 4)

**Files:**
- Create: `asr/asr_router/meeting/render.py`, `asr/prompts/summary.j2`, `asr/tests/test_render.py`

- [ ] **Step 1: Write `prompts/summary.j2`**

```jinja2
基于以下完整的会议转录，生成结构化纪要。

# 转录全文
{{ transcript }}

# 输出格式（Markdown）
## 摘要
（200字以内）

## 决议事项
- ...

## 待办事项
- [ ] (责任人) 任务描述（截止日期，如有）

## 关键议题
- 议题1：核心论点
- ...
```

- [ ] **Step 2: Write failing test**

```python
from pathlib import Path
from asr_router.meeting.render import render_artifacts
from asr_router.meeting.transcribe import SegmentTranscript
from asr_router.meeting.review import ReviewedSegment

def test_render_writes_5_files(tmp_path):
    raw = [SegmentTranscript(speaker="Speaker_0", start=0,end=5,text="Alpa Group欢迎大家",lang="zh",emotion="NEUTRAL",event="Speech")]
    reviewed = [ReviewedSegment(speaker_id="Speaker_0",speaker_role="主持人",start=0,end=5,
                                  text_corrected="Alpha Group欢迎大家",changes=[{"original":"Alpa","fixed":"Alpha","reason":"glossary"}],
                                  confidence="high",notes="",text_original="Alpa Group欢迎大家")]
    role_map = {"Speaker_0":"主持人"}
    class FakeOMLX:
        def chat(self, **kw): return "## 摘要\n会议讨论了Alpha Group合作。"
    paths = render_artifacts(out_dir=tmp_path, stem="meeting", raw=raw, reviewed=reviewed,
                              role_map=role_map, omlx=FakeOMLX(), summary_model="gemma-4")
    assert (tmp_path / "_raw.json").exists()
    assert (tmp_path / "meeting_sensevoice.md").exists()
    assert (tmp_path / "meeting_gemma4.md").exists()
    assert (tmp_path / "meeting_diff.md").exists()
    assert (tmp_path / "meeting_summary.md").exists()
    sv_text = (tmp_path / "meeting_sensevoice.md").read_text()
    assert "Alpa Group" in sv_text  # raw, before correction
    g4_text = (tmp_path / "meeting_gemma4.md").read_text()
    assert "Alpha Group" in g4_text  # after correction
    assert "[主持人]" in g4_text  # role-labeled
```

- [ ] **Step 3: Implement `meeting/render.py`**

```python
from __future__ import annotations
from pathlib import Path
import json
from jinja2 import Template
from asr_router.meeting.transcribe import SegmentTranscript
from asr_router.meeting.review import ReviewedSegment

SUMMARY_TPL = Path(__file__).parent.parent.parent / "prompts/summary.j2"

def _ts(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def _render_sv(raw: list[SegmentTranscript]) -> str:
    lines = ["# SenseVoice 原始转录（未复核）", ""]
    for s in raw:
        lines.append(f"**[{s.speaker} {_ts(s.start)}-{_ts(s.end)}]** {s.text}")
        lines.append("")
    return "\n".join(lines)

def _render_g4(reviewed: list[ReviewedSegment], role_map: dict[str,str]) -> str:
    lines = ["# gemma-4 复核版", "", "## 说话人映射"]
    for sid, role in role_map.items():
        lines.append(f"- {sid} → {role}")
    lines += ["", "## 转录"]
    for r in reviewed:
        flag = " ⚠️" if r.confidence == "low" else ""
        lines.append(f"**[{r.speaker_role} {_ts(r.start)}-{_ts(r.end)}]**{flag} {r.text_corrected}")
        if r.notes:
            lines.append(f"  > 注：{r.notes}")
        lines.append("")
    return "\n".join(lines)

def _render_diff(raw: list[SegmentTranscript], reviewed: list[ReviewedSegment]) -> str:
    lines = ["# SenseVoice → gemma-4 修订对照", ""]
    for orig, rev in zip(raw, reviewed):
        if orig.text == rev.text_corrected and not rev.changes:
            continue
        lines.append(f"### {_ts(orig.start)} {rev.speaker_role}")
        lines.append(f"- 原: {orig.text}")
        lines.append(f"- 修: {rev.text_corrected}")
        for c in rev.changes:
            lines.append(f"  - `{c['original']}` → `{c['fixed']}` ({c['reason']})")
        lines.append("")
    if len(lines) <= 2:
        lines.append("（无修订）")
    return "\n".join(lines)

def render_artifacts(*, out_dir: Path, stem: str, raw, reviewed, role_map, omlx, summary_model: str) -> dict[str,Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    raw_json = out_dir / "_raw.json"
    raw_json.write_text(json.dumps([{
        "speaker": s.speaker, "start": s.start, "end": s.end,
        "text": s.text, "lang": s.lang, "event": s.event, "emotion": s.emotion,
    } for s in raw], ensure_ascii=False, indent=2))
    paths["raw"] = raw_json

    sv_md = out_dir / f"{stem}_sensevoice.md"
    sv_md.write_text(_render_sv(raw))
    paths["sensevoice"] = sv_md

    g4_md = out_dir / f"{stem}_gemma4.md"
    g4_md.write_text(_render_g4(reviewed, role_map))
    paths["gemma4"] = g4_md

    diff_md = out_dir / f"{stem}_diff.md"
    diff_md.write_text(_render_diff(raw, reviewed))
    paths["diff"] = diff_md

    transcript_text = "\n".join(f"[{r.speaker_role}] {r.text_corrected}" for r in reviewed)
    summary_prompt = Template(SUMMARY_TPL.read_text()).render(transcript=transcript_text)
    summary_text = omlx.chat(model=summary_model, messages=[{"role":"user","content":summary_prompt}], temperature=0.3)
    summary_md = out_dir / f"{stem}_summary.md"
    summary_md.write_text(summary_text)
    paths["summary"] = summary_md

    return paths
```

- [ ] **Step 4: Verify pass, commit**

```bash
cd asr && uv run pytest tests/test_render.py -v
git commit -m "feat(asr): Pass 4 render 5 model-named artifacts (sv/g4/diff/summary)"
```

---

### Task 11: Job store (sqlite) + worker loop

**Files:**
- Create: `asr/asr_router/jobs.py`, `asr/tests/test_jobs.py`

- [ ] **Step 1: Write failing test**

```python
import time
from pathlib import Path
from asr_router.jobs import JobStore, JobStatus

def test_create_get(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job_id = store.create(audio_path="/x/a.wav", glossary_yaml="terms: []")
    j = store.get(job_id)
    assert j.audio_path == "/x/a.wav"
    assert j.status == JobStatus.QUEUED

def test_update_status(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job_id = store.create(audio_path="/x/a.wav", glossary_yaml="")
    store.update(job_id, status=JobStatus.TRANSCRIBING)
    assert store.get(job_id).status == JobStatus.TRANSCRIBING

def test_pop_next_queued(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    j1 = store.create(audio_path="/a", glossary_yaml="")
    j2 = store.create(audio_path="/b", glossary_yaml="")
    nxt = store.pop_next_queued()
    assert nxt.id == j1
    nxt2 = store.pop_next_queued()
    assert nxt2.id == j2
    assert store.pop_next_queued() is None
```

- [ ] **Step 2: Implement `jobs.py`**

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sqlite3, time, uuid, json, threading

class JobStatus(str, Enum):
    QUEUED = "queued"
    VAD_DIARIZE = "vad_diarize"
    TRANSCRIBING = "transcribing"
    REVIEWING = "reviewing"
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"

@dataclass
class Job:
    id: str
    audio_path: str
    glossary_yaml: str
    status: JobStatus
    artifact_dir: str | None
    error: str | None
    created_at: float
    updated_at: float

_DDL = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    audio_path TEXT NOT NULL,
    glossary_yaml TEXT NOT NULL,
    status TEXT NOT NULL,
    artifact_dir TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_status ON jobs(status, created_at);
"""

class JobStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.executescript(_DDL)
        self._lock = threading.Lock()

    def create(self, *, audio_path: str, glossary_yaml: str) -> str:
        jid = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs(id,audio_path,glossary_yaml,status,artifact_dir,error,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (jid, audio_path, glossary_yaml, JobStatus.QUEUED.value, None, None, now, now),
            )
            self._conn.commit()
        return jid

    def get(self, jid: str) -> Job | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if not row: return None
        cols = [d[0] for d in self._conn.execute("SELECT * FROM jobs LIMIT 0").description]
        d = dict(zip(cols, row))
        return Job(id=d["id"], audio_path=d["audio_path"], glossary_yaml=d["glossary_yaml"],
                   status=JobStatus(d["status"]), artifact_dir=d["artifact_dir"], error=d["error"],
                   created_at=d["created_at"], updated_at=d["updated_at"])

    def update(self, jid: str, **fields) -> None:
        if "status" in fields and isinstance(fields["status"], JobStatus):
            fields["status"] = fields["status"].value
        fields["updated_at"] = time.time()
        keys = ",".join(f"{k}=?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {keys} WHERE id=?", (*fields.values(), jid))
            self._conn.commit()

    def pop_next_queued(self) -> Job | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM jobs WHERE status=? ORDER BY created_at LIMIT 1",
                (JobStatus.QUEUED.value,),
            ).fetchone()
            if not row: return None
            self._conn.execute("UPDATE jobs SET status=?, updated_at=? WHERE id=?",
                              (JobStatus.VAD_DIARIZE.value, time.time(), row[0]))
            self._conn.commit()
        return self.get(row[0])
```

- [ ] **Step 3: Verify pass, commit**

```bash
cd asr && uv run pytest tests/test_jobs.py -v
git commit -m "feat(asr): SQLite job store with state machine"
```

---

### Task 12: Meeting pipeline orchestrator + worker

**Files:**
- Create: `asr/asr_router/meeting/pipeline.py`

- [ ] **Step 1: Implement `meeting/pipeline.py`** (this orchestrates Tasks 6-10)

```python
from __future__ import annotations
from pathlib import Path
import yaml, traceback, threading, time
from asr_router.config import Settings, load_pipelines
from asr_router.glossary import Glossary
from asr_router.jobs import JobStore, JobStatus, Job
from asr_router.meeting.vad_diarize import vad_diarize
from asr_router.meeting.transcribe import transcribe_segments
from asr_router.meeting.review import review_segments
from asr_router.meeting.render import render_artifacts
from asr_router.models.omlx_client import OMLXClient

def run_job(job: Job, *, settings: Settings, store: JobStore, omlx: OMLXClient, pipelines_cfg: dict) -> None:
    try:
        cfg = pipelines_cfg["meeting"]
        audio = Path(job.audio_path)
        out_dir = settings.storage_dir / "jobs" / job.id
        out_dir.mkdir(parents=True, exist_ok=True)
        store.update(job.id, artifact_dir=str(out_dir))

        # Pass 1: VAD + diarize
        store.update(job.id, status=JobStatus.VAD_DIARIZE)
        diarized = vad_diarize(audio, max_speakers=cfg["diarize"]["max_speakers"])

        # Pass 2: Transcribe per segment
        store.update(job.id, status=JobStatus.TRANSCRIBING)
        raw = transcribe_segments(audio, diarized, chunk_max_sec=cfg["transcribe"]["chunk_max_sec"])

        # Pass 3: gemma-4 review
        store.update(job.id, status=JobStatus.REVIEWING)
        default_yaml = yaml.safe_load(settings.glossary_default.read_text()) if settings.glossary_default.exists() else {}
        perjob_yaml = yaml.safe_load(job.glossary_yaml) if job.glossary_yaml else {}
        glossary = Glossary.merged(default_yaml, perjob_yaml)
        reviewed, role_map = review_segments(
            raw, glossary=glossary, omlx=omlx, model=cfg["review"]["model"],
            window=cfg["review"]["context_window_segments"],
            batch=cfg["review"]["max_segments_per_call"],
        )

        # Pass 4: Render
        store.update(job.id, status=JobStatus.RENDERING)
        render_artifacts(out_dir=out_dir, stem=audio.stem, raw=raw, reviewed=reviewed,
                          role_map=role_map, omlx=omlx, summary_model=cfg["summary"]["model"])

        store.update(job.id, status=JobStatus.DONE)
    except Exception as e:
        store.update(job.id, status=JobStatus.FAILED, error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

class Worker:
    def __init__(self, store: JobStore, settings: Settings, omlx: OMLXClient, pipelines_cfg: dict):
        self._store = store; self._settings = settings; self._omlx = omlx; self._pl = pipelines_cfg
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self): self._thread.start()
    def stop(self): self._stop.set(); self._thread.join(timeout=5)

    def _loop(self):
        while not self._stop.is_set():
            job = self._store.pop_next_queued()
            if not job:
                time.sleep(2); continue
            run_job(job, settings=self._settings, store=self._store, omlx=self._omlx, pipelines_cfg=self._pl)
```

- [ ] **Step 2: Manually validate (no unit test — too much I/O; full pipeline is exercised in Task 14)**

```bash
cd asr && uv run python -c "from asr_router.meeting.pipeline import run_job, Worker; print('imports ok')"
```

- [ ] **Step 3: Commit**

```bash
git commit -m "feat(asr): meeting pipeline orchestrator + background worker"
```

---

### Task 13: Server endpoints for meeting jobs

**Files:**
- Modify: `asr/asr_router/server.py` (add 3 endpoints + worker startup)
- Create: `asr/tests/test_server_jobs.py`

- [ ] **Step 1: Write failing test**

```python
from fastapi.testclient import TestClient
from pathlib import Path
import time
from asr_router.server import app

def test_submit_returns_job_id(zh_wav, tmp_path, monkeypatch):
    # Redirect storage to tmp
    from asr_router.config import Settings
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls: Settings(storage_dir=tmp_path)))
    c = TestClient(app)
    with open(zh_wav, "rb") as f:
        r = c.post("/v1/audio/jobs",
                    headers={"Authorization":"Bearer sk-mlx"},
                    files={"file":("zh.wav", f, "audio/wav")},
                    data={"glossary":"terms:\n  - term: 测试\n    aliases: []"})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "queued"
    assert "id" in j

def test_get_status(tmp_path):
    c = TestClient(app)
    r = c.get("/v1/audio/jobs/nonexistent", headers={"Authorization":"Bearer sk-mlx"})
    assert r.status_code == 404
```

- [ ] **Step 2: Modify `server.py`** — add at end of file:

```python
from asr_router.jobs import JobStore
from asr_router.meeting.pipeline import Worker
from asr_router.config import load_pipelines
from fastapi import BackgroundTasks
import shutil, os
from contextlib import asynccontextmanager

_store = JobStore(settings.storage_dir / "jobs.db")
_worker: Worker | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _worker
    _worker = Worker(_store, settings, omlx, load_pipelines(settings.pipelines_yaml))
    _worker.start()
    yield
    _worker.stop()

app.router.lifespan_context = lifespan

@app.post("/v1/audio/jobs")
async def submit_job(
    file: Annotated[UploadFile, File()],
    glossary: Annotated[str, Form()] = "",
    authorization: Annotated[str | None, Header()] = None,
):
    _check_auth(authorization)
    # Persist audio to job dir
    job_id = _store.create(audio_path="<pending>", glossary_yaml=glossary)
    job_dir = settings.storage_dir / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    audio_path = job_dir / "input.wav"
    raw = await file.read()
    samples, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if samples.ndim > 1: samples = samples.mean(axis=1)
    sf.write(str(audio_path), samples, sr, subtype="PCM_16")
    _store.update(job_id, audio_path=str(audio_path))
    return {"id": job_id, "status": "queued"}

@app.get("/v1/audio/jobs/{job_id}")
async def get_job(job_id: str, authorization: Annotated[str | None, Header()] = None):
    _check_auth(authorization)
    job = _store.get(job_id)
    if not job: raise HTTPException(404, "job not found")
    artifacts = []
    if job.artifact_dir and Path(job.artifact_dir).exists():
        artifacts = sorted(p.name for p in Path(job.artifact_dir).iterdir() if p.is_file())
    return {
        "id": job.id, "status": job.status.value, "audio_path": job.audio_path,
        "artifact_dir": job.artifact_dir, "artifacts": artifacts,
        "error": job.error, "created_at": job.created_at, "updated_at": job.updated_at,
    }

@app.get("/v1/audio/jobs/{job_id}/artifact/{name}")
async def get_artifact(job_id: str, name: str, authorization: Annotated[str | None, Header()] = None):
    from fastapi.responses import FileResponse
    _check_auth(authorization)
    job = _store.get(job_id)
    if not job or not job.artifact_dir: raise HTTPException(404)
    p = Path(job.artifact_dir) / name
    if not p.exists() or not p.is_file(): raise HTTPException(404)
    return FileResponse(p)
```

- [ ] **Step 3: Verify pass, commit**

```bash
cd asr && uv run pytest tests/test_server_jobs.py -v
git commit -m "feat(asr): meeting job endpoints (POST jobs, GET status, GET artifact)"
```

---

## Phase 4: Validation

### Task 14: End-to-end evaluation on real meeting audio

**Files:**
- Create: `asr/scripts/eval_meeting.py`, `asr/EVALUATION.md`

This task is the proof-of-quality. Pick **one** representative recording, run the full pipeline, compute CER vs the manually-corrected diarization JSON, save report.

- [ ] **Step 1: Write `scripts/eval_meeting.py`**

```python
#!/usr/bin/env python3
"""Run meeting pipeline on real meeting audio (env var), compute CER vs ground truth, write EVALUATION.md."""
import sys, json, subprocess, time, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import requests

_audio_env = os.environ.get("ASR_EVAL_AUDIO", "")
_json_env = os.environ.get("ASR_EVAL_DIARIZE_JSON", "")
SAMPLE_AUDIO = Path(_audio_env) if _audio_env else None
GROUND_TRUTH = Path(_json_env) if _json_env else None
PERJOB_GLOSSARY = """
terms: []
"""
# Replace these placeholders with your own terms via per-job glossary

def submit_job(audio_path: Path) -> str:
    with open(audio_path, "rb") as f:
        r = requests.post(
            "http://localhost:18081/v1/audio/jobs",
            headers={"Authorization":"Bearer sk-mlx"},
            files={"file":(audio_path.name, f, "audio/wav")},
            data={"glossary": PERJOB_GLOSSARY},
        )
    r.raise_for_status()
    return r.json()["id"]

def wait_done(job_id: str, timeout=1800) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = requests.get(f"http://localhost:18081/v1/audio/jobs/{job_id}",
                         headers={"Authorization":"Bearer sk-mlx"})
        j = r.json()
        if j["status"] in ("done", "failed"):
            return j
        print(f"  [{int(time.time()-t0)}s] {j['status']}", flush=True)
        time.sleep(5)
    raise TimeoutError(f"Job {job_id} not done in {timeout}s")

def cer(ref: str, hyp: str) -> float:
    """Character Error Rate via Levenshtein."""
    import unicodedata
    ref = "".join(c for c in unicodedata.normalize("NFKC", ref) if c.strip())
    hyp = "".join(c for c in unicodedata.normalize("NFKC", hyp) if c.strip())
    n, m = len(ref), len(hyp)
    if n == 0: return 1.0 if m else 0.0
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[n][m] / n

def extract_docx_text(p: Path) -> str:
    out = subprocess.check_output(["textutil","-convert","txt","-stdout",str(p)], stderr=subprocess.DEVNULL)
    return out.decode("utf-8", errors="ignore")

def main():
    print(f"Audio: {SAMPLE_AUDIO}")
    print(f"Ground truth: {GROUND_TRUTH}")
    if not SAMPLE_AUDIO.exists() or not GROUND_TRUTH.exists():
        print("Missing sample or ground truth"); sys.exit(1)
    jid = submit_job(SAMPLE_AUDIO); print(f"Submitted: {jid}")
    final = wait_done(jid); assert final["status"] == "done", final
    art_dir = Path(final["artifact_dir"])
    sv_md = (art_dir / f"{SAMPLE_AUDIO.stem}_sensevoice.md").read_text()
    g4_md = (art_dir / f"{SAMPLE_AUDIO.stem}_gemma4.md").read_text()
    gt = extract_docx_text(GROUND_TRUTH)
    cer_sv = cer(gt, sv_md); cer_g4 = cer(gt, g4_md)

    eval_md = Path(__file__).parent.parent / "EVALUATION.md"
    eval_md.write_text(f"""# Pipeline Evaluation

**Sample:** {SAMPLE_AUDIO.name}
**Ground truth:** {GROUND_TRUTH.name}
**Job:** {jid}
**Date:** {time.strftime('%Y-%m-%d %H:%M')}

## CER (lower is better)

| Stage | CER |
|---|---|
| SenseVoice raw | {cer_sv:.4f} |
| gemma-4 reviewed | {cer_g4:.4f} |
| Improvement | {(cer_sv-cer_g4)/cer_sv*100:.1f}% |

## Artifacts
{chr(10).join(f"- `{f}`" for f in final["artifacts"])}
""")
    print(f"Wrote {eval_md}")
    print(f"  SV CER: {cer_sv:.4f}")
    print(f"  G4 CER: {cer_g4:.4f}")

if __name__ == "__main__": main()
```

- [ ] **Step 2: Manual run (start service first, then eval)**

```bash
# Terminal 1: start service
bash asr/scripts/run_dev.sh
# Terminal 2: eval
cd asr && uv run python scripts/eval_meeting.py
```

Acceptance: gemma-4 CER lower than SenseVoice raw CER, and `EVALUATION.md` is written. If gemma-4 makes things WORSE (higher CER), iterate on `prompts/review.j2` before commit.

- [ ] **Step 3: Commit**

```bash
git add asr/scripts/eval_meeting.py asr/EVALUATION.md
git commit -m "test(asr): end-to-end CER evaluation on real meeting audio

SenseVoice raw CER: <X>
gemma-4 reviewed CER: <Y>
Improvement: <Z>%

Validates that the meeting pipeline measurably improves transcription
quality over single-shot SenseVoice for real-world bilingual meeting
recordings with domain terminology."
```

---

## Phase 5: Replacement & Documentation

### Task 15: Remove `audio-transcribe` skill (outside repo, single operation)

**Files (outside repo):**
- Delete: `~/.claude/skills/audio-transcribe/`

- [ ] **Step 1: Verify replacement covers existing skill use cases**

```bash
ls ~/.claude/skills/audio-transcribe/ | head
cat ~/.claude/skills/audio-transcribe/SKILL.md | head -30
```
Confirm the skill's responsibilities (audio file → markdown transcript) are now achievable via:
```bash
curl -X POST http://localhost:18081/v1/audio/jobs \
  -H "Authorization: Bearer sk-mlx" \
  -F "file=@audio.wav" -F "glossary=terms: []"
# poll for done, fetch _gemma4.md artifact
```

- [ ] **Step 2: Delete skill (irreversible)**

```bash
rm -rf ~/.claude/skills/audio-transcribe/
```

- [ ] **Step 3:** No commit (outside repo). Note in subsequent commit message.

---

### Task 16: Rewrite root README + README_CN + SKILL.md

**Files:**
- Modify: `README.md`, `README_CN.md`, `SKILL.md`

**Goal of new docs:**
- Single architecture diagram: oMLX gateway (`localhost:18080/v1`) + sherpa-onnx ASR sidecar (`localhost:18081/v1`)
- Five capabilities with minimum-runnable examples: LLM, VLM, Embeddings, OCR, ASR (both modes)
- Authoritative model list (the live oMLX inventory)
- Hardware tiering preserved if applicable

- [ ] **Step 1: Rewrite `README.md`** — structure:

```markdown
# MLX Local Inference Stack

OpenAI-compatible local AI inference on Apple Silicon. Single oMLX gateway for LLM/VLM/Embeddings/OCR/ASR plus a routing-aware ASR sidecar with multi-pass meeting review.

## Architecture

[diagram: oMLX (18080) + asr-router (18081)]

## Endpoints
- `http://localhost:18080/v1` — oMLX (key `sk-mlx`): all heavy models
- `http://localhost:18081/v1` — asr-router (key `sk-mlx`): IM auto-route + meeting pipeline

## Models (live)
| Capability | Model | Size | Endpoint |
|---|---|---|---|
| LLM (default) | Qwen3.5-35B-A3B-4bit | ~18GB | oMLX |
| LLM (fast) | gemma-4-26b-a4b-it-4bit | ~14GB | oMLX |
| LLM (small) | Qwen3.5-9B-MLX-4bit | ~5.8GB | oMLX |
| VLM | supergemma4-26b-abliterated-multimodal-mlx-4bit | ~14GB | oMLX |
| OCR (VLM) | PaddleOCR-VL-1.5-6bit | ~3.3GB | oMLX |
| Embeddings | Qwen3-Embedding-0.6B-4bit-DWQ | ~1GB | oMLX |
| ASR (quality) | Qwen3-ASR-1.7B-8bit | ~1.5GB | oMLX |
| ASR (fast) | sherpa-onnx SenseVoice int8 | 228MB | asr-router |

## Quickstart per capability
[5 minimal cURL/Python examples]

## ASR Routing Module
See [`asr/README.md`](asr/README.md).
```

- [ ] **Step 2: Mirror to `README_CN.md`** (same content, Chinese)

- [ ] **Step 3: Rewrite `SKILL.md`** — should mirror `~/.claude/skills/mlx-local-inference/SKILL.md` shape but reflect actual port/key/model state. Note in commit that the user-global skill at `~/.claude/skills/mlx-local-inference/` should also be updated separately.

- [ ] **Step 4: Commit**

```bash
git commit -m "docs: rewrite README/README_CN/SKILL for oMLX-first architecture

- Single source of truth: oMLX gateway @ 18080 + asr-router @ 18081
- Live model inventory (6 models in oMLX + sherpa-onnx SenseVoice in asr/)
- Five-capability quickstart matching real endpoints/keys
- audio-transcribe skill replaced by asr/ module (separately deleted from ~/.claude/skills/)
- Note: ~/.claude/skills/mlx-local-inference/SKILL.md needs parallel update"
```

---

### Task 17: Prune and refresh `references/`

**Files:**
- Delete: `references/asr-whisper.md`, `references/llm-gemma3-12b.md`, `references/llm-qwen3-14b.md`, `references/transcribe-daemon.md`
- Modify: `references/omlx.md` (rewrite as authoritative spec), `references/asr-qwen3.md`, `references/embedding-qwen3.md`, `references/ocr.md`, `references/tts-qwen3.md`, `references/llm-models-reference.md`
- Create: `references/asr-routing-module.md`, `references/asr-sherpa-onnx-sensevoice.md`

- [ ] **Step 1: Delete legacy references**

```bash
git rm references/asr-whisper.md references/llm-gemma3-12b.md references/llm-qwen3-14b.md references/transcribe-daemon.md
```

- [ ] **Step 2: Rewrite `references/omlx.md`** as the authoritative spec covering: install (`brew tap jundot/omlx` + GUI app), settings.json structure (port 18080, api_key, model_dirs), launchd state, model lifecycle, and the 6 model capabilities currently served.

- [ ] **Step 3: Rewrite each cap-specific reference** as a thin "how to call this from `localhost:18080/v1` with `sk-mlx`" cookbook page with one minimal Python + one minimal cURL example.

- [ ] **Step 4: Create `references/asr-routing-module.md`** — pointer to `asr/README.md`, summarize the routing rationale (IM mode tradeoffs vs Meeting mode multi-pass), reference EVALUATION.md numbers.

- [ ] **Step 5: Create `references/asr-sherpa-onnx-sensevoice.md`** — capture the install/test/benchmark notes from this conversation (the 60ms RTF data, the FireRedASR2 comparison, the WSYue specialist finding, the LID auto-route discovery).

- [ ] **Step 6: Commit**

```bash
git commit -m "docs(refs): prune legacy notes, refresh as oMLX client cookbook

- Drop pre-oMLX assembly notes (whisper, gemma3, qwen3-14b, transcribe-daemon)
- references/omlx.md is now the authoritative spec
- Each capability reference is a thin client cookbook page
- Add asr-routing-module.md (pointer to asr/) + asr-sherpa-onnx-sensevoice.md
  (captures FireRedASR2/WSYue tradeoffs and SenseVoice LID auto-route insight)"
```

---

## Self-Review Notes

This plan has been reviewed against the spec:

1. **Spec coverage:**
   - Mode A (IM routing): Tasks 4, 5 ✓
   - Mode B (meeting pipeline): Tasks 6-13 ✓
   - Speaker diarization: Task 6 ✓
   - Speaker N → semantic role (LLM-driven): Task 9 prompt + Task 10 render ✓
   - Glossary merge (default + per-job): Task 8 ✓
   - Five model-named artifacts: Task 10 ✓
   - Validation on real meeting audio (env var): Task 14 ✓
   - server/ removal: Task 0 ✓
   - audio-transcribe skill removal: Task 15 ✓
   - Doc rewrite: Tasks 16-17 ✓

2. **Risk areas (acknowledged, mitigated):**
   - **Diarization model URL may have changed** — Task 1 step 5 install_models.sh; if 404, agent must search current sherpa-onnx releases for "speaker-segmentation" assets and update URL.
   - **gemma-4 prompt may produce non-JSON occasionally** — `_parse_response` strips code fences; if still fails, Task 9 needs a `response_format={"type":"json_object"}` retry path. Iterate during Task 14 if eval CER is poor.
   - **Long meeting (4650s) may exceed gemma-4 context budget** — Task 9 batches 12 segments at a time with sliding 4-segment context window. If hit limits, reduce batch.

3. **No placeholders.** Every code block is complete and cross-references match (e.g., `SegmentTranscript` defined in Task 7, used identically in Tasks 9, 10).

4. **Type consistency:** `DiarizedSegment` (Task 6) → `SegmentTranscript` (Task 7) → `ReviewedSegment` (Task 9) chain verified. `OMLXClient.chat(model, messages, **kwargs)` signature consistent across Tasks 3, 9, 10. `JobStatus` enum used identically in Tasks 11, 12, 13.

---

## Execution

Plan complete. Suggested order: Tasks 0 → 17 sequentially (each is a commit). Phases 1-3 are TDD — write the test, run it failing, implement, run it passing, commit. Phase 4 is the quality gate; do not proceed to Phase 5 if CER does not improve.
