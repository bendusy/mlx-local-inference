<p align="center">
  <h1 align="center">MLX Local Inference Stack</h1>
  <p align="center">
    oMLX gateway + asr-router sidecar — fully local AI on Apple Silicon.
  </p>
  <p align="center">
    <a href="https://clawhub.ai/skills/mlx-local-inference"><img src="https://img.shields.io/badge/ClawHub-mlx--local--inference-FF5A36?style=flat-square" alt="ClawHub"></a>
    <a href="#"><img src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-000?style=flat-square&logo=apple&logoColor=white" alt="Platform"></a>
    <a href="#"><img src="https://img.shields.io/badge/gateway-oMLX-blue?style=flat-square" alt="oMLX"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"></a>
  </p>
  <p align="center">
    <a href="README_CN.md"><b>中文</b></a> · English
  </p>
</p>

---

Two processes serve all local AI on this machine: **oMLX** (the GUI app) handles LLM/VLM/Embeddings/OCR/ASR-quality inference with continuous batching and SSD-backed model cache; **asr-router** is a FastAPI sidecar that wraps sherpa-onnx SenseVoice for sub-100 ms IM transcription and exposes an async 4-pass meeting pipeline (VAD+diarize → SenseVoice → gemma-4 contextual review → render 5 model-named artifacts). Both expose an OpenAI-compatible REST API on separate ports.

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                  Apple Silicon (Mac M-series)                      │
│                                                                    │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐│
│  │  oMLX gateway   :18080/v1    │  │  asr-router    :18081/v1     ││
│  │  key: sk-mlx                 │  │  key: sk-mlx                 ││
│  │                              │  │                              ││
│  │  • Qwen3.5-35B-A3B  (LLM)    │  │  IM mode      (Whisper-API): ││
│  │  • gemma-4-26b      (LLM)    │  │   sherpa-onnx SenseVoice     ││
│  │  • Qwen3.5-9B       (LLM)    │  │   ↔ oMLX Qwen3-ASR (auto)    ││
│  │  • supergemma4-26b  (VLM)    │  │                              ││
│  │  • PaddleOCR-VL-1.5 (OCR)    │  │  Meeting mode (async jobs):  ││
│  │  • Qwen3-Embedding-0.6B      │  │   VAD+diarize →              ││
│  │  • Qwen3-ASR-1.7B   (ASR-Q)  │  │   SenseVoice →               ││
│  │                              │  │   gemma-4 review (oMLX) →    ││
│  │  continuous batching, SSD    │  │   render 5 artifacts         ││
│  │  cache (managed by oMLX.app) │  │                              ││
│  └──────────────────────────────┘  └──────────────────────────────┘│
└────────────────────────────────────────────────────────────────────┘
```

## Endpoints

| Service | URL | API Key |
|---------|-----|---------|
| oMLX gateway | `http://localhost:18080/v1` | `sk-mlx` |
| asr-router | `http://localhost:18081/v1` | `sk-mlx` |

oMLX settings are managed by the GUI app; the authoritative config lives at `~/.omlx/settings.json`.

## Live Model Inventory

| Capability | Model ID | Size | Host |
|-----------|----------|------|------|
| LLM (flagship) | `Qwen3.5-35B-A3B-4bit` | ~18 GB | oMLX |
| LLM (fast, OpenClaw default) | `gemma-4-26b-a4b-it-4bit` | ~14 GB | oMLX |
| LLM (small) | `Qwen3.5-9B-MLX-4bit` | ~5.8 GB | oMLX |
| VLM | `supergemma4-26b-abliterated-multimodal-mlx-4bit` | ~14 GB | oMLX |
| OCR (VLM) | `PaddleOCR-VL-1.5-6bit` | ~3.3 GB | oMLX |
| Embeddings | `Qwen3-Embedding-0.6B-4bit-DWQ` | ~1 GB | oMLX |
| ASR (quality) | `Qwen3-ASR-1.7B-8bit` | ~1.5 GB | oMLX |
| ASR (fast, local) | sherpa-onnx SenseVoice int8 | 228 MB | asr-router |

SenseVoice model path: `~/models/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/`  
Languages: zh / en / yue / ja / ko. Decode latency: ~60–90 ms for 5–7 s clips (RTF ≈ 0.01).

## Quickstart

### LLM

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:18080/v1", api_key="sk-mlx")
resp = client.chat.completions.create(
    model="Qwen3.5-35B-A3B-4bit",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)
```

### VLM (image understanding)

```python
import base64
from openai import OpenAI
client = OpenAI(base_url="http://localhost:18080/v1", api_key="sk-mlx")
img_b64 = base64.b64encode(open("photo.jpg", "rb").read()).decode()
resp = client.chat.completions.create(
    model="supergemma4-26b-abliterated-multimodal-mlx-4bit",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
    ]}],
)
print(resp.choices[0].message.content)
```

### Embeddings

```bash
curl -s http://localhost:18080/v1/embeddings \
  -H "Authorization: Bearer sk-mlx" \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-Embedding-0.6B-4bit-DWQ", "input": "Hello"}' \
  | jq .data[0].embedding
```

### OCR (PaddleOCR-VL via chat completions)

```python
import base64
from openai import OpenAI
client = OpenAI(base_url="http://localhost:18080/v1", api_key="sk-mlx")
img_b64 = base64.b64encode(open("doc.jpg", "rb").read()).decode()
resp = client.chat.completions.create(
    model="PaddleOCR-VL-1.5-6bit",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "OCR this document."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
    ]}],
)
print(resp.choices[0].message.content)
```

### ASR — IM mode (asr-router auto-routes)

```bash
curl -s http://localhost:18081/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-mlx" \
  -F "file=@voice.wav" \
  -F "model=auto"
# returns Whisper-compatible JSON + x_route (sense_voice|omlx) + x_tags (lang/event/emotion)
```

Python equivalent:

```python
import httpx

with open("voice.wav", "rb") as f:
    r = httpx.post(
        "http://localhost:18081/v1/audio/transcriptions",
        headers={"Authorization": "Bearer sk-mlx"},
        files={"file": ("voice.wav", f, "audio/wav")},
        data={"model": "auto"},
    )
print(r.json())  # {"text": "...", "x_route": "sense_voice", "x_tags": {...}}
```

### ASR — Meeting mode (async job pipeline)

```bash
# 1. Submit
JOB=$(curl -s http://localhost:18081/v1/audio/jobs \
  -H "Authorization: Bearer sk-mlx" \
  -F "file=@meeting.wav" \
  -F 'glossary=terms:
  - term: Alpha Group
    aliases: [Alpa Group]' \
  | jq -r .id)

# 2. Poll
while true; do
  S=$(curl -s "http://localhost:18081/v1/audio/jobs/$JOB" \
       -H "Authorization: Bearer sk-mlx" | jq -r .status)
  echo "$S"; [ "$S" = "done" ] || [ "$S" = "failed" ] && break
  sleep 5
done

# 3. Fetch artifacts
curl -s "http://localhost:18081/v1/audio/jobs/$JOB/artifact/meeting_gemma4.md" \
  -H "Authorization: Bearer sk-mlx"
```

## ASR Routing Module

The `asr/` directory contains a standalone FastAPI service that runs alongside oMLX. It exposes two endpoints on port 18081: a synchronous Whisper-compatible transcription endpoint (`POST /v1/audio/transcriptions`) and an asynchronous meeting pipeline (`POST /v1/audio/jobs`).

**IM mode** auto-routes between two backends. Short clips (≤ 30 s) with no ambiguous event tags go straight to sherpa-onnx SenseVoice (60–90 ms decode, RTF ~0.01). Longer audio, audio flagged with uncertain emotion/event tags, or requests with `quality=high` are forwarded to oMLX's Qwen3-ASR-1.7B-8bit for higher accuracy. The response is Whisper-compatible JSON extended with `x_route` (which backend was used) and `x_tags` (language, event, emotion from SenseVoice).

**Meeting mode** runs a 4-pass pipeline asynchronously. Pass 1: VAD + speaker diarization segments the audio. Pass 2: SenseVoice transcribes each segment with language and speaker tags. Pass 3: gemma-4-26b on oMLX performs contextual review, applying the per-job glossary to correct proper nouns, domain terminology, and cross-lingual homophones. Pass 4: the reviewed transcript is rendered into 5 model-named artifacts (raw SenseVoice output, gemma-4 reviewed Markdown, speaker timeline JSON, segment SRT, summary). Artifacts are retrieved via `GET /v1/audio/jobs/{id}/artifact/{name}`.

Per-job glossary is submitted as YAML in the `glossary` multipart field. The default glossary seeded from project contact data lives at `asr/glossary/default.yaml`. See [`asr/README.md`](asr/README.md) for setup and operations, and [`asr/AGENTS.md`](asr/AGENTS.md) for the **AI agent integration spec** (endpoints, JSON shapes, polling pattern, code examples in curl / Python / TypeScript).

asr-router is installed as a launchd agent (`com.user.asr-router`) so it auto-starts on login and auto-restarts on crash. After installing the asr/ deps once, run `bash asr/scripts/install_launchd.sh`. The service then listens on `0.0.0.0:18081` and is reachable from any LAN device via `http://<your-mac>.local:18081/v1` with key `sk-mlx`.

**Validated performance:** gemma-4 contextual review reduces Character Error Rate from 32.08% to 22.64% — a **29.4% relative CER improvement** — on a 2-minute slice of real bilingual meeting audio with per-job glossary applied. See [`asr/EVALUATION.md`](asr/EVALUATION.md) for methodology and full results.

## Hardware Requirements

- Apple Silicon Mac (M1 / M2 / M3 / M4 series)
- **16 GB unified memory** minimum for the full stack with 26 B models loaded; 32 GB recommended for concurrent LLM + VLM use
- The 35 B MoE flagship (`Qwen3.5-35B-A3B-4bit`, ~18 GB weights) requires at least 24 GB free unified memory; oMLX manages on-demand loading and SSD-backed KV cache so models are evicted when idle
- SenseVoice (228 MB int8) and Qwen3-Embedding (1 GB) stay resident continuously; total resident footprint with both small models is ~1.3 GB

## Status

| Component | State |
|-----------|-------|
| oMLX gateway | Running — 7 models loaded (6 in oMLX + SenseVoice in asr-router) |
| asr-router IM mode | Implemented — SenseVoice ↔ Qwen3-ASR auto-routing |
| asr-router meeting pipeline | Implemented — 4-pass: VAD/diarize → SenseVoice → gemma-4 review → render |
| gemma-4 contextual review | Validated — 29.4% CER reduction on real bilingual meeting audio |
| Per-job glossary | Working — YAML submitted at job submission time |

## Acknowledgements

- [oMLX](https://github.com/jundot/oMLX) — local MLX inference GUI and OpenAI-compatible gateway (`brew tap jundot/omlx`)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — fast ONNX runtime for SenseVoice
- [SenseVoice / FunAudioLLM](https://github.com/FunAudioLLM/SenseVoice) — multilingual ASR with emotion and event detection
- [mlx-community on Hugging Face](https://huggingface.co/mlx-community) — quantized MLX model weights

## License

[MIT](LICENSE)
