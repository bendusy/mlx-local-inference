<p align="center">
  <h1 align="center">🧠 MLX Local Inference Stack</h1>
  <p align="center">
    Give your Apple Silicon Mac the power to hear, see, read, speak, think — all locally.
  </p>
  <p align="center">
    <a href="https://clawhub.ai/skills/mlx-local-inference"><img src="https://img.shields.io/badge/ClawHub-mlx--local--inference-FF5A36?style=flat-square" alt="ClawHub"></a>
    <a href="#"><img src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-000?style=flat-square&logo=apple&logoColor=white" alt="Platform"></a>
    <a href="#"><img src="https://img.shields.io/badge/runtime-MLX--VLM-blue?style=flat-square" alt="MLX-VLM"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"></a>
  </p>
  <p align="center">
    <a href="README_CN.md"><b>中文</b></a> · English
  </p>
</p>

---

## One-Line Install

```bash
clawhub install mlx-local-inference
```

Or clone directly:

```bash
git clone https://github.com/bendusy/mlx-local-inference.git
```

## Why This Exists

Your M-series Mac has a powerful Neural Engine and unified memory — yet most AI workflows still send every request to the cloud. **MLX Local Inference Stack** turns your Mac into a fully self-contained AI workstation, with a memory-efficient design that works on **16 GB machines**.

## Memory Strategy

| Profile | Idle RAM | Always Loaded |
|:--------|:---------|:--------------|
| **16 GB** | ~3 GB | Embedding (0.6B) + ASR (1.7B) |
| **32 GB** | ~3 GB | Same — LLM/VLM loaded on demand |

**Key principle:** Nothing loads until you call it. On first request, the model loads automatically (expect a few seconds delay). After an idle timeout, it unloads to free memory.

## What Your Mac Gains

| Ability | Model | Memory | Load Strategy |
|:--------|:------|:-------|:--------------|
| 📐 **Embed** | Qwen3-Embedding-0.6B | ~1 GB | **Always loaded** |
| 👂 **Hear** | Qwen3-ASR-1.7B | ~1.5 GB | **Always loaded** |
| 🧠 **Think** | Qwen3.5-35B-A3B (32GB) / Qwen3-14B (16GB) | 20 GB / 9 GB | **On-demand** |
| 👁️ **See** | PaddleOCR-VL-1.5 | ~3.3 GB | **On-demand** |
| 🗣️ **Speak** | Qwen3-TTS-1.7B | ~2 GB | **On-demand (opt-in)** |

## How On-Demand Loading Works

The server uses a **lazy proxy** pattern:

1. All models are registered at startup but **not loaded into memory**
2. On first request, the proxy loads the model transparently — the caller just waits
3. An **idle watchdog** monitors each model; after a configurable timeout with no requests, it unloads the model

```
Request arrives
      │
      ▼
 Model loaded? ──No──▶ Load now (caller waits) ──▶ Serve request
      │Yes                                               │
      └──────────────────────────────────────────────────┘
                                                         │
                                              [idle timeout]
                                                         │
                                                    Unload ✓
```

No client changes needed — it's fully transparent via the OpenAI-compatible API.

## Auto-Download

Missing models are detected automatically on first call:

```
[mlx-server] Model not found: mlx-community/Qwen3-ASR-1.7B-8bit
[mlx-server] Downloading... (1.7 GB, ~2 min on fast connection)
[mlx-server] Download complete. Loading model...
```

Pre-download all default models at once:

```bash
python ~/.mlx-server/download_models.py
```

Or a specific model:

```bash
huggingface-cli download mlx-community/Qwen3-ASR-1.7B-8bit
huggingface-cli download mlx-community/qwen3-embedding-0.6b-4bit
```

## Architecture

```
                        ┌─────────────────┐
                        │   Your Agent    │
                        │  (OpenClaw etc) │
                        └────────┬────────┘
                                 │ OpenAI-compatible API
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
          ┌────────────┐  ┌───────────┐  ┌────────────┐
          │  Port 8787 │  │ Port 8788 │  │    CLI     │
          │  always-on │  │ always-on │  │  on-demand │
          │            │  │           │  │            │
          │ · Embed ✅  │  │ · ASR ✅  │  │ · OCR      │
          │ · LLM/VLM  │  │ · TTS     │  │            │
          │   (lazy)   │  │  (lazy)   │  │            │
          └────────────┘  └───────────┘  └────────────┘
```

✅ = always loaded | lazy = loads on first request, unloads when idle

## Usage

### 📐 Embed — Text Vectorization (always-on)

```bash
curl http://localhost:8787/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-0.6b", "input": "hello world"}'
```

### 👂 Hear — Speech Recognition (always-on)

```bash
curl http://localhost:8788/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=mlx-community/Qwen3-ASR-1.7B-8bit \
  -F language=zh
```

Supported formats: `wav`, `mp3`, `m4a`, `flac`, `ogg`, `webm`

### 🧠 Think — LLM / Vision-Language (on-demand)

```bash
# Text
curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-35b", "messages": [{"role": "user", "content": "Hello"}]}'

# Vision
curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "text", "text": "What is in this image?"}
      ]
    }]
  }'
```

> **16 GB tip:** Use `qwen3-14b` (~9 GB) instead of `qwen3.5-35b` (~20 GB).

### 👁️ See — OCR (on-demand)

```bash
python -m mlx_vlm.generate \
  --model mlx-community/PaddleOCR-VL-1.5-6bit \
  --image document.jpg --prompt "OCR:" --max-tokens 512 --temp 0.0
```

### 🗣️ Speak — TTS (on-demand, opt-in)

```bash
curl http://localhost:8788/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-TTS", "input": "Hello world"}' \
  -o speech.wav
```

### 📝 Transcribe — Auto Pipeline

Drop audio into `~/transcribe/` — the daemon handles the rest:

1. Qwen3-ASR transcribes → `filename_raw.md`
2. LLM corrects errors, adds punctuation → `filename_corrected.md`
3. Archived to `~/transcribe/done/`

## Admin API

The server exposes a lightweight admin API for manual model management:

```bash
# List all models and load status
curl http://localhost:8787/v1/admin/models

# Manually unload a model
curl -X POST http://localhost:8787/v1/admin/models/qwen3.5-35b/unload

# Manually load a model
curl -X POST http://localhost:8787/v1/admin/models/qwen3.5-35b/load

# Queue stats
curl http://localhost:8787/v1/admin/models/qwen3.5-35b/stats
```

## Model Selection by RAM

### 16 GB Mac

| Role | Model | RAM |
|:-----|:------|:----|
| Embedding | `qwen3-embedding-0.6b` | ~1 GB |
| ASR | `Qwen3-ASR-1.7B-8bit` | ~1.5 GB |
| LLM (on-demand) | `Qwen3-14B-4bit` | ~9 GB |
| OCR (on-demand) | `PaddleOCR-VL-1.5-6bit` | ~3.3 GB |
| TTS (opt-in) | `Qwen3-TTS-1.7B-8bit` | ~2 GB |

> ⚠️ On 16 GB, avoid running LLM + OCR simultaneously.

### 32 GB Mac

| Role | Model | RAM |
|:-----|:------|:----|
| Embedding | `qwen3-embedding-0.6b` | ~1 GB |
| ASR | `Qwen3-ASR-1.7B-8bit` | ~1.5 GB |
| LLM/VLM (on-demand) | `Qwen3.5-35B-A3B-4bit` | ~20 GB |
| OCR (on-demand) | `PaddleOCR-VL-1.5-6bit` | ~3.3 GB |
| TTS (opt-in) | `Qwen3-TTS-1.7B-8bit` | ~2 GB |

## Service Management

```bash
# Restart main server
launchctl kickstart -k gui/$(id -u)/com.mlx-server

# Restart ASR/TTS server
launchctl kickstart -k gui/$(id -u)/com.mlx-audio-server

# Restart transcription daemon
launchctl kickstart gui/$(id -u)/com.mlx-transcribe-daemon

# Logs
tail -f ~/.mlx-server/logs/server.log
tail -f ~/.mlx-server/logs/mlx-audio-server.err.log
```

## Requirements

- Apple Silicon Mac (M1 / M2 / M3 / M4)
- macOS 14+
- Python 3.10+
- **16 GB RAM minimum** (32 GB recommended for 35B models)
- mlx-vlm >= 0.3.12

## Project Structure

```
mlx-local-inference/
├── SKILL.md              # OpenClaw skill definition
├── README.md             # English (this file)
├── README_CN.md          # 中文
├── LICENSE
└── references/           # Per-model technical docs
```

## License

[MIT](LICENSE)
