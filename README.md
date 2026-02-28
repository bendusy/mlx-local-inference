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

## Memory Profiles

| Profile | RAM Usage | What's Always Loaded |
|:--------|:----------|:---------------------|
| **16 GB** | ~3 GB idle | Embedding (0.6B) + ASR (1.7B) |
| **32 GB** | ~3 GB idle | Same — LLM/VLM loaded on demand |

**Key principle:** Nothing loads until you call it. Models are fetched from cache on first use, then unloaded when idle. The only always-on services are the lightweight API servers themselves.

## What Your Mac Gains

| Ability | Model | Memory | Load Strategy |
|:--------|:------|:-------|:--------------|
| 📐 **Embed** | Qwen3-Embedding-0.6B | ~1 GB | **Always loaded** |
| 👂 **Hear** | Qwen3-ASR-1.7B | ~1.5 GB | **Always loaded** |
| 🧠 **Think** | Qwen3.5-35B-A3B (32GB) / Qwen3-14B (16GB) | 20 GB / 9 GB | **On-demand** |
| 👁️ **See** | PaddleOCR-VL-1.5 | ~3.3 GB | **On-demand** |
| 🗣️ **Speak** | Qwen3-TTS-1.7B | ~2 GB | **On-demand (not default)** |

## Auto-Download

Missing models are detected automatically on first call. The server will:

1. Check if the model exists in `~/.cache/huggingface/hub/`
2. If missing, print a clear message and download automatically:

```
[mlx-server] Model not found: mlx-community/Qwen3-ASR-1.7B-8bit
[mlx-server] Downloading... (1.7 GB, ~2 min on fast connection)
[mlx-server] Download complete. Loading model...
```

You can also pre-download all default models at once:

```bash
python ~/.mlx-server/download_models.py
```

Or download a specific model:

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
          │   (demand) │  │  (demand) │  │            │
          └────────────┘  └───────────┘  └────────────┘
```

✅ = always loaded at startup | others = loaded on first call, unloaded when idle

## Usage

### 📐 Embed — Text Vectorization (always-on)

```bash
curl http://localhost:8787/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-0.6b", "input": "hello world"}'
```

### 👂 Hear — Speech Recognition (always-on)

```bash
# Chinese / Cantonese / mixed
curl http://localhost:8788/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=mlx-community/Qwen3-ASR-1.7B-8bit \
  -F language=zh
```

Supported formats: `wav`, `mp3`, `m4a`, `flac`, `ogg`, `webm`

### 🧠 Think — LLM / Vision-Language (on-demand via mlx-vlm)

```bash
# Text
curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-35b", "messages": [{"role": "user", "content": "Hello"}]}'

# Vision (image + text)
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

> **16 GB tip:** Use `qwen3-14b` instead of `qwen3.5-35b`. The 14B model uses ~9 GB and leaves enough headroom for embedding + ASR.

### 👁️ See — OCR (on-demand)

```bash
python -m mlx_vlm.generate \
  --model mlx-community/PaddleOCR-VL-1.5-6bit \
  --image document.jpg --prompt "OCR:" --max-tokens 512 --temp 0.0
```

### 🗣️ Speak — TTS (on-demand, not loaded by default)

```bash
curl http://localhost:8788/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-TTS", "input": "Hello world"}' \
  -o speech.wav
```

TTS loads on first call and unloads after a configurable idle timeout (default: 5 min).

### 📝 Transcribe — Auto Pipeline

Drop audio into `~/transcribe/` — the daemon handles the rest:

1. Qwen3-ASR transcribes → `filename_raw.md`
2. LLM corrects errors, adds punctuation → `filename_corrected.md`
3. Results archived to `~/transcribe/done/`

## Model Selection by RAM

### 16 GB Mac

| Role | Model | RAM |
|:-----|:------|:----|
| Embedding | `qwen3-embedding-0.6b` | ~1 GB |
| ASR | `Qwen3-ASR-1.7B-8bit` | ~1.5 GB |
| LLM (on-demand) | `Qwen3-14B-4bit` | ~9 GB |
| OCR (on-demand) | `PaddleOCR-VL-1.5-6bit` | ~3.3 GB |
| TTS (opt-in) | `Qwen3-TTS-1.7B-8bit` | ~2 GB |

> ⚠️ On 16 GB, avoid running LLM + OCR simultaneously. The daemon handles this automatically by unloading between phases.

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
# Restart main server (embedding + on-demand LLM/VLM)
launchctl kickstart -k gui/$(id -u)/com.mlx-server

# Restart ASR server (ASR always-on + TTS on-demand)
launchctl kickstart -k gui/$(id -u)/com.mlx-audio-server

# Restart transcription daemon
launchctl kickstart gui/$(id -u)/com.mlx-transcribe-daemon

# Logs
tail -f ~/.mlx-server/logs/server.log
tail -f ~/.mlx-server/logs/mlx-audio-server.err.log
```

## Upgrading Models

```bash
# 1. Download new model
huggingface-cli download mlx-community/<new-model>

# 2. Update config (~/.mlx-server/config.yaml)
# 3. Restart service
launchctl kickstart -k gui/$(id -u)/com.mlx-server
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
    ├── asr-qwen3.md
    ├── embedding-qwen3.md
    ├── llm-qwen3-14b.md
    ├── llm-qwen35-35b.md
    ├── ocr.md
    ├── transcribe-daemon.md
    └── tts-qwen3.md
```

## Contributing

Issues and PRs welcome.

## License

[MIT](LICENSE)
