---
name: mlx-local-inference
description: >
  Full local AI inference stack on Apple Silicon Macs via MLX.
  Default framework: mlx-vlm (unified vision-language + text).
  Memory-efficient design: only Embedding (0.6B) and ASR (1.7B) are always loaded (~3 GB idle).
  LLM/VLM, OCR loaded on-demand. TTS not loaded by default.
  Missing models are auto-downloaded on first call.
  Works on 16 GB Macs. 32 GB recommended for 35B models.
  Use when the user needs local AI: text generation, vision-language, speech recognition,
  embeddings, OCR, TTS — without cloud API calls.
metadata: { "openclaw": { "os": ["darwin"], "requires": { "anyBins": ["python3"] } } }
---

# MLX Local Inference Stack

Full local AI inference on Apple Silicon Macs. **mlx-vlm** is the unified framework for all vision-language and text generation. Designed to run on **16 GB machines** — only essential models stay resident.

## Memory Strategy

| Always Loaded | On-Demand | Not Loaded by Default |
|:--------------|:----------|:----------------------|
| Embedding 0.6B (~1 GB) | LLM/VLM (9–20 GB) | TTS (~2 GB) |
| ASR 1.7B (~1.5 GB) | OCR (~3.3 GB) | |
| **Total idle: ~3 GB** | | |

**Key principle:** Nothing loads until you call it. Models are fetched from cache on first use, then unloaded when idle. The only always-on services are the lightweight API servers themselves.

## Auto-Download

Missing models are detected automatically on first call:

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

### 1. Embedding (always-on, ~1 GB)

```bash
curl http://localhost:8787/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-0.6b", "input": "text to embed"}'
```

---

### 2. ASR — Speech-to-Text (always-on, ~1.5 GB)

```bash
# Chinese / Cantonese / mixed
curl http://localhost:8788/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=mlx-community/Qwen3-ASR-1.7B-8bit \
  -F language=zh
```

Supported: `wav`, `mp3`, `m4a`, `flac`, `ogg`, `webm`

---

### 3. LLM / Vision-Language (on-demand via mlx-vlm)

#### Model by RAM

| RAM | Model | Memory |
|-----|-------|--------|
| 16 GB | `qwen3-14b` | ~9 GB |
| 32 GB | `qwen3.5-35b` | ~20 GB |

#### Text

```bash
curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-35b", "messages": [{"role": "user", "content": "Hello"}]}'
```

#### Vision (image + text)

```bash
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

#### Python (mlx-vlm direct)

```python
from mlx_vlm import load, generate

model, processor = load("mlx-community/Qwen3.5-35B-A3B-4bit")
response = generate(model, processor, prompt="你好", max_tokens=200)
print(response)
```

> **16 GB tip:** Use `qwen3-14b` instead of `qwen3.5-35b`. The 14B model uses ~9 GB and leaves enough headroom for embedding + ASR.

#### Qwen3 Think Mode

Strip chain-of-thought tags if needed:
```python
import re
text = re.sub(r'<think>.*?必修课\s*', '', text, flags=re.DOTALL)
```

---

### 4. OCR (on-demand, ~3.3 GB)

```bash
python -m mlx_vlm.generate \
  --model mlx-community/PaddleOCR-VL-1.5-6bit \
  --image document.jpg \
  --prompt "OCR:" \
  --max-tokens 512 --temp 0.0
```

---

### 5. TTS — Text-to-Speech (on-demand, not loaded by default)

```bash
curl http://localhost:8788/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-TTS", "input": "Hello world"}' \
  -o speech.wav
```

Loads on first call, unloads after 5 min idle.

---

### 6. Transcribe Daemon — Auto Pipeline

Drop audio into `~/transcribe/` for automatic processing:

1. ASR transcribes → `filename_raw.md`
2. LLM corrects → `filename_corrected.md`
3. Archived to `~/transcribe/done/`

> On 16 GB: daemon unloads ASR before loading LLM to avoid memory contention.

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

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+, mlx-vlm >= 0.3.12
- **16 GB RAM minimum** (32 GB for 35B models)
