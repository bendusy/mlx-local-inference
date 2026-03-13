---
name: mlx-local-inference
description: >
  Use when calling local AI on this Mac — text generation, embeddings,
  speech-to-text, OCR, or image understanding. Unified omlx gateway at
  localhost:8000/v1. Models load on demand, stored permanently in ~/models.
  Works offline. Use instead of cloud APIs for privacy or low latency.
metadata: { "openclaw": { "os": ["darwin"], "requires": { "anyBins": ["omlx"] } } }
---

# MLX Local Inference — omlx Unified Gateway

All local AI served via **omlx** on a single endpoint. Models load on first
request, unload when idle. Permanent model storage at `~/models/`.

**Endpoint:** `http://localhost:8000/v1` (LAN: `http://10.11.12.34:8000/v1`)

## Models

| Capability | Model (directory name) | Size |
|-----------|------------------------|------|
| 📐 Embed | `Qwen3-Embedding-0.6B-4bit-DWQ` | ~1 GB |
| 🎤 ASR | `Qwen3-ASR-1.7B-8bit` | ~1.5 GB |
| 👁️ OCR/VLM | `PaddleOCR-VL-1.5-6bit` | ~3.3 GB |
| 💬 LLM (fast) | `Qwen3-14B-4bit` | ~8 GB |
| 💬 LLM (large) | `Qwen3.5-35B-A3B-4bit` | ~22 GB |

All models stored in `~/models/<directory-name>/`. Add more by downloading to that dir.

## Usage

### Embeddings

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")

resp = client.embeddings.create(
    model="Qwen3-Embedding-0.6B-4bit-DWQ",
    input="text to embed"
)
vector = resp.data[0].embedding  # 1024-dim
```

```bash
curl http://localhost:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-Embedding-0.6B-4bit-DWQ", "input": "hello"}'
```

---

### ASR — Speech-to-Text

```python
with open("audio.wav", "rb") as f:
    resp = client.audio.transcriptions.create(
        model="Qwen3-ASR-1.7B-8bit",
        file=f,
        language="zh"   # or "en", omit for auto-detect
    )
print(resp.text)
```

Supported formats: `wav`, `mp3`, `m4a`, `flac`, `ogg`, `webm`

---

### OCR

```python
import base64

with open("image.png", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = client.chat.completions.create(
    model="PaddleOCR-VL-1.5-6bit",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            {"type": "text", "text": "OCR:"}
        ]
    }],
    temperature=0.0
)
print(resp.choices[0].message.content)
```

**Note:** prompt must be exactly `"OCR:"`, temperature must be `0.0`.

---

### LLM / Vision-Language

```python
# Text
resp = client.chat.completions.create(
    model="Qwen3-14B-4bit",
    messages=[{"role": "user", "content": "Hello"}]
)

# Vision (image + text)
resp = client.chat.completions.create(
    model="Qwen3-14B-4bit",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
            {"type": "text", "text": "What is in this image?"}
        ]
    }]
)
```

---

## Service Management

```bash
# Check running models
curl http://localhost:8000/v1/models

# Restart
launchctl kickstart -k gui/$(id -u)/com.omlx-server

# Logs
tail -f /tmp/omlx-server.log

# Manual start (debug)
omlx serve --model-dir ~/models --port 8000
```

## Add More Models

```bash
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='mlx-community/Qwen3-14B-4bit',
    local_dir='$HOME/models/Qwen3-14B-4bit',
    local_dir_use_symlinks=False  # required — symlinks break omlx discovery
)
"
```

## Error Handling

| Error | Cause | Fix |
|-------|-------|-----|
| Connection refused | omlx not running | `launchctl kickstart -k gui/$(id -u)/com.omlx-server` |
| Model not found in /v1/models | Model not downloaded | Run `setup.sh` or download manually |
| 504 / OOM | Not enough RAM | Wait for other models to unload, or restart |
| 0 models returned | Wrong model dir structure | Ensure `~/models/ModelName/config.json` exists (no symlinks) |

## Setup

```bash
git clone https://github.com/bendusy/mlx-local-inference
cd mlx-local-inference
bash setup.sh
```

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 13+, Python 3.10+
- omlx (installed by setup.sh)
- 16 GB RAM minimum (32 GB for 35B models)
