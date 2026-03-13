---
name: mlx-local-inference
description: >
  Use when calling local AI on this Mac — text generation, embeddings,
  speech-to-text, OCR, or image understanding. LLM/VLM via oMLX gateway at
  localhost:8000/v1. Embedding/ASR/OCR via Python libraries (mlx-lm, mlx-vlm).
  Works offline. Use instead of cloud APIs for privacy or low latency.
metadata: { "openclaw": { "os": ["darwin"], "requires": { "anyBins": ["python3"] } } }
---

# MLX Local Inference Stack

Local AI inference on Apple Silicon. **oMLX** handles LLM/VLM with continuous batching.
Python libraries handle Embedding/ASR/OCR directly.

## Architecture

```
┌─────────────────────────────────────┐
│  oMLX (localhost:8000/v1)           │
│  - LLM (Qwen3-14B, etc.)            │
│  - VLM (vision-language models)     │
│  - Continuous batching + SSD cache  │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  Python Libraries (direct call)     │
│  - mlx-lm: Embedding                │
│  - mlx-vlm: OCR (PaddleOCR-VL)      │
│  - mlx-whisper: ASR (Qwen3-ASR)     │
└─────────────────────────────────────┘
```

## Models

| Capability | Implementation | Model | Size |
|-----------|---------------|-------|------|
| 💬 LLM | oMLX API | `Qwen3-14B-4bit` | ~8 GB |
| 👁️ VLM | oMLX API | Any mlx-vlm model | varies |
| 📐 Embed | mlx-lm (Python) | `Qwen3-Embedding-0.6B-4bit-DWQ` | ~1 GB |
| 🎤 ASR | mlx-whisper (Python) | `Qwen3-ASR-1.7B-8bit` | ~1.5 GB |
| 👁️ OCR | mlx-vlm (Python) | `PaddleOCR-VL-1.5-6bit` | ~3.3 GB |

## Usage

### LLM / Vision-Language (via oMLX API)

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")

# Text generation
resp = client.chat.completions.create(
    model="Qwen3-14B-4bit",
    messages=[{"role": "user", "content": "Hello"}]
)
print(resp.choices[0].message.content)

# Vision (image + text)
import base64
with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

resp = client.chat.completions.create(
    model="Qwen3-14B-4bit",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": "What is in this image?"}
        ]
    }]
)
```

---

### Embeddings (via mlx-lm Python library)

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")

# Get embeddings
text = "text to embed"
inputs = tokenizer(text, return_tensors="np")
embeddings = model(**inputs).last_hidden_state.mean(axis=1)
print(embeddings.shape)  # (1, 1024)
```

---

### ASR — Speech-to-Text (via mlx-whisper Python library)

```python
import mlx_whisper

# Transcribe audio
result = mlx_whisper.transcribe(
    "audio.wav",
    path_or_hf_repo="mlx-community/Qwen3-ASR-1.7B-8bit",
    language="zh"  # or "en", omit for auto-detect
)
print(result["text"])
```

Supported formats: `wav`, `mp3`, `m4a`, `flac`, `ogg`, `webm`

---

### OCR (via mlx-vlm Python library)

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_image

model, processor = load("mlx-community/PaddleOCR-VL-1.5-6bit")
image = load_image("document.jpg")

prompt = apply_chat_template(
    processor, 
    config=model.config, 
    prompt="OCR:",
    num_images=1
)

output = generate(
    model, 
    processor, 
    image, 
    prompt, 
    max_tokens=512,
    temp=0.0
)
print(output)
```

> **Note:** For OCR, prompt must be `"OCR:"` and temperature `0.0`.

---

## Service Management (oMLX only)

```bash
# Check running models
curl http://localhost:8000/v1/models

# Restart oMLX
launchctl kickstart -k gui/$(id -u)/com.omlx-server

# Logs
tail -f /tmp/omlx-server.log

# Manual start (debug)
omlx serve --model-dir ~/models --port 8000
```

## Installation

```bash
# Install oMLX for LLM/VLM
brew install omlx
# or: pip install omlx

# Install Python libraries for Embedding/ASR/OCR
pip install mlx-lm mlx-vlm mlx-whisper
```

## Setup

```bash
git clone https://github.com/bendusy/mlx-local-inference
cd mlx-local-inference

# Install dependencies
pip install mlx-lm mlx-vlm mlx-whisper

# Install and start oMLX (for LLM/VLM)
brew install omlx
omlx serve --model-dir ~/models --port 8000
```

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 13+, Python 3.10+
- 16 GB RAM minimum (32 GB for 35B models)
