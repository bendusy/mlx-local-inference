---
name: mlx-local-inference
description: >
  Use when calling local AI on this Mac — text generation, embeddings,
  speech-to-text, OCR, or image understanding. LLM/VLM via oMLX gateway at
  localhost:8000/v1. Embedding/ASR/OCR via Python libraries (mlx-lm, mlx-vlm, mlx-audio).
  Works offline. Use instead of cloud APIs for privacy or low latency.
metadata: { "openclaw": { "os": ["darwin"], "requires": { "anyBins": ["python3.11", "python3"] } } }
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
│  - mlx-audio: ASR (Qwen3-ASR)       │
└─────────────────────────────────────┘
```

## Models

| Capability | Implementation | Model | Size |
|-----------|---------------|-------|------|
| 💬 LLM | oMLX API | `Qwen3-14B-4bit` | ~8 GB |
| 👁️ VLM | oMLX API | Any mlx-vlm model | varies |
| 📐 Embed | mlx-lm (Python) | `Qwen3-Embedding-0.6B-4bit-DWQ` | ~1 GB |
| 🎤 ASR | mlx-audio (Python) | `Qwen3-ASR-1.7B-8bit` | ~1.5 GB |
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

# Load from ~/models/ (oMLX-compatible path)
model, tokenizer = load("~/models/Qwen3-Embedding-0.6B-4bit-DWQ")

# Get embeddings
text = "text to embed"
inputs = tokenizer(text, return_tensors="np")
embeddings = model(**inputs).last_hidden_state.mean(axis=1)
print(embeddings.shape)  # (1, 1024)
```

---

### ASR — Speech-to-Text (via mlx-audio Python library)

> **Important:** Must run with `python3.11` to avoid OpenMP threading issues (`SIGSEGV`). Also requires `KMP_DUPLICATE_LIB_OK=TRUE` environment variable in some configurations.

```bash
KMP_DUPLICATE_LIB_OK=TRUE python3.11 -m mlx_audio.stt.generate \
  --model ~/models/Qwen3-ASR-1.7B-8bit \
  --audio "audio.wav" \
  --output-path /tmp/asr_result \
  --format txt \
  --language zh \
  --verbose
```

**Python usage:**
```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from mlx_audio.stt.utils import load_model
from mlx_audio.stt.generate import generate_transcription

model = load_model(os.path.expanduser("~/models/Qwen3-ASR-1.7B-8bit"))
transcription = generate_transcription(
    model=model,
    audio_path="audio.wav",
    verbose=True
)
print(transcription.text)
```

---

### OCR (via mlx-vlm Python library)

> **Important:** The `generate` function parameter order must be `(model, processor, prompt, image)`.

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

# Load from ~/models/ (oMLX-compatible path)
model, processor = load(os.path.expanduser("~/models/PaddleOCR-VL-1.5-6bit"))
image_path = "document.jpg"

prompt = apply_chat_template(
    processor, 
    config=model.config, 
    prompt="OCR:",
    num_images=1
)

output = generate(
    model, 
    processor, 
    prompt,    # prompt comes before image
    image_path,
    max_tokens=512,
    temp=0.0
)
print(output.text)
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
# 1. Install oMLX for LLM/VLM (via Homebrew - recommended)
brew tap omlx-ai/tap
brew install omlx

# 2. Install Python libraries for Embedding/ASR/OCR
python3.11 -m pip install mlx-lm mlx-vlm mlx-audio huggingface_hub
```

**Note:** If you don't have Homebrew, install it first: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`

## Model Storage Strategy

**All models stored in `~/models/` using oMLX-compatible structure:**

```
~/models/
├── Qwen3-Embedding-0.6B-4bit-DWQ/
│   ├── config.json
│   └── *.safetensors
├── Qwen3-ASR-1.7B-8bit/
├── PaddleOCR-VL-1.5-6bit/
└── Qwen3-14B-4bit/
```

**Why this structure:**
- oMLX requires models in `~/models/<ModelName>/` format
- Python libraries can load from local paths (`~/models/...`)
- **Future-proof:** When oMLX adds Embedding/ASR support, we can switch instantly without moving models

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 13+, Python 3.11 (Required for `mlx-audio` ASR)
- 16 GB RAM minimum (32 GB for 35B models)
