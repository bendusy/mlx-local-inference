<p align="center">
  <h1 align="center">🧠 MLX Local Inference Stack</h1>
  <p align="center">
    Full local AI inference on Apple Silicon — LLM · ASR · Embedding · OCR · TTS · Transcription
  </p>
  <p align="center">
    <a href="https://clawhub.ai/skills/mlx-local-inference"><img src="https://img.shields.io/badge/ClawHub-mlx--local--inference-FF5A36?style=flat-square" alt="ClawHub"></a>
    <a href="#"><img src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-000?style=flat-square&logo=apple&logoColor=white" alt="Platform"></a>
    <a href="#"><img src="https://img.shields.io/badge/runtime-MLX-blue?style=flat-square" alt="MLX"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"></a>
  </p>
</p>

---

An [OpenClaw](https://github.com/openclaw/openclaw) skill that provides a complete local AI inference stack running entirely on Apple Silicon Macs via [MLX](https://github.com/ml-explore/mlx). Zero cloud dependencies, zero API costs, full privacy.

## What's Included

| Capability | Models | Port | Description |
|:-----------|:-------|:-----|:------------|
| **LLM** | Qwen3-14B, Gemma3-12B | 8787 | Chat completions with streaming, chain-of-thought reasoning |
| **ASR** | Qwen3-ASR, Whisper-v3-turbo | 8788 / 8787 | Speech-to-text, strong Cantonese/Mandarin + 99 languages |
| **Embedding** | Qwen3-Embedding 0.6B / 4B | 8787 | Text vectorization for RAG, semantic search, indexing |
| **OCR** | PaddleOCR-VL-1.5 | CLI | Scene text, receipts, documents (Chinese + English) |
| **TTS** | Qwen3-TTS-1.7B | 8788 / CLI | Text-to-speech with custom voice cloning |
| **Transcribe** | ASR + LLM pipeline | daemon | File-watch auto-transcription with smart correction |

All services expose **OpenAI-compatible APIs** — use them with the standard `openai` Python SDK, `curl`, or any compatible client.

## Requirements

- **Hardware**: Apple Silicon Mac (M1 / M2 / M3 / M4)
- **OS**: macOS 14+
- **RAM**: 32 GB+ recommended (for running multiple models simultaneously)
- **Python**: 3.10+

## Installation

### As an OpenClaw Skill

```bash
clawhub install mlx-local-inference
```

### Standalone Setup

```bash
# Clone
git clone https://github.com/bendusy/mlx-local-inference.git
cd mlx-local-inference

# Install Python dependencies
pip install mlx mlx-lm mlx-audio mlx-vlm openai
```

### Download Models

Models auto-download on first use. To pre-fetch:

```bash
# LLM
huggingface-cli download Qwen/Qwen3-14B-MLX-4bit
huggingface-cli download mlx-community/gemma-3-text-12b-it-4bit

# ASR
huggingface-cli download mlx-community/Qwen3-ASR-1.7B-8bit
huggingface-cli download mlx-community/whisper-large-v3-turbo

# Embedding
huggingface-cli download mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ

# OCR
huggingface-cli download mlx-community/PaddleOCR-VL-1.5-6bit

# TTS (optional)
huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
```

## Quick Start

### LLM Chat

```bash
curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b",
    "messages": [{"role": "user", "content": "Explain quantum computing in one sentence"}]
  }'
```

<details>
<summary>Python example</summary>

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8787/v1", api_key="unused")
response = client.chat.completions.create(
    model="qwen3-14b",
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7,
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

</details>

> **Note:** Qwen3 includes `<think>...</think>` chain-of-thought tags. Strip them with:
> ```python
> import re
> text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
> ```

### Speech-to-Text

```bash
# Qwen3-ASR — best for Cantonese / Mandarin
curl http://localhost:8788/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=mlx-community/Qwen3-ASR-1.7B-8bit \
  -F language=zh

# Whisper — multilingual (99 languages)
curl http://localhost:8787/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-large-v3-turbo
```

Supported formats: `wav`, `mp3`, `m4a`, `flac`, `ogg`, `webm`

### Embeddings

```bash
# Single text
curl http://localhost:8787/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-0.6b", "input": "text to embed"}'

# Batch
curl http://localhost:8787/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-4b", "input": ["text 1", "text 2"]}'
```

### OCR

```bash
python -m mlx_vlm.generate \
  --model mlx-community/PaddleOCR-VL-1.5-6bit \
  --image photo.jpg \
  --prompt "OCR:" \
  --max-tokens 512 \
  --temp 0.0
```

### Text-to-Speech

```bash
curl http://localhost:8788/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit","input":"Hello world"}' \
  -o speech.wav
```

### Auto-Transcription Daemon

Drop audio files into `~/transcribe/` and the daemon automatically:

1. **Transcribes** via Qwen3-ASR → `filename_raw.md`
2. **Corrects** via Qwen3-14B LLM → `filename_corrected.md`
3. **Archives** to `~/transcribe/done/`

Correction includes: homophone fixes, Cantonese character preservation (嘅/唔/咁/喺), punctuation, filler word removal.

## Architecture

```
┌──────────────────────────────────────────────┐
│           Apple Silicon Mac (MLX)            │
├──────────────────┬───────────────────────────┤
│    Port 8787     │       Port 8788           │
│    (LAN)         │       (localhost)          │
│                  │                           │
│  · Qwen3-14B    │  · Qwen3-ASR              │
│  · Gemma3-12B   │  · Qwen3-TTS              │
│  · Whisper      │                           │
│  · Embedding    │                           │
│    0.6B / 4B    │                           │
├──────────────────┴───────────────────────────┤
│  OCR: PaddleOCR-VL         (CLI, on-demand)  │
│  Transcribe Daemon    (file-watch pipeline)  │
└──────────────────────────────────────────────┘
```

## Model Selection Guide

### LLM

| Scenario | Recommended |
|:---------|:------------|
| Chinese / Cantonese | `qwen3-14b` |
| English / Code | `gemma-3-12b` |
| Deep reasoning | `qwen3-14b` (think mode) |
| Quick Q&A | `gemma-3-12b` |

### ASR

| Scenario | Recommended |
|:---------|:------------|
| Cantonese / Mandarin | Qwen3-ASR |
| Multilingual (99 langs) | Whisper |

### Embedding

| Scenario | Recommended |
|:---------|:------------|
| Fast retrieval | `qwen3-embedding-0.6b` |
| High-accuracy matching | `qwen3-embedding-4b` |

## Service Management

```bash
# Start / restart main service (LLM + Whisper + Embedding)
launchctl kickstart -k gui/$(id -u)/com.mlx-server

# Start / restart ASR + TTS service
launchctl kickstart -k gui/$(id -u)/com.mlx-audio-server

# Start transcription daemon
launchctl kickstart gui/$(id -u)/com.mlx-transcribe-daemon
```

## Project Structure

```
mlx-local-inference/
├── SKILL.md              # OpenClaw skill definition
├── README.md             # This file
├── LICENSE               # MIT
└── references/           # Detailed per-model docs
    ├── asr-qwen3.md
    ├── asr-whisper.md
    ├── embedding-qwen3.md
    ├── llm-qwen3-14b.md
    ├── llm-gemma3-12b.md
    ├── llm-models-reference.md
    ├── ocr.md
    ├── transcribe-daemon.md
    └── tts-qwen3.md
```

## Contributing

Issues and PRs welcome. For detailed model documentation, see the `references/` directory.

## License

[MIT](LICENSE)
