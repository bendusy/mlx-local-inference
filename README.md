<p align="center">
  <h1 align="center">🧠 MLX Local Inference Stack</h1>
  <p align="center">
    Give your Apple Silicon Mac the power to hear, see, read, speak, think — all locally.
  </p>
  <p align="center">
    <a href="https://clawhub.ai/skills/mlx-local-inference"><img src="https://img.shields.io/badge/ClawHub-mlx--local--inference-FF5A36?style=flat-square" alt="ClawHub"></a>
    <a href="#"><img src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-000?style=flat-square&logo=apple&logoColor=white" alt="Platform"></a>
    <a href="#"><img src="https://img.shields.io/badge/runtime-MLX-blue?style=flat-square" alt="MLX"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"></a>
  </p>
  <p align="center">
    <a href="README_CN.md"><b>中文</b></a> · English
  </p>
</p>

---

## Why This Exists

Your M-series Mac has a powerful Neural Engine and unified memory sitting right there — yet most AI workflows still send every request to the cloud. That's wasteful, slow, and unnecessary for a huge number of tasks.

**MLX Local Inference Stack** turns your Mac into a fully self-contained AI workstation. We've tested and curated the best-performing MLX models across every modality — speech recognition, text generation, OCR, text-to-speech, and embeddings — so you don't have to. One install, and your Mac can **hear, see, read, speak, and think**, entirely offline.

This is especially useful when paired with AI agents like [OpenClaw](https://github.com/openclaw/openclaw). Instead of routing every tool call through cloud APIs, the agent can leverage your local hardware for transcription, text correction, document reading, voice output, and semantic search — making interactions faster, cheaper, and more private.

## What Your Mac Gains

| Ability | What It Does | Curated Model |
|:--------|:-------------|:--------------|
| 👂 **Hear** | Transcribe speech in 99 languages, with native Cantonese/Mandarin accuracy | Qwen3-ASR-1.7B · Whisper-v3-turbo |
| 👁️ **See** | Extract text from photos, screenshots, receipts, documents | PaddleOCR-VL-1.5 |
| 🧠 **Think** | Chat, reason, write code, translate, summarize | Qwen3-14B · Gemma3-12B |
| 🗣️ **Speak** | Generate natural speech with custom voice cloning | Qwen3-TTS-1.7B |
| 📐 **Understand** | Vectorize text for semantic search, RAG, and document indexing | Qwen3-Embedding 0.6B · 4B |
| 📝 **Transcribe** | Drop an audio file, get corrected transcripts automatically | ASR + LLM correction pipeline |

Every model was selected through hands-on testing for quality, speed, and memory efficiency on Apple Silicon. They're packaged together as one coherent stack — not a collection of random tools, but an integrated local AI runtime.

## How It Fits Together

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
          │            │  │           │  │            │
          │ · LLM      │  │ · ASR     │  │ · OCR      │
          │ · Whisper  │  │ · TTS     │  │            │
          │ · Embed    │  │           │  │            │
          └────────────┘  └───────────┘  └────────────┘
                 │               │               │
                 └───────────────┴───────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   Apple Silicon (MLX)   │
                    │   Unified Memory GPU    │
                    └─────────────────────────┘
```

All services expose **OpenAI-compatible APIs**. Any tool, SDK, or agent that speaks the OpenAI protocol works out of the box — no adapters, no wrappers.

## Requirements

- Apple Silicon Mac (M1 / M2 / M3 / M4)
- macOS 14+
- Python 3.10+
- 32 GB+ RAM recommended (16 GB works with fewer concurrent models)

## Get Started

### Install as OpenClaw Skill

```bash
clawhub install mlx-local-inference
```

### Or Clone Directly

```bash
git clone https://github.com/bendusy/mlx-local-inference.git
cd mlx-local-inference
pip install mlx mlx-lm mlx-audio mlx-vlm openai
```

Models download automatically on first use. To pre-fetch everything:

<details>
<summary>Pre-download all models</summary>

```bash
huggingface-cli download Qwen/Qwen3-14B-MLX-4bit
huggingface-cli download mlx-community/gemma-3-text-12b-it-4bit
huggingface-cli download mlx-community/Qwen3-ASR-1.7B-8bit
huggingface-cli download mlx-community/whisper-large-v3-turbo
huggingface-cli download mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
huggingface-cli download mlx-community/PaddleOCR-VL-1.5-6bit
huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
```

</details>

## Usage

### Think — LLM Chat

```bash
curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b",
    "messages": [{"role": "user", "content": "Explain quantum computing briefly"}]
  }'
```

<details>
<summary>Python</summary>

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8787/v1", api_key="unused")
r = client.chat.completions.create(
    model="qwen3-14b",
    messages=[{"role": "user", "content": "Hello"}],
)
print(r.choices[0].message.content)
```

</details>

Two LLMs are included: **Qwen3-14B** (strongest Chinese + reasoning with built-in chain-of-thought) and **Gemma3-12B** (fast English + code). Pick based on your task.

### Hear — Speech Recognition

```bash
# Cantonese / Mandarin → Qwen3-ASR
curl http://localhost:8788/v1/audio/transcriptions \
  -F file=@audio.wav -F model=mlx-community/Qwen3-ASR-1.7B-8bit -F language=zh

# Any of 99 languages → Whisper
curl http://localhost:8787/v1/audio/transcriptions \
  -F file=@audio.wav -F model=whisper-large-v3-turbo
```

Supports: `wav`, `mp3`, `m4a`, `flac`, `ogg`, `webm`

### See — OCR

```bash
python -m mlx_vlm.generate \
  --model mlx-community/PaddleOCR-VL-1.5-6bit \
  --image document.jpg --prompt "OCR:" --max-tokens 512 --temp 0.0
```

### Speak — Text-to-Speech

```bash
curl http://localhost:8788/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit","input":"Hello world"}' \
  -o speech.wav
```

### Understand — Embeddings

```bash
curl http://localhost:8787/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-0.6b", "input": ["document 1", "document 2"]}'
```

Two sizes: **0.6B** for fast retrieval, **4B** for high-accuracy semantic matching.

### Transcribe — Auto Pipeline

Drop audio files into `~/transcribe/` and walk away:

1. Qwen3-ASR transcribes → `filename_raw.md`
2. Qwen3-14B corrects errors, adds punctuation, preserves Cantonese → `filename_corrected.md`
3. Results archived to `~/transcribe/done/`

No commands needed. Just drop and go.

## Model Selection

Every model was chosen for the best balance of quality and efficiency on Apple Silicon:

| Modality | Model | Why This One |
|:---------|:------|:-------------|
| LLM (Chinese) | Qwen3-14B 4bit | Best bilingual performance at this size; native chain-of-thought |
| LLM (English) | Gemma3-12B 4bit | Fast, strong code generation, lean memory footprint |
| ASR (Chinese) | Qwen3-ASR-1.7B 8bit | Superior Cantonese/Mandarin accuracy, on-demand loading |
| ASR (Multi) | Whisper-v3-turbo | 99 languages, always loaded, battle-tested |
| Embedding (Fast) | Qwen3-Embedding-0.6B 4bit | Low latency, good enough for most retrieval |
| Embedding (Accurate) | Qwen3-Embedding-4B 4bit | High-precision semantic matching |
| OCR | PaddleOCR-VL-1.5 6bit | ~185 tokens/s, 3.3 GB, best accuracy-to-speed ratio |
| TTS | Qwen3-TTS-1.7B 8bit | Custom voice cloning, ~2 GB |

## Service Management

```bash
# Main service (LLM + Whisper + Embedding)
launchctl kickstart -k gui/$(id -u)/com.mlx-server

# ASR + TTS service
launchctl kickstart -k gui/$(id -u)/com.mlx-audio-server

# Auto-transcription daemon
launchctl kickstart gui/$(id -u)/com.mlx-transcribe-daemon
```

## Project Structure

```
mlx-local-inference/
├── SKILL.md              # OpenClaw skill definition
├── README.md             # English (this file)
├── README_CN.md          # 中文
├── LICENSE
└── references/           # Detailed per-model documentation
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

Issues and PRs welcome. See `references/` for detailed technical documentation on each model.

## License

[MIT](LICENSE)
