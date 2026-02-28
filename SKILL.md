---
name: mlx-local-inference
description: >
  Full local AI inference stack on Apple Silicon Macs via MLX.
  **Default framework: mlx-vlm** for all capabilities including vision-language,
  text generation, speech-to-text ASR (Qwen3-ASR, Whisper), text embeddings,
  OCR (PaddleOCR-VL), and TTS (Qwen3-TTS).
  All models run locally via MLX with OpenAI-compatible APIs.
  Use when the user needs local AI capabilities: text generation, vision-language,
  speech recognition, embeddings/vector search, OCR, text-to-speech,
  or batch audio transcription — without cloud API calls.
metadata: { "openclaw": { "os": ["darwin"], "requires": { "anyBins": ["python3"] } } }
---

# MLX Local Inference Stack

Full local AI inference on Apple Silicon Macs using **mlx-vlm** as the unified framework. All services expose OpenAI-compatible APIs.

## Quick Start

```bash
# Vision-Language inference
python3 << 'PYEOF'
from mlx_vlm import load, generate
model, processor = load("mlx-community/Qwen3.5-35B-A3B-4bit")
response = generate(model, processor, "描述这张图片", "image.jpg")
print(response)
PYEOF
```

## Services Overview

| Service | Framework | Port | Models |
|---------|-----------|--------|--------|
| **Vision-Language + LLM** | mlx-vlm | 8787 | Qwen3.5-35B-A3B, Qwen3-14B |
| **ASR (Speech)** | mlx-audio | 8788 | Qwen3-ASR, Whisper |
| **Embeddings** | mlx-embeddings | 8787 | Qwen3-Embedding |
| **OCR** | mlx-vlm | 8787 | PaddleOCR-VL |
| **TTS** | mlx-audio | 8788 | Qwen3-TTS |

---

## 1. Vision-Language & LLM — Unified via mlx-vlm

### Recommended Model: Qwen3.5-35B-A3B-4bit

| Spec | Value |
|------|-------|
| Total params | 35B (A3B active) |
| Quantization | 4-bit MLX |
| Memory | ~20GB at load |
| Best for | Chinese, English, vision-language |

### Text-only Inference

```python
from mlx_vlm import load, generate

model, processor = load("mlx-community/Qwen3.5-35B-A3B-4bit")

# Chinese
response = generate(model, processor, 
                   prompt="你好，用一句话介绍你自己",
                   max_tokens=60)
print(response)  # 我是一个由阿里巴巴云开发的超大规模语言模型...

# English  
response = generate(model, processor,
                   prompt="What is 2+2? Answer briefly.",
                   max_tokens=30)
print(response)  # 4.
```

### Vision-Language Inference

```python
from mlx_vlm import load, generate

model, processor = load("mlx-community/Qwen3.5-35B-A3B-4bit")

# Describe image
response = generate(model, processor,
                   prompt="描述这张图片的内容",
                   image="photo.jpg",
                   max_tokens=200)

# OCR with visual context
response = generate(model, processor,
                   prompt="Extract all text from this image",
                   image="document.jpg",
                   max_tokens=500)
```

### API Server (OpenAI-compatible)

```bash
# Start server
cd ~/.mlx-server
source venv/bin/activate
python -m mlx_openai_server.launch --config config.yaml

# Chat completions
curl -X POST http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

## 2. ASR — Speech-to-Text

### Qwen3-ASR (Chinese/Cantonese optimized)

```bash
curl -X POST http://127.0.0.1:8788/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=mlx-community/Qwen3-ASR-1.7B-8bit" \
  -F "language=zh"
```

### Whisper (Multilingual)

```bash
curl -X POST http://localhost:8787/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=whisper-large-v3-turbo"
```

---

## 3. Embeddings — Text Vectorization

```bash
# Fast (0.6B)
curl -X POST http://localhost:8787/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-0.6b", "input": "text"}'

# High accuracy (4B)  
curl -X POST http://localhost:8787/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-4b", "input": ["text 1", "text 2"]}'
```

---

## 4. OCR — Image Text Extraction

```bash
cd ~/.mlx-server/venv
python -m mlx_vlm.generate \
  --model mlx-community/PaddleOCR-VL-1.5-6bit \
  --image document.jpg \
  --prompt "OCR:" \
  --max-tokens 512 --temp 0.0
```

---

## 5. TTS — Text-to-Speech

```bash
~/.mlx-server/venv/bin/mlx_audio.tts.generate \
  --model mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit \
  --text "你好，这是一段测试语音"
```

Or via API:
```bash
curl -X POST http://127.0.0.1:8788/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-TTS", "input": "你好世界"}' \
  --output speech.wav
```

---

## Service Management

```bash
# Start all services
launchctl load ~/Library/LaunchAgents/com.mlx-server.plist
launchctl load ~/Library/LaunchAgents/com.mlx-audio-server.plist

# Restart
launchctl kickstart -k gui/$(id -u)/com.mlx-server

# Logs
tail -f ~/.mlx-server/logs/server.log
tail -f ~/.mlx-server/logs/mlx-audio-server.err.log
```

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+ with mlx, mlx-vlm, mlx-audio, mlx-embeddings
- **Recommended: 32GB+ RAM** for 35B models
- For vision-language: mlx-vlm >= 0.3.12

## Migration from mlx-lm to mlx-vlm

Previous versions used `mlx-lm` as the default text-only framework. The current version has migrated to `mlx-vlm` as the **unified framework** for all capabilities including:
- Vision-Language (images + text)
- Text-only generation
- OCR via vision-language
- Unified API across all modalities

Models loaded via `mlx-vlm` (e.g., `Qwen3.5-35B-A3B-4bit`) support both text and vision inputs seamlessly.

## License

MIT License - see [LICENSE](LICENSE) file

## Contributing

欢迎提交 PR 和 Issue！

- GitHub: https://github.com/bendusy/mlx-local-inference
- 报告问题: https://github.com/bendusy/mlx-local-inference/issues