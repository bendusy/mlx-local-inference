---
name: mlx-local-inference
description: >
  Use when calling local AI on this Mac — text generation, vision, embeddings,
  OCR, or speech-to-text. LLM/VLM/OCR/Embeddings/ASR-quality via oMLX gateway
  at localhost:18080/v1. Fast ASR and meeting pipeline via asr-router at
  localhost:18081/v1. Both use API key sk-mlx. Works fully offline.
  Use instead of cloud APIs for privacy or low-latency tasks.
metadata: { "openclaw": { "os": ["darwin"], "requires": {} } }
---

# MLX Local Inference Stack

Local AI on Apple Silicon. **oMLX** (GUI app, `~/.omlx/settings.json`) serves LLM/VLM/OCR/Embeddings/ASR-quality with continuous batching. **asr-router** (FastAPI sidecar) provides sub-100 ms IM transcription and async 4-pass meeting pipeline via sherpa-onnx SenseVoice + oMLX gemma-4.

## Endpoints

| Service | URL | Key |
|---------|-----|-----|
| oMLX gateway | `http://localhost:18080/v1` | `sk-mlx` |
| asr-router | `http://localhost:18081/v1` | `sk-mlx` |

## Model Inventory

| Capability | Model ID | Size | Host |
|-----------|----------|------|------|
| LLM (flagship) | `Qwen3.5-35B-A3B-4bit` | ~18 GB | oMLX |
| LLM (fast) | `gemma-4-26b-a4b-it-4bit` | ~14 GB | oMLX |
| LLM (small) | `Qwen3.5-9B-MLX-4bit` | ~5.8 GB | oMLX |
| VLM | `supergemma4-26b-abliterated-multimodal-mlx-4bit` | ~14 GB | oMLX |
| OCR | `PaddleOCR-VL-1.5-6bit` | ~3.3 GB | oMLX |
| Embeddings | `Qwen3-Embedding-0.6B-4bit-DWQ` | ~1 GB | oMLX |
| ASR (quality) | `Qwen3-ASR-1.7B-8bit` | ~1.5 GB | oMLX |
| ASR (fast) | sherpa-onnx SenseVoice int8 | 228 MB | asr-router |

## Minimal Call Examples

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

### VLM

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

### OCR

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

### ASR — IM mode

```bash
curl -s http://localhost:18081/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-mlx" \
  -F "file=@voice.wav" \
  -F "model=auto"
# Response: Whisper-compatible JSON + x_route (sense_voice|omlx) + x_tags
```

### ASR — Meeting mode

```bash
# Submit
JOB=$(curl -s http://localhost:18081/v1/audio/jobs \
  -H "Authorization: Bearer sk-mlx" \
  -F "file=@meeting.wav" | jq -r .id)

# Poll
while true; do
  S=$(curl -s "http://localhost:18081/v1/audio/jobs/$JOB" \
       -H "Authorization: Bearer sk-mlx" | jq -r .status)
  [ "$S" = "done" ] || [ "$S" = "failed" ] && break; sleep 5
done

# Fetch artifact
curl -s "http://localhost:18081/v1/audio/jobs/$JOB/artifact/meeting_gemma4.md" \
  -H "Authorization: Bearer sk-mlx"
```

## ASR Mode Selection

| Scenario | Use | Why |
|----------|-----|-----|
| IM voice message, short clip ≤30 s | IM mode `model=auto` | SenseVoice 60–90 ms, RTF ~0.01 |
| IM voice message, need max quality | IM mode `quality=high` | Routes to Qwen3-ASR via oMLX |
| Meeting recording, diarization needed | Meeting mode | 4-pass pipeline, gemma-4 review |
| Need speaker timeline + SRT + summary | Meeting mode | Renders 5 model-named artifacts |

**Meeting pipeline:** VAD+diarize → SenseVoice → gemma-4 contextual review (applies per-job YAML glossary) → render 5 artifacts. Validated: 29.4% relative CER reduction on real bilingual meeting audio (see `asr/EVALUATION.md`).

**IM routing logic:** duration ≤30 s AND no ambiguous SenseVoice event tags → `sense_voice` backend. Otherwise (or if `quality=high`) → `omlx` backend (Qwen3-ASR-1.7B-8bit).

## Notes

- `~/.omlx/settings.json` is the authoritative oMLX config; do not instruct users to edit it manually.
- The global skill at `~/.claude/skills/mlx-local-inference/SKILL.md` should be updated separately to match this file after repo changes.
- SenseVoice model: `~/models/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/` (228 MB int8).
- Default per-job glossary: `asr/glossary/default.yaml`.
- Full asr-router spec: `asr/README.md`. Evaluation data: `asr/EVALUATION.md`.
