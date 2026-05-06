# LLM Model Routing Chart

Three LLMs are currently live on oMLX (`:18080`). Auth: see `references/omlx.md`.

## Models

### `Qwen3.5-35B-A3B-4bit` — Default

| | |
|--|--|
| `is_default` | true (pre-loads on startup) |
| Use when | Hardest reasoning, multi-step analysis, bilingual (zh/en) tasks |
| Tradeoff | Slow TTFT (~3-5s); highest quality |

### `gemma-4-26b-a4b-it-4bit` — Pinned

| | |
|--|--|
| `is_pinned` | true (never evicted) |
| Use when | Tool use, function calling, fast structured output; OpenClaw default |
| Tradeoff | Lower reasoning ceiling than 35B; stays in VRAM — always-ready |

### `Qwen3.5-9B-MLX-4bit` — Small

| | |
|--|--|
| Use when | Dev iteration, low-memory sessions, quick drafts |
| Tradeoff | Weakest reasoning; smallest footprint (~5 GB) |

## VLM / OCR

- **VLM:** `supergemma4-26b-abliterated-multimodal-mlx-4bit` — image + text tasks
- **OCR:** `PaddleOCR-VL-1.5-6bit` — see `references/ocr.md`

## Quick Reference

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:18080/v1", api_key="sk-mlx")

# Use default (hardest tasks)
resp = client.chat.completions.create(
    model="Qwen3.5-35B-A3B-4bit",
    messages=[{"role": "user", "content": "Explain quantum entanglement."}],
)

# Use fast/pinned (tool use, structured output)
resp = client.chat.completions.create(
    model="gemma-4-26b-a4b-it-4bit",
    messages=[{"role": "user", "content": "Parse this JSON: ..."}],
)
```

## Notes

- The asr-router (`:18081`) uses `gemma-4-26b-a4b-it-4bit` internally for meeting-mode contextual review.
- Check `/v1/models` for current load state and any additional models added to `~/.omlx/model_settings.json`.
