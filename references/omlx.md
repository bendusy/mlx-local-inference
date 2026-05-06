# oMLX — Authoritative Spec

oMLX is a macOS-native OpenAI-compatible inference server that manages MLX models on Apple Silicon. It handles model load/unload automatically — no manual lazy-loading scripts needed. Installed as `/Applications/oMLX.app` via `brew tap jundot/omlx && brew install omlx`.

## Configuration

`~/.omlx/settings.json` is the source of truth:
```json
{
  "port": 18080,
  "api_key": "sk-mlx",
  "max_model_memory": 32,
  "max_concurrent_requests": 8,
  "ssd_cache": true,
  "server_aliases": ["localhost", "127.0.0.1", "<your-mac>.local"]
}
```

Individual model settings (default, pinned) live in `~/.omlx/model_settings.json`.

## Authentication

All requests require `Authorization: Bearer sk-mlx` (the default key above).

## Endpoints

Base URL: `http://localhost:18080/v1`

| Endpoint | Function |
|----------|----------|
| `GET /v1/models` | List registered models |
| `POST /v1/chat/completions` | LLM / VLM / OCR |
| `POST /v1/embeddings` | Text embeddings |
| `POST /v1/audio/transcriptions` | ASR (Whisper-compatible) |

## Live Model Inventory

| Role | Model ID |
|------|----------|
| LLM default | `Qwen3.5-35B-A3B-4bit` |
| LLM fast | `Qwen3.5-9B-MLX-4bit` |
| LLM pinned | `gemma-4-26b-a4b-it-4bit` |
| VLM | `supergemma4-26b-abliterated-multimodal-mlx-4bit` |
| OCR | `PaddleOCR-VL-1.5-6bit` |
| Embeddings | `Qwen3-Embedding-0.6B-4bit-DWQ` |
| ASR | `Qwen3-ASR-1.7B-8bit` |

No TTS model is live.

## Model Lifecycle

oMLX loads models on first request and evicts least-recently-used models when memory pressure exceeds `max_model_memory`. `is_default: true` on `Qwen3.5-35B-A3B-4bit` means it pre-loads on startup. `is_pinned: true` on `gemma-4-26b-a4b-it-4bit` prevents eviction.

## Quick Health Check

```bash
curl http://localhost:18080/v1/models \
  -H "Authorization: Bearer sk-mlx" | python3 -m json.tool
```

## Server Aliases

`server_aliases` in settings.json allows other LAN devices to reach oMLX via `<your-mac>.local:18080` or the LAN IP. The ASR router (`:18081`) uses `http://localhost:18080` internally.
