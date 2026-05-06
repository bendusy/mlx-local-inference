# asr-router

Standalone ASR routing service for Apple Silicon. Sits next to oMLX as a sidecar and exposes one OpenAI-compatible endpoint that does the right thing for both short voice messages and long meeting recordings.

```
┌──────────────────────────────────────────────────────────────────┐
│  asr-router                  http://<host>:18081/v1   key sk-mlx │
│                                                                  │
│  IM mode  /v1/audio/transcriptions                               │
│           sherpa-onnx SenseVoice (fast) ↔ oMLX Qwen3-ASR (qual.) │
│           auto-routes by duration / event tags / quality= param  │
│                                                                  │
│  Meeting  /v1/audio/jobs                                         │
│  mode     VAD+diarize → SenseVoice → gemma-4 review → 5 outputs  │
│           async; poll /v1/audio/jobs/{id}; CER -29% vs raw       │
└──────────────────────────────────────────────────────────────────┘
```

## For AI agents calling this service

→ **[`AGENTS.md`](./AGENTS.md)** — full integration guide (endpoints, JSON shapes, glossary format, polling pattern, error codes, code examples in curl / Python / TypeScript).

## Install (one-time)

```bash
# 1. Download required models (SenseVoice + silero-vad + pyannote-segmentation + 3D-Speaker)
bash scripts/install_models.sh

# 2. Sync Python deps
uv sync

# 3. Install as launchd agent (auto-starts on login, auto-restarts on crash)
bash scripts/install_launchd.sh
```

After that, the service is reachable at:
- `http://localhost:18081/v1` (same machine)
- `http://<your-mac>.local:18081/v1` (any LAN device)

API key: `sk-mlx`.

## Service management

```bash
launchctl list | grep com.user.asr-router    # status
tail -f logs/asr-router.err.log               # logs
launchctl kickstart -k gui/$(id -u)/com.user.asr-router  # restart
bash scripts/uninstall_launchd.sh             # remove daemon (keeps logs/jobs)
```

To run in the foreground for development:

```bash
bash scripts/run_dev.sh                       # uvicorn --reload on :18081
```

## Verify it's up

```bash
curl -s http://localhost:18081/v1/models -H "Authorization: Bearer sk-mlx"
```

Expected: `{"object":"list","data":[...]}`.

## Quick smoke test (IM mode)

```bash
curl -s http://localhost:18081/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-mlx" \
  -F "file=@some-voice.wav" -F "model=auto"
```

Expected: a JSON envelope with `text`, `language`, `duration`, plus `x_route` (which upstream answered) and `x_tags` (LID + event + emotion).

For meeting recordings, see `AGENTS.md §4`.

## Architecture summary

| Layer | Module | Role |
|---|---|---|
| HTTP | `asr_router/server.py` | FastAPI app; auth, IM endpoint, async job endpoints |
| IM router | `asr_router/im/router.py` | First-match-wins rule evaluator over `routing.yaml` |
| Models | `asr_router/models/sense_voice.py` | sherpa-onnx SenseVoice singleton (LID + event tags) |
| | `asr_router/models/omlx_client.py` | OpenAI-compatible client for `localhost:18080/v1` |
| Meeting pipeline | `asr_router/meeting/vad_diarize.py` | Pass 1: pyannote-segmentation + 3D-Speaker clustering |
| | `asr_router/meeting/transcribe.py` | Pass 2: SenseVoice per diarized segment |
| | `asr_router/meeting/review.py` | Pass 3: gemma-4 contextual review with glossary |
| | `asr_router/meeting/render.py` | Pass 4: write 5 model-named artifacts |
| | `asr_router/meeting/pipeline.py` | Orchestrator + background Worker |
| Persistence | `asr_router/jobs.py` | SQLite job state machine |
| Config | `routing.yaml` / `pipelines.yaml` / `glossary/default.yaml` | Declarative tunables |
| Prompts | `prompts/review.j2` / `prompts/summary.j2` | gemma-4 templates |

## Validation

See `EVALUATION.md`. SenseVoice raw CER 0.3208 → gemma-4 reviewed CER 0.2264 = **+29.4% relative improvement** on a 2-minute slice of a real bilingual meeting recording with per-job glossary applied.
