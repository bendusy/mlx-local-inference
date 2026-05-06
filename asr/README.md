# asr-router

Standalone ASR routing service. Two modes:

1. **IM mode** — Whisper-compatible `POST /v1/audio/transcriptions` that auto-routes between local SenseVoice (fast) and oMLX Qwen3-ASR (quality) based on duration / event tags / explicit `quality=high|fast` param.

2. **Meeting mode** — Async `POST /v1/audio/jobs` runs a 4-pass pipeline (VAD + speaker diarization → SenseVoice transcription → gemma-4 contextual review with glossary → render 5 model-named artifacts).

## Quickstart (after Tasks 2-13 implemented)

```bash
bash scripts/install_models.sh           # download SenseVoice + VAD + diarization
uv sync --extra dev
bash scripts/run_dev.sh                   # serves on :18081
```

See `docs/superpowers/plans/2026-05-06-asr-routing-module.md` (in repo root) for the full implementation plan.
