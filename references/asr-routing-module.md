# ASR Routing Module

The asr-router is a standalone FastAPI service on `:18081` that wraps SenseVoice (local, fast) and oMLX Qwen3-ASR (quality) with auto-routing logic. Two modes: **IM** (real-time, Whisper-compatible) and **Meeting** (async, multi-pass pipeline).

Full API reference: `asr/README.md`. Evaluation numbers: `asr/EVALUATION.md`.

## IM Mode

Endpoint: `POST /v1/audio/transcriptions` (Whisper-compatible drop-in).

**Routing rules** (first match wins):

| Condition | Backend |
|-----------|---------|
| `quality=high` param | oMLX Qwen3-ASR |
| `quality=fast` param | SenseVoice |
| duration > 30s | oMLX Qwen3-ASR |
| event tag in `{BGM, Applause, Laughter}` | oMLX Qwen3-ASR |
| default | SenseVoice |

**Speed:** ~60-90ms decode for 5-7s audio (SenseVoice path). oMLX path adds ~2-4s TTFT.

**Quality tradeoff:** SenseVoice occasionally misreads uncommon homophones (e.g. "gold" → "code" on edge cases). For high-stakes short clips, pass `quality=high`.

## Meeting Mode

Endpoint: `POST /v1/audio/jobs` (async; poll `GET /v1/audio/jobs/{job_id}`).

4-pass pipeline:
1. **VAD + Diarization** — sherpa-onnx pyannote-segmentation-3-0 + 3D-Speaker eres2net segments audio by speaker
2. **SenseVoice transcription** — each segment transcribed with lang/event metadata
3. **gemma-4 contextual review** — `gemma-4-26b-a4b-it-4bit` re-reads all segments with per-job glossary + inferred speaker roles
4. **Render artifacts** — 5 named files written to `~/.asr-router/jobs/{job_id}/`

**Output artifacts:**

| File | Contents |
|------|----------|
| `_raw.json` | Raw diarization + SenseVoice JSON |
| `_sensevoice.md` | Markdown transcript from SenseVoice |
| `_gemma4.md` | Reviewed transcript from gemma-4 |
| `_diff.md` | Diff between SenseVoice and gemma-4 passes |
| `_summary.md` | Meeting summary with action items |

## Validation (from `asr/EVALUATION.md`)

Tested on a 120s slice of `04-22会议录音_1.wav` (7 Chinese segments, 179 chars GT):

| Stage | CER |
|-------|-----|
| SenseVoice raw | 0.3208 |
| gemma-4 reviewed | 0.2264 |
| Relative improvement | +29.4% |

Per-job glossary was applied. Larger-sample evaluation is future work.
