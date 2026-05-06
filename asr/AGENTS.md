# asr-router — Integration Guide for AI Agents

This document is the **authoritative integration spec** for AI agents (Claude, OpenAI, Gemini, custom LLM clients, automation scripts) calling the local ASR routing service.

If you are an AI agent reading this: skim §1, then jump to §3 (IM mode) or §4 (Meeting mode) depending on the task. All examples are runnable.

---

## 1. Quick Reference

```
Service       :  asr-router (FastAPI, runs as launchd com.user.asr-router)
Base URL      :  http://<host>:18081/v1
                 - localhost / 127.0.0.1 (same machine)
                 - M4.local / 10.x.x.x   (any LAN device)
API key       :  sk-mlx                  (Bearer token)
Compatibility :  Whisper-compatible /v1/audio/transcriptions
                 OpenAI-compatible /v1/models
Two modes     :  IM (sync, ≤ 1s) and Meeting (async job, minutes)
```

### Decide which mode

| Audio characteristic | Use this mode |
|---|---|
| ≤ 30 seconds, voice message / dictation / command | **IM mode** (§3) |
| Long-form meeting, > 30 seconds, multi-speaker, want speaker labels | **Meeting mode** (§4) |
| Need raw text fast, will post-process yourself | **IM mode** with `quality=fast` |
| Need highest quality single-pass | **IM mode** with `quality=high` (forces oMLX upstream) |
| Need glossary / domain terms applied | **Meeting mode** (only meeting mode supports glossary) |

### Authentication

Every request must include:
```
Authorization: Bearer sk-mlx
```
Missing or wrong key → `401 Unauthorized`.

---

## 2. Endpoints At-A-Glance

| Method | Path | Purpose | Mode |
|---|---|---|---|
| GET  | `/v1/models` | List available logical models (`auto`, `sense_voice`, `Qwen3-ASR-1.7B-8bit`) | meta |
| POST | `/v1/audio/transcriptions` | Sync transcribe + auto-route | IM |
| POST | `/v1/audio/jobs` | Submit a meeting recording for async multi-pass processing | Meeting |
| GET  | `/v1/audio/jobs/{id}` | Poll job status + list artifacts | Meeting |
| GET  | `/v1/audio/jobs/{id}/artifact/{name}` | Download a specific artifact file | Meeting |

---

## 3. IM Mode — `POST /v1/audio/transcriptions`

Whisper-compatible. Sends one short audio clip, gets one JSON back. The router internally chooses between local SenseVoice (fast, 60–300 ms) and the oMLX Qwen3-ASR upstream (slower, higher quality) based on duration / event tags / explicit `quality` hint.

### Request (multipart/form-data)

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | binary | yes | wav / mp3 / m4a / flac. Mono preferred (stereo is averaged). Any sample rate; recommended 16 kHz. |
| `model` | string | no, default `auto` | `auto` (let router decide), `sense_voice` (force fast), `Qwen3-ASR-1.7B-8bit` (force quality). |
| `quality` | string | no | `fast` → force SenseVoice. `high` → force oMLX. Overrides default routing. |
| `response_format` | string | no, default `json` | Only `json` is implemented. |

### Response 200 (sense_voice path)

```json
{
  "text": "开放时间早上9点至下午5点。",
  "language": "zh",
  "duration": 5.592,
  "x_route": {
    "upstream": "sense_voice",
    "reason": "default",
    "decode_ms": 274.6
  },
  "x_tags": {
    "emotion": "NEUTRAL",
    "event": "Speech"
  }
}
```

### Response 200 (omlx path, `quality=high` or duration > 30 s)

```json
{
  "text": "...",
  "language": "Chinese",
  "duration": 0.69,
  "segments": [{ "text": "...", "start": 0.0, "end": 5.6 }],
  "x_route": {
    "upstream": "omlx",
    "reason": "rule[0]:{'request_param': {'quality': 'high'}}"
  },
  "x_tags": {
    "emotion": "NEUTRAL",
    "event": "Speech",
    "lid": "zh"
  }
}
```

### Field semantics

- **`text`** — plain transcription. Preserve as-is.
- **`language`** — sense_voice path: BCP-47-ish (`zh`/`en`/`yue`/`ja`/`ko`/`nospeech`). omlx path: full English name (`Chinese`, `English`, …). Treat case-insensitively.
- **`x_route.upstream`** ∈ {`sense_voice`, `omlx`}. Routing trace; useful for telemetry.
- **`x_route.reason`** — which rule matched (or `default`).
- **`x_route.decode_ms`** — present only on sense_voice path. Local decode time.
- **`x_tags.event`** ∈ {`Speech`, `BGM`, `Applause`, `Laughter`, `Cry`, `Sneeze`, `Breath`, `Cough`, `unknown`}.
- **`x_tags.emotion`** ∈ {`NEUTRAL`, `HAPPY`, `SAD`, `ANGRY`, `FEARFUL`, `DISGUSTED`, `SURPRISED`, ``}.

### Default routing rules

(From `routing.yaml`. First match wins; otherwise `sense_voice`.)

| Rule | Condition | Routed to |
|---|---|---|
| 1 | `quality=high` | omlx |
| 2 | `quality=fast` | sense_voice |
| 3 | duration > 30 s | omlx |
| 4 | event ∈ {BGM, Applause, Laughter} | omlx |
| _default_ | (everything else) | sense_voice |

### Examples

**curl:**
```bash
curl -s http://localhost:18081/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-mlx" \
  -F "file=@voice.wav" \
  -F "model=auto"
```

**Python (OpenAI SDK — works because Whisper-compatible):**
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:18081/v1", api_key="sk-mlx")
with open("voice.wav", "rb") as f:
    r = client.audio.transcriptions.create(model="auto", file=f)
print(r.text)
# r is a TranscriptionVerbose; SDK won't expose x_route/x_tags.
# For routing visibility use raw httpx (below).
```

**Python (raw httpx — exposes routing fields):**
```python
import httpx
with open("voice.wav", "rb") as f:
    r = httpx.post(
        "http://localhost:18081/v1/audio/transcriptions",
        headers={"Authorization": "Bearer sk-mlx"},
        files={"file": ("voice.wav", f, "audio/wav")},
        data={"model": "auto", "quality": "high"},
        timeout=120,
    )
print(r.json())
```

**TypeScript / Node fetch:**
```ts
const fd = new FormData();
fd.append("file", new Blob([buf], { type: "audio/wav" }), "voice.wav");
fd.append("model", "auto");
const res = await fetch("http://localhost:18081/v1/audio/transcriptions", {
  method: "POST",
  headers: { Authorization: "Bearer sk-mlx" },
  body: fd,
});
const json = await res.json();
console.log(json.text, json.x_route);
```

### IM mode error responses

| Code | Meaning | Likely cause |
|---|---|---|
| 401 | Missing or wrong API key | Add `Authorization: Bearer sk-mlx` |
| 422 | Bad multipart body | `file` field missing |
| 5xx | Upstream / model load failure | Check `asr/logs/asr-router.err.log` |

---

## 4. Meeting Mode — async pipeline

Async job for long-form meeting recordings. Pipeline:

```
audio.wav  ─▶  Pass 1 VAD + speaker diarization (sherpa-onnx)
            ─▶ Pass 2 SenseVoice transcription per diarized segment
            ─▶ Pass 3 gemma-4 contextual review (with merged glossary
                       + speaker-role inference)
            ─▶ Pass 4 render 5 model-named artifacts
```

State machine: `queued → vad_diarize → transcribing → reviewing → rendering → done` (or `failed`).

Validated: SenseVoice raw CER 0.32 → gemma-4 reviewed CER 0.23 = **+29.4% relative improvement** on a real bilingual meeting recording (see `asr/EVALUATION.md`).

### 4.1 Submit — `POST /v1/audio/jobs`

**Request (multipart/form-data):**

| Field | Type | Required | Notes |
|---|---|---|---|
| `file` | binary | yes | The full meeting recording. Will be persisted to `~/.asr-router/jobs/{job_id}/input.wav`. |
| `glossary` | string | no, default `""` | YAML string — see §4.5. Merged with `asr/glossary/default.yaml`. |

**Response 200:**

```json
{ "id": "186665f7d531", "status": "queued" }
```

The `id` is opaque (12-char hex). Persist it for polling.

### 4.2 Poll — `GET /v1/audio/jobs/{id}`

**Response 200 (in progress):**

```json
{
  "id": "186665f7d531",
  "status": "transcribing",
  "audio_path": "/Users/<u>/.asr-router/jobs/186665f7d531/input.wav",
  "artifact_dir": "/Users/<u>/.asr-router/jobs/186665f7d531",
  "artifacts": [],
  "error": null,
  "created_at": 1778097400.123,
  "updated_at": 1778097430.456
}
```

**Response 200 (done):**

```json
{
  "id": "186665f7d531",
  "status": "done",
  "audio_path": "/Users/<u>/.asr-router/jobs/186665f7d531/input.wav",
  "artifact_dir": "/Users/<u>/.asr-router/jobs/186665f7d531",
  "artifacts": [
    "_raw.json",
    "input.wav",
    "input_diff.md",
    "input_gemma4.md",
    "input_sensevoice.md",
    "input_summary.md"
  ],
  "error": null,
  "created_at": 1778097400.123,
  "updated_at": 1778097520.789
}
```

**Response 200 (failed):**

```json
{
  "id": "...",
  "status": "failed",
  "error": "RuntimeError: ...\n<full traceback>",
  ...
}
```

**Response 404** — unknown `id`.

### 4.3 Polling pattern

Recommended: poll every **5 seconds**, max wait **600 s** for short recordings (≤ 5 min audio), **1800 s** for long meetings (~1 hour audio). Job state changes are infrequent — don't hammer faster than 1 Hz.

```python
import httpx, time
def wait_done(job_id: str, base="http://localhost:18081/v1", key="sk-mlx", timeout=1800):
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        r = httpx.get(f"{base}/audio/jobs/{job_id}",
                      headers={"Authorization": f"Bearer {key}"}, timeout=10)
        r.raise_for_status()
        j = r.json()
        if j["status"] != last:
            print(f"[{int(time.time()-t0):4d}s] {j['status']}")
            last = j["status"]
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(5)
    raise TimeoutError(f"job {job_id} not done in {timeout}s")
```

### 4.4 Fetch artifact — `GET /v1/audio/jobs/{id}/artifact/{name}`

Returns the file body. Names listed in the `artifacts` array of the status response.

| Filename | Format | Use it for |
|---|---|---|
| `{stem}_sensevoice.md` | Markdown | Raw transcription with `Speaker_N` IDs and timestamps. Compare with `_gemma4.md` to audit corrections. |
| `{stem}_gemma4.md` | Markdown | **Final corrected transcript** with semantic speaker roles (e.g. `[主持人]`, `[Chairman Zhang]`), `⚠️` flags on low-confidence segments. **Use this as primary output.** |
| `{stem}_diff.md` | Markdown | Side-by-side `原:` / `修:` for each modified segment with a reason (`glossary` / contextual). For audit / human review. |
| `{stem}_summary.md` | Markdown | gemma-4-generated `## 摘要 / 决议 / 待办 / 关键议题` sections. Skim for action items. |
| `_raw.json` | JSON | Machine-readable raw segments (one per diarized chunk) with timestamps, lang, event, emotion. For programmatic post-processing. |

`{stem}` is derived from the uploaded filename. Currently audio is persisted as `input.wav`, so artifact stems are `input_*.md`.

**Path-safety note:** `name` cannot contain `/`, `\`, or start with `.`. Trying to escape the job dir returns 400.

```bash
curl -s "http://localhost:18081/v1/audio/jobs/$JOB/artifact/input_gemma4.md" \
  -H "Authorization: Bearer sk-mlx"
```

### 4.5 Glossary YAML format

Glossary is merged from two sources:
1. `asr/glossary/default.yaml` (server-side defaults, baked in)
2. `glossary` form field in your job submission (per-job, optional)

Aliases from both sources are **unioned** for the same term. Same shape both:

```yaml
terms:
  - term: "Alpha Group"          # canonical form
    aliases: ["Alpha", "Alpa Group"]   # known mis-transcriptions
  - term: "Chairman Zhang"
    aliases: ["Mr. Zhang", "Zhang"]
  - term: "Beta Corp"
    aliases: []                  # canonical with no known aliases
```

The reviewer prompt receives this glossary verbatim and is instructed to:
1. Replace any alias occurrence with the canonical `term` and record the change.
2. Apply consistent capitalization / character form for the canonical term.

**Tip for AI agents:** if you have access to the user's contact list, project codename list, or domain term list, build the glossary dynamically per job. The marginal CER win from a tight glossary is significant (we measured ~10 pp).

### 4.6 Full meeting-mode example

```python
import httpx, time

BASE = "http://localhost:18081/v1"
KEY = "sk-mlx"
H = {"Authorization": f"Bearer {KEY}"}

GLOSSARY = """\
terms:
  - term: "Alpha Group"
    aliases: ["Alpha", "Alpa Group"]
  - term: "Chairman Zhang"
    aliases: ["Mr. Zhang"]
"""

# 1. Submit
with open("meeting.wav", "rb") as f:
    r = httpx.post(f"{BASE}/audio/jobs", headers=H,
                    files={"file": ("meeting.wav", f, "audio/wav")},
                    data={"glossary": GLOSSARY}, timeout=60)
job_id = r.json()["id"]

# 2. Poll
t0 = time.time()
while True:
    j = httpx.get(f"{BASE}/audio/jobs/{job_id}", headers=H, timeout=10).json()
    print(f"[{int(time.time()-t0):4d}s] {j['status']}")
    if j["status"] in ("done", "failed"):
        break
    time.sleep(5)

if j["status"] == "failed":
    raise RuntimeError(j["error"])

# 3. Fetch the corrected transcript and summary
transcript = httpx.get(f"{BASE}/audio/jobs/{job_id}/artifact/input_gemma4.md",
                        headers=H, timeout=30).text
summary = httpx.get(f"{BASE}/audio/jobs/{job_id}/artifact/input_summary.md",
                     headers=H, timeout=30).text
print(transcript)
print("---")
print(summary)
```

### 4.7 Meeting mode error handling

| Code | Meaning |
|---|---|
| 401 | Missing/wrong API key |
| 404 | Unknown job_id, or artifact not found |
| 400 | Malformed artifact name (path traversal attempt) |
| `status: failed` (200 OK with body) | Pipeline stage threw. Read `error` field; common causes are out-of-disk, audio format issues, oMLX dropping connection mid-review. |

---

## 5. Operational Notes

### Service status

```bash
launchctl list | grep com.user.asr-router  # PID + last exit code
tail -f asr/logs/asr-router.err.log         # uvicorn + Worker logs
```

### Restart

```bash
launchctl kickstart -k gui/$(id -u)/com.user.asr-router
```

### Cold start cost

- First IM call after launchd boot: SenseVoice loads (~500 ms model init).
- First `quality=high` (oMLX path): oMLX may be cold-loading `Qwen3-ASR-1.7B-8bit` from SSD — can take 5-30 s on first call. Subsequent calls are 600-800 ms.
- First meeting job: pyannote-segmentation + 3D-Speaker embedding load on Pass 1 (~2 s).

After warmup, latency is stable.

### Concurrency

The Worker runs sequentially, processing **one meeting job at a time** to avoid VRAM thrash with the gemma-4-26b LLM. IM mode has no such serialization — short clips are processed concurrently up to FastAPI's default thread pool.

### Storage layout

```
~/.asr-router/
├── jobs.db                              # SQLite job state
└── jobs/
    └── <job_id>/
        ├── input.wav                    # uploaded audio (persisted, not deleted)
        ├── _raw.json
        ├── input_sensevoice.md
        ├── input_gemma4.md
        ├── input_diff.md
        └── input_summary.md
```

Old jobs accumulate. There is currently **no auto-cleanup** — your agent can `rm -rf ~/.asr-router/jobs/<id>` after fetching artifacts if desired.

### What if oMLX (port 18080) is down?

- IM mode `quality=fast` and `default → sense_voice` rules → **still works** (no oMLX dependency).
- IM mode `quality=high` → 5xx (oMLX path unreachable).
- Meeting mode → Pass 3 (gemma-4 review) will fail; jobs end in `failed` with traceback. Pass 1+2 outputs (`_raw.json`, `_sensevoice.md`) are still written before failure, so the unreviewed transcript is recoverable.

---

## 6. Choosing the right tool — decision flow for agents

```
                  ┌─────────────────────────────────┐
                  │  Have audio, want text?         │
                  └─────────────────────────────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                ▼                  ▼                  ▼
        ≤ 30 s clip        > 30 s recording      Need glossary
        no glossary        OR want speaker        applied
        OK with raw        labels OR action
        text               items
                │                  │                  │
                ▼                  ▼                  ▼
         POST /v1/audio/    POST /v1/audio/    POST /v1/audio/
         transcriptions     jobs               jobs
         (sub-second)       (1-30 min)         (1-30 min)
```

**Short test for "is this a meeting?":** if the audio has more than one speaker OR is longer than ~1 minute, prefer meeting mode. The CER win from contextual review is consistent at +25–30 % relative.

---

## 7. Versioning & stability

This service follows **internal-API stability**:
- Endpoint paths, HTTP methods, request/response field names: stable; will not change without a deprecation cycle.
- Routing rule defaults in `routing.yaml`: tunable; an agent should not rely on a specific rule index.
- Pipeline internals (model versions, prompt templates): may change without notice. Output artifact filenames are stable.
- The `x_route.reason` string is human-readable but **not machine-stable**. Don't pattern-match on it; check `x_route.upstream` instead.
