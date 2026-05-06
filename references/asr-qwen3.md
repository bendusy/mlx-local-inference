# Qwen3-ASR via oMLX

`Qwen3-ASR-1.7B-8bit` is served by oMLX at `POST /v1/audio/transcriptions` with a Whisper-compatible API. This is the **quality path** — use it when accuracy matters and latency is secondary. For auto-routing (fast IM messages vs. quality fallback), see `references/asr-routing-module.md`.

Auth: see `references/omlx.md`.

## cURL

```bash
curl http://localhost:18080/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-mlx" \
  -F "file=@audio.wav" \
  -F "model=Qwen3-ASR-1.7B-8bit" \
  -F "language=zh"
```

## Python

```python
import httpx

with open("audio.wav", "rb") as f:
    resp = httpx.post(
        "http://localhost:18080/v1/audio/transcriptions",
        headers={"Authorization": "Bearer sk-mlx"},
        files={"file": ("audio.wav", f, "audio/wav")},
        data={"model": "Qwen3-ASR-1.7B-8bit"},
        timeout=120.0,
    )
resp.raise_for_status()
print(resp.json()["text"])
```

## Response Shape

```json
{
  "text": "transcribed content",
  "language": "Chinese",
  "duration": 5.42,
  "segments": [{"start": 0.0, "end": 5.42, "text": "..."}]
}
```

## Quirks

- `language` returns English names (`"Chinese"`, `"English"`, `"Cantonese"`) — not BCP-47 codes.
- Supported audio: wav, mp3, m4a, flac, ogg, webm.
- 8bit quantized, ~2.3 GB VRAM. oMLX manages load/unload automatically.
