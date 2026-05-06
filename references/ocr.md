# PaddleOCR-VL via oMLX

`PaddleOCR-VL-1.5-6bit` is a vision model served by oMLX at `POST /v1/chat/completions`. It accepts images via `image_url` content blocks (base64 data URL). Auth: see `references/omlx.md`.

This is a general VLM — it answers free-form prompts about images. For pure OCR use prompt `"OCR this document."` or just `"OCR:"`. You can also ask `"What is the total on this receipt?"` etc.

## Python

```python
import base64, httpx

def ocr_image(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    resp = httpx.post(
        "http://localhost:18080/v1/chat/completions",
        headers={"Authorization": "Bearer sk-mlx"},
        json={
            "model": "PaddleOCR-VL-1.5-6bit",
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": "OCR this document."},
                ],
            }],
        },
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

print(ocr_image("scan.jpg"))
```

## cURL

```bash
B64=$(base64 -i scan.jpg)
curl http://localhost:18080/v1/chat/completions \
  -H "Authorization: Bearer sk-mlx" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"PaddleOCR-VL-1.5-6bit\",\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,${B64}\"}},{\"type\":\"text\",\"text\":\"OCR this document.\"}]}]}"
```

## Notes

- `temperature=0.0` for deterministic output.
- RGBA images must be converted to RGB before sending (PIL: `Image.open(p).convert("RGB")`).
- Speed: ~185 t/s on M4. Memory: ~3.3 GB.
- oMLX manages load/unload; `is_pinned` not set — model evicts under memory pressure.
