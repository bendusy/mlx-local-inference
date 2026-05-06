# Qwen3-Embedding via oMLX

`Qwen3-Embedding-0.6B-4bit-DWQ` is served by oMLX at `POST /v1/embeddings`. Standard OpenAI embeddings API shape. Auth: see `references/omlx.md`.

## cURL

```bash
curl http://localhost:18080/v1/embeddings \
  -H "Authorization: Bearer sk-mlx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Embedding-0.6B-4bit-DWQ",
    "input": "text to embed"
  }'
```

## Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:18080/v1", api_key="sk-mlx")

resp = client.embeddings.create(
    model="Qwen3-Embedding-0.6B-4bit-DWQ",
    input=["first document", "second document"],
)
vectors = [d.embedding for d in resp.data]
print(f"dim={len(vectors[0])}")
```

## Response Shape

```json
{
  "object": "list",
  "data": [{"object": "embedding", "index": 0, "embedding": [0.012, ...]}],
  "model": "Qwen3-Embedding-0.6B-4bit-DWQ",
  "usage": {"prompt_tokens": 4, "total_tokens": 4}
}
```

## Notes

- Embedding dimension: check live via `len(resp.data[0].embedding)` — verify against live response.
- Batch up to ~256 inputs in a single request.
- 4bit DWQ quantization. oMLX manages load/unload; no service restart needed.
- For higher precision semantic matching, a larger model may be preferable — check `/v1/models` for available options.
