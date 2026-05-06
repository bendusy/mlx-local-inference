from __future__ import annotations
from pathlib import Path
import httpx
from openai import OpenAI

from asr_router.config import Settings


class OMLXClient:
    """Thin wrapper around oMLX's OpenAI-compatible endpoints.

    Provides:
    - list_model_ids(): GET /models
    - transcribe(wav, model): POST /audio/transcriptions (Whisper-compatible)
    - chat(model, messages, **kwargs): POST /chat/completions (returns content string)
    """

    def __init__(self, base_url: str, api_key: str):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._http = httpx.Client(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=300.0,
        )
        self._oa = OpenAI(base_url=self._base_url, api_key=api_key)

    @classmethod
    def from_settings(cls) -> "OMLXClient":
        s = Settings.load()
        return cls(s.omlx_base_url, s.omlx_api_key)

    def list_model_ids(self) -> list[str]:
        r = self._http.get("/models")
        r.raise_for_status()
        return [m["id"] for m in r.json()["data"]]

    def transcribe(
        self,
        wav_path: Path | str,
        *,
        model: str = "Qwen3-ASR-1.7B-8bit",
    ) -> dict:
        with open(wav_path, "rb") as f:
            r = self._http.post(
                "/audio/transcriptions",
                data={"model": model},
                files={"file": (Path(wav_path).name, f, "audio/wav")},
            )
        r.raise_for_status()
        return r.json()

    def chat(self, *, model: str, messages: list[dict], **kwargs) -> str:
        resp = self._oa.chat.completions.create(model=model, messages=messages, **kwargs)
        return resp.choices[0].message.content or ""

    def close(self) -> None:
        self._http.close()
