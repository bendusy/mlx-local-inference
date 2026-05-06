import pytest
from asr_router.models.omlx_client import OMLXClient


@pytest.fixture(scope="module")
def client():
    return OMLXClient.from_settings()


def test_models_list(client):
    ids = client.list_model_ids()
    assert "Qwen3-ASR-1.7B-8bit" in ids
    assert "gemma-4-26b-a4b-it-4bit" in ids


def test_transcribe(client, zh_wav):
    r = client.transcribe(zh_wav, model="Qwen3-ASR-1.7B-8bit")
    assert r["text"]
    # oMLX returns "Chinese" or "zh" depending on model — accept both
    lang = r.get("language", "").lower()
    assert "chinese" in lang or "zh" in lang


def test_chat_review(client):
    r = client.chat(
        model="gemma-4-26b-a4b-it-4bit",
        messages=[
            {"role": "system", "content": "You output JSON only."},
            {"role": "user", "content": "Output exactly this JSON object and nothing else: {\"ok\": true}"},
        ],
        temperature=0.0,
    )
    assert "ok" in r.lower()
