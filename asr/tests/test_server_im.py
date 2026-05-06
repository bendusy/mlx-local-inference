from fastapi.testclient import TestClient
from asr_router.server import app


def test_auth_required():
    c = TestClient(app)
    r = c.post("/v1/audio/transcriptions")
    # Missing auth header -> 401. Note: missing file would be 422; the auth check
    # must happen before file parsing for the 401 to surface here.
    assert r.status_code == 401


def test_short_zh_via_sense_voice(zh_wav):
    c = TestClient(app)
    with open(zh_wav, "rb") as f:
        r = c.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer sk-mlx"},
            data={"model": "auto"},
            files={"file": ("zh.wav", f, "audio/wav")},
        )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["text"]
    assert j["language"] == "zh"
    assert j["x_route"]["upstream"] == "sense_voice"


def test_quality_high_routes_to_omlx(zh_wav):
    c = TestClient(app)
    with open(zh_wav, "rb") as f:
        r = c.post(
            "/v1/audio/transcriptions",
            headers={"Authorization": "Bearer sk-mlx"},
            data={"model": "auto", "quality": "high"},
            files={"file": ("zh.wav", f, "audio/wav")},
        )
    assert r.status_code == 200, r.text
    assert r.json()["x_route"]["upstream"] == "omlx"


def test_models_endpoint():
    c = TestClient(app)
    r = c.get("/v1/models", headers={"Authorization": "Bearer sk-mlx"})
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert "auto" in ids
    assert "sense_voice" in ids
    assert "Qwen3-ASR-1.7B-8bit" in ids
