import io
import time
from pathlib import Path

import pytest
import soundfile as sf
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Construct app fresh per-test with storage redirected to tmp_path."""
    monkeypatch.setenv("ASR_PORT", "18099")  # avoid port collision in dev
    # Reset module-level singletons so they pick up monkeypatched paths
    import asr_router.server as srv
    srv._settings = None
    srv._store = None
    srv._worker = None
    # Redirect storage_dir at the Settings level
    from asr_router.config import Settings
    real_load = Settings.load
    def patched_load():
        s = real_load()
        # Replace storage_dir with tmp_path
        return Settings(
            host=s.host, port=s.port, api_key=s.api_key,
            omlx_base_url=s.omlx_base_url, omlx_api_key=s.omlx_api_key,
            sense_voice_dir=s.sense_voice_dir, silero_vad_path=s.silero_vad_path,
            diarize_dir=s.diarize_dir, speaker_embed_path=s.speaker_embed_path,
            storage_dir=tmp_path,
            glossary_default=s.glossary_default, routing_yaml=s.routing_yaml,
            pipelines_yaml=s.pipelines_yaml,
        )
    monkeypatch.setattr(Settings, "load", staticmethod(patched_load))
    return TestClient(srv.app)


def test_submit_returns_job_id(client, zh_wav):
    with open(zh_wav, "rb") as f:
        r = client.post(
            "/v1/audio/jobs",
            headers={"Authorization": "Bearer sk-mlx"},
            files={"file": ("zh.wav", f, "audio/wav")},
            data={"glossary": "terms:\n  - term: 测试\n    aliases: []"},
        )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["status"] == "queued"
    assert "id" in j and len(j["id"]) >= 8


def test_submit_requires_auth(client):
    r = client.post("/v1/audio/jobs")
    assert r.status_code == 401


def test_get_unknown_job_returns_404(client):
    r = client.get(
        "/v1/audio/jobs/nonexistent",
        headers={"Authorization": "Bearer sk-mlx"},
    )
    assert r.status_code == 404


def test_submit_then_get_shape(client, zh_wav):
    with open(zh_wav, "rb") as f:
        sub = client.post(
            "/v1/audio/jobs",
            headers={"Authorization": "Bearer sk-mlx"},
            files={"file": ("zh.wav", f, "audio/wav")},
        )
    assert sub.status_code == 200
    jid = sub.json()["id"]
    r = client.get(
        f"/v1/audio/jobs/{jid}",
        headers={"Authorization": "Bearer sk-mlx"},
    )
    assert r.status_code == 200
    j = r.json()
    assert j["id"] == jid
    assert j["status"] in ("queued", "vad_diarize", "transcribing", "reviewing", "rendering", "done", "failed")
    # Audio path should have been persisted into the job dir
    assert "input.wav" in j["audio_path"]
    assert isinstance(j["artifacts"], list)


def test_artifact_404_when_missing(client):
    r = client.get(
        "/v1/audio/jobs/nope/artifact/_summary.md",
        headers={"Authorization": "Bearer sk-mlx"},
    )
    assert r.status_code == 404


def test_artifact_path_traversal_rejected(client, zh_wav, tmp_path):
    """Regression: name with '..' or '/' must NOT escape the artifact dir."""
    # Submit a real job so we have a valid job_id with an artifact_dir
    with open(zh_wav, "rb") as f:
        sub = client.post(
            "/v1/audio/jobs",
            headers={"Authorization": "Bearer sk-mlx"},
            files={"file": ("zh.wav", f, "audio/wav")},
        )
    jid = sub.json()["id"]
    # Plant a file outside the artifact dir but inside storage_dir to prove
    # the guard works against a real traversal target.
    (tmp_path / "secret.db").write_text("sensitive")
    for hostile in (
        "../secret.db",
        "../../etc/passwd",
        "..%2Fsecret.db",  # url-encoded — FastAPI decodes before routing
        ".hidden",
        "subdir/file",
    ):
        r = client.get(
            f"/v1/audio/jobs/{jid}/artifact/{hostile}",
            headers={"Authorization": "Bearer sk-mlx"},
        )
        # 400 (rejected) or 404 (allowed but not found within artifact_dir).
        # 200 would mean the guard failed and a file outside the dir was served.
        assert r.status_code in (400, 404), f"hostile={hostile!r} got {r.status_code}"
