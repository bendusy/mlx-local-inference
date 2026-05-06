import time
from asr_router.jobs import JobStore, JobStatus, Job


def test_create_get(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job_id = store.create(audio_path="/x/a.wav", glossary_yaml="terms: []")
    j = store.get(job_id)
    assert isinstance(j, Job)
    assert j.id == job_id
    assert j.audio_path == "/x/a.wav"
    assert j.glossary_yaml == "terms: []"
    assert j.status == JobStatus.QUEUED
    assert j.artifact_dir is None
    assert j.error is None
    assert j.created_at > 0
    assert j.updated_at >= j.created_at


def test_get_unknown_returns_none(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    assert store.get("nonexistent") is None


def test_update_status(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job_id = store.create(audio_path="/x/a.wav", glossary_yaml="")
    before_ts = store.get(job_id).updated_at
    time.sleep(0.01)
    store.update(job_id, status=JobStatus.TRANSCRIBING)
    after = store.get(job_id)
    assert after.status == JobStatus.TRANSCRIBING
    assert after.updated_at > before_ts


def test_update_arbitrary_field(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job_id = store.create(audio_path="/x/a.wav", glossary_yaml="")
    store.update(job_id, artifact_dir="/tmp/jobs/abc", error=None)
    j = store.get(job_id)
    assert j.artifact_dir == "/tmp/jobs/abc"


def test_pop_next_queued_returns_oldest_first(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    j1 = store.create(audio_path="/a", glossary_yaml="")
    time.sleep(0.01)
    j2 = store.create(audio_path="/b", glossary_yaml="")
    nxt = store.pop_next_queued()
    assert nxt.id == j1
    assert nxt.status == JobStatus.VAD_DIARIZE  # transitioned out of queued
    nxt2 = store.pop_next_queued()
    assert nxt2.id == j2
    assert store.pop_next_queued() is None


def test_status_string_round_trip(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job_id = store.create(audio_path="/a", glossary_yaml="")
    store.update(job_id, status=JobStatus.DONE)
    j = store.get(job_id)
    assert j.status == JobStatus.DONE
    assert j.status.value == "done"


def test_failure_state(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    job_id = store.create(audio_path="/a", glossary_yaml="")
    store.update(job_id, status=JobStatus.FAILED, error="boom")
    j = store.get(job_id)
    assert j.status == JobStatus.FAILED
    assert j.error == "boom"
