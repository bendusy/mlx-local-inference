def test_pipeline_imports():
    """Smoke test: orchestrator + worker import cleanly with all deps wired."""
    from asr_router.meeting.pipeline import run_job, Worker
    assert callable(run_job)
    assert Worker is not None
