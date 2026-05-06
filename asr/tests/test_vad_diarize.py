import os
from pathlib import Path
import pytest

from asr_router.meeting.vad_diarize import vad_diarize, DiarizedSegment


_sample_env = os.environ.get("ASR_TEST_DIARIZE_AUDIO", "")
SAMPLE = Path(_sample_env) if _sample_env else None

_sample_missing = SAMPLE is None or not SAMPLE.exists()


@pytest.mark.skipif(_sample_missing,
                    reason="set ASR_TEST_DIARIZE_AUDIO to a multi-speaker wav to run this test")
def test_returns_segments():
    segs = vad_diarize(SAMPLE)
    assert len(segs) >= 1
    assert all(isinstance(s, DiarizedSegment) for s in segs)
    assert all(s.end > s.start for s in segs)
    assert all(s.speaker.startswith("Speaker_") for s in segs)
    speakers = {s.speaker for s in segs}
    # 30s sample is unlikely to have only 1 speaker but allow it
    assert 1 <= len(speakers) <= 8


@pytest.mark.skipif(_sample_missing,
                    reason="set ASR_TEST_DIARIZE_AUDIO to a multi-speaker wav to run this test")
def test_segments_sorted():
    segs = vad_diarize(SAMPLE)
    starts = [s.start for s in segs]
    assert starts == sorted(starts), "segments must be sorted by start time"
