from asr_router.meeting.vad_diarize import DiarizedSegment
from asr_router.meeting.transcribe import transcribe_segments, SegmentTranscript


def test_transcribe_yields_one_per_diarize(zh_wav):
    segs = [
        DiarizedSegment(start=0.0, end=2.5, speaker="Speaker_0"),
        DiarizedSegment(start=2.5, end=5.5, speaker="Speaker_1"),
    ]
    out = transcribe_segments(zh_wav, segs, chunk_max_sec=30)
    assert len(out) == 2
    assert all(isinstance(t, SegmentTranscript) for t in out)
    assert all(t.text for t in out)
    assert out[0].speaker == "Speaker_0"
    assert out[1].speaker == "Speaker_1"
    # speaker assignment preserved
    assert out[0].start == 0.0
    assert out[1].start == 2.5


def test_skips_short_slivers(zh_wav):
    """Segments < 200ms should be skipped silently."""
    segs = [
        DiarizedSegment(start=0.0, end=0.05, speaker="Speaker_0"),  # 50ms — skip
        DiarizedSegment(start=0.05, end=2.5, speaker="Speaker_1"),
    ]
    out = transcribe_segments(zh_wav, segs)
    assert len(out) == 1
    assert out[0].speaker == "Speaker_1"


def test_long_segment_split_into_chunks(zh_wav):
    """Segment longer than chunk_max_sec is split (rare for diarized output, but support it)."""
    # zh.wav is ~5.6s. Use chunk_max_sec=2 to force split.
    segs = [DiarizedSegment(start=0.0, end=5.5, speaker="Speaker_0")]
    out = transcribe_segments(zh_wav, segs, chunk_max_sec=2.0)
    assert len(out) >= 2
    assert all(t.speaker == "Speaker_0" for t in out)
    # Sub-chunk timestamps still relative to full audio
    assert out[0].start == 0.0
    assert out[-1].end <= 5.5 + 0.01
