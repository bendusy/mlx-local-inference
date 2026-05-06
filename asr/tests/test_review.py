import json
import pytest

from asr_router.glossary import Glossary
from asr_router.meeting.review import review_segments, ReviewedSegment, _parse_response
from asr_router.meeting.transcribe import SegmentTranscript


class FakeOMLX:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls = []

    def chat(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self._responses.pop(0)


def _seg(speaker, start, end, text):
    return SegmentTranscript(
        speaker=speaker, start=start, end=end, text=text,
        lang="zh", emotion="NEUTRAL", event="Speech",
    )


def test_review_corrects_glossary_term():
    raw = [_seg("Speaker_0", 0.0, 5.0, "我们参观了Alpa Group的工厂")]
    glossary = Glossary.merged({"terms": [{"term": "Alpha Group", "aliases": ["Alpa Group"]}]})
    fake = FakeOMLX([json.dumps({
        "segments": [{
            "speaker_id": "Speaker_0", "speaker_role": "主持人",
            "start": 0.0, "end": 5.0,
            "text_corrected": "我们参观了Alpha Group的工厂",
            "changes": [{"original": "Alpa Group", "fixed": "Alpha Group", "reason": "glossary"}],
            "confidence": "high", "notes": "",
        }],
        "speaker_role_map": {"Speaker_0": "主持人"},
    })])
    reviewed, role_map = review_segments(
        raw, glossary=glossary, omlx=fake, model="gemma-4-26b-a4b-it-4bit",
        window=2, batch=12,
    )
    assert reviewed[0].text_corrected == "我们参观了Alpha Group的工厂"
    assert reviewed[0].text_original == "我们参观了Alpa Group的工厂"
    assert reviewed[0].speaker_role == "主持人"
    assert role_map["Speaker_0"] == "主持人"


def test_review_handles_code_fenced_json():
    """gemma-4 sometimes wraps output in ```json fences. Parser must handle that."""
    raw = [_seg("Speaker_0", 0.0, 5.0, "测试")]
    g = Glossary.merged({})
    fenced = "```json\n" + json.dumps({
        "segments": [{
            "speaker_id": "Speaker_0", "speaker_role": "Speaker_0",
            "start": 0.0, "end": 5.0, "text_corrected": "测试",
            "changes": [], "confidence": "high", "notes": "",
        }],
        "speaker_role_map": {"Speaker_0": "Speaker_0"},
    }) + "\n```"
    fake = FakeOMLX([fenced])
    reviewed, _ = review_segments(raw, glossary=g, omlx=fake, model="m", window=2, batch=12)
    assert reviewed[0].text_corrected == "测试"


def test_review_handles_preamble_before_json():
    """Some models prefix with 'Here is the result:' — parser should still find the JSON."""
    raw = [_seg("Speaker_0", 0.0, 5.0, "测试")]
    g = Glossary.merged({})
    text = '回答：\n' + json.dumps({
        "segments": [{
            "speaker_id": "Speaker_0", "speaker_role": "Speaker_0",
            "start": 0.0, "end": 5.0, "text_corrected": "测试",
            "changes": [], "confidence": "high", "notes": "",
        }],
        "speaker_role_map": {},
    })
    fake = FakeOMLX([text])
    reviewed, _ = review_segments(raw, glossary=g, omlx=fake, model="m", window=2, batch=12)
    assert reviewed[0].text_corrected == "测试"


def test_review_batches_segments():
    """Tasks larger than `batch` should produce multiple chat calls."""
    raw = [_seg(f"Speaker_{i%2}", float(i), float(i + 1), f"片段{i}") for i in range(5)]
    g = Glossary.merged({})

    def make_resp(start_idx: int, count: int) -> str:
        return json.dumps({
            "segments": [
                {
                    "speaker_id": f"Speaker_{(start_idx + j) % 2}",
                    "speaker_role": "Role",
                    "start": float(start_idx + j),
                    "end": float(start_idx + j + 1),
                    "text_corrected": f"片段{start_idx + j}",
                    "changes": [], "confidence": "high", "notes": "",
                }
                for j in range(count)
            ],
            "speaker_role_map": {},
        })

    fake = FakeOMLX([make_resp(0, 2), make_resp(2, 2), make_resp(4, 1)])
    reviewed, _ = review_segments(raw, glossary=g, omlx=fake, model="m", window=2, batch=2)
    assert len(reviewed) == 5
    assert len(fake.calls) == 3


def test_parse_response_plain_json():
    s = '{"segments": [], "speaker_role_map": {}}'
    parsed = _parse_response(s)
    assert parsed["segments"] == []


def test_parse_response_invalid_raises():
    with pytest.raises((ValueError, Exception)):
        _parse_response("not json at all, no braces")
