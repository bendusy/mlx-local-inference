import json
from pathlib import Path

from asr_router.meeting.render import render_artifacts
from asr_router.meeting.review import ReviewedSegment
from asr_router.meeting.transcribe import SegmentTranscript


class FakeOMLX:
    def __init__(self, summary: str = "## 摘要\n会议讨论了Alpha Group合作。"):
        self._summary = summary
        self.calls = []

    def chat(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self._summary


def _raw():
    return [
        SegmentTranscript(
            speaker="Speaker_0", start=0.0, end=5.0,
            text="Alpa Group欢迎大家", lang="zh", emotion="NEUTRAL", event="Speech",
        ),
        SegmentTranscript(
            speaker="Speaker_1", start=5.0, end=10.0,
            text="多谢", lang="yue", emotion="HAPPY", event="Speech",
        ),
    ]


def _reviewed():
    return [
        ReviewedSegment(
            speaker_id="Speaker_0", speaker_role="主持人",
            start=0.0, end=5.0,
            text_corrected="Alpha Group欢迎大家",
            changes=[{"original": "Alpa", "fixed": "Alpha", "reason": "glossary"}],
            confidence="high", notes="",
            text_original="Alpa Group欢迎大家",
        ),
        ReviewedSegment(
            speaker_id="Speaker_1", speaker_role="主席",
            start=5.0, end=10.0,
            text_corrected="多谢", changes=[],
            confidence="low", notes="背景音较吵",
            text_original="多谢",
        ),
    ]


def test_writes_5_files(tmp_path):
    paths = render_artifacts(
        out_dir=tmp_path, stem="04-22会议录音_1",
        raw=_raw(), reviewed=_reviewed(),
        role_map={"Speaker_0": "主持人", "Speaker_1": "主席"},
        omlx=FakeOMLX(), summary_model="gemma-4-26b-a4b-it-4bit",
    )
    assert (tmp_path / "_raw.json").exists()
    assert (tmp_path / "04-22会议录音_1_sensevoice.md").exists()
    assert (tmp_path / "04-22会议录音_1_gemma4.md").exists()
    assert (tmp_path / "04-22会议录音_1_diff.md").exists()
    assert (tmp_path / "04-22会议录音_1_summary.md").exists()
    assert paths["raw"].name == "_raw.json"


def test_sensevoice_md_keeps_original_text(tmp_path):
    render_artifacts(
        out_dir=tmp_path, stem="x",
        raw=_raw(), reviewed=_reviewed(), role_map={},
        omlx=FakeOMLX(), summary_model="m",
    )
    sv_text = (tmp_path / "x_sensevoice.md").read_text()
    assert "Alpa Group" in sv_text
    assert "Speaker_0" in sv_text


def test_gemma4_md_uses_corrected_text_and_role(tmp_path):
    render_artifacts(
        out_dir=tmp_path, stem="x",
        raw=_raw(), reviewed=_reviewed(),
        role_map={"Speaker_0": "主持人", "Speaker_1": "主席"},
        omlx=FakeOMLX(), summary_model="m",
    )
    g4 = (tmp_path / "x_gemma4.md").read_text()
    assert "Alpha Group" in g4
    assert "[主持人" in g4
    assert "[主席" in g4
    # speaker mapping section present
    assert "说话人映射" in g4
    assert "Speaker_0 → 主持人" in g4
    # low-confidence flag present (⚠️) for the second segment
    assert "⚠️" in g4
    assert "背景音较吵" in g4


def test_diff_md_lists_changes(tmp_path):
    render_artifacts(
        out_dir=tmp_path, stem="x",
        raw=_raw(), reviewed=_reviewed(), role_map={},
        omlx=FakeOMLX(), summary_model="m",
    )
    diff = (tmp_path / "x_diff.md").read_text()
    assert "Alpa" in diff  # original
    assert "Alpha" in diff  # corrected
    assert "glossary" in diff


def test_diff_md_says_no_changes_when_identical(tmp_path):
    raw_only = [SegmentTranscript("Speaker_0", 0.0, 5.0, "测试", "zh", "NEUTRAL", "Speech")]
    rev_only = [ReviewedSegment("Speaker_0", "Speaker_0", 0.0, 5.0, "测试", [], "high", "", "测试")]
    render_artifacts(
        out_dir=tmp_path, stem="x",
        raw=raw_only, reviewed=rev_only, role_map={},
        omlx=FakeOMLX(), summary_model="m",
    )
    diff = (tmp_path / "x_diff.md").read_text()
    assert "无修订" in diff


def test_raw_json_is_valid(tmp_path):
    render_artifacts(
        out_dir=tmp_path, stem="x",
        raw=_raw(), reviewed=_reviewed(), role_map={},
        omlx=FakeOMLX(), summary_model="m",
    )
    data = json.loads((tmp_path / "_raw.json").read_text())
    assert len(data) == 2
    assert data[0]["speaker"] == "Speaker_0"
    assert data[0]["text"] == "Alpa Group欢迎大家"
    assert data[0]["lang"] == "zh"
    assert data[0]["event"] == "Speech"


def test_summary_calls_omlx(tmp_path):
    fake = FakeOMLX()
    render_artifacts(
        out_dir=tmp_path, stem="x",
        raw=_raw(), reviewed=_reviewed(), role_map={},
        omlx=fake, summary_model="my-model",
    )
    assert len(fake.calls) == 1
    assert fake.calls[0]["model"] == "my-model"
    summary = (tmp_path / "x_summary.md").read_text()
    assert "## 摘要" in summary
