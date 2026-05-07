from __future__ import annotations
from pathlib import Path
import json

from jinja2 import Template

from asr_router.meeting.review import ReviewedSegment
from asr_router.meeting.transcribe import SegmentTranscript


_SUMMARY_TPL_PATH = Path(__file__).parent.parent.parent / "prompts/summary.j2"


def _ts(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _render_sv(raw: list[SegmentTranscript]) -> str:
    lines = ["# SenseVoice 原始转录（未复核）", ""]
    for s in raw:
        lines.append(f"**[{s.speaker} {_ts(s.start)}-{_ts(s.end)}]** {s.text}")
        lines.append("")
    return "\n".join(lines)


def _render_g4(reviewed: list[ReviewedSegment], role_map: dict[str, str]) -> str:
    lines = ["# gemma-4 复核版", ""]
    if role_map:
        lines.append("## 说话人映射")
        lines.append("")
        for sid, role in role_map.items():
            lines.append(f"- {sid} → {role}")
        lines.append("")
    lines.append("## 转录")
    lines.append("")
    for r in reviewed:
        flag = " ⚠️" if r.confidence == "low" else ""
        lines.append(f"**[{r.speaker_role} {_ts(r.start)}-{_ts(r.end)}]**{flag} {r.text_corrected}")
        if r.notes:
            lines.append(f"  > 注：{r.notes}")
        lines.append("")
    return "\n".join(lines)


def _render_diff(raw: list[SegmentTranscript], reviewed: list[ReviewedSegment]) -> str:
    lines = ["# SenseVoice → gemma-4 修订对照", ""]
    changed = False
    for orig, rev in zip(raw, reviewed):
        if orig.text == rev.text_corrected and not rev.changes:
            continue
        changed = True
        lines.append(f"### {_ts(orig.start)} {rev.speaker_role}")
        lines.append("")
        lines.append(f"- 原: {orig.text}")
        lines.append(f"- 修: {rev.text_corrected}")
        for c in rev.changes:
            lines.append(f"  - `{c['original']}` → `{c['fixed']}` ({c['reason']})")
        lines.append("")
    if not changed:
        lines.append("（无修订）")
    return "\n".join(lines)


def render_artifacts(
    *,
    out_dir: Path,
    stem: str,
    raw: list[SegmentTranscript],
    reviewed: list[ReviewedSegment],
    role_map: dict[str, str],
    omlx,
    summary_model: str,
    summary_timeout_sec: float | None = 180.0,
) -> dict[str, Path]:
    """Write 5 artifacts to `out_dir`. Returns name -> path mapping.

    Files:
    - _raw.json                  : machine-readable raw segments (audit trail)
    - {stem}_sensevoice.md       : SenseVoice raw concat (Speaker_N + timestamps)
    - {stem}_gemma4.md           : gemma-4 reviewed (semantic roles + confidence flags)
    - {stem}_diff.md             : sv -> g4 modifications, with reasons
    - {stem}_summary.md          : gemma-4 generated summary / decisions / actions
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    raw_json = out_dir / "_raw.json"
    raw_json.write_text(
        json.dumps(
            [
                {
                    "speaker": s.speaker, "start": s.start, "end": s.end,
                    "text": s.text, "lang": s.lang, "event": s.event,
                    "emotion": s.emotion,
                }
                for s in raw
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["raw"] = raw_json

    sv_md = out_dir / f"{stem}_sensevoice.md"
    sv_md.write_text(_render_sv(raw), encoding="utf-8")
    paths["sensevoice"] = sv_md

    g4_md = out_dir / f"{stem}_gemma4.md"
    g4_md.write_text(_render_g4(reviewed, role_map), encoding="utf-8")
    paths["gemma4"] = g4_md

    diff_md = out_dir / f"{stem}_diff.md"
    diff_md.write_text(_render_diff(raw, reviewed), encoding="utf-8")
    paths["diff"] = diff_md

    transcript_text = "\n".join(
        f"[{r.speaker_role}] {r.text_corrected}" for r in reviewed
    )
    summary_prompt = Template(_SUMMARY_TPL_PATH.read_text(encoding="utf-8")).render(
        transcript=transcript_text
    )
    summary_text = omlx.chat(
        model=summary_model,
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.3,
        timeout=summary_timeout_sec,
    )
    summary_md = out_dir / f"{stem}_summary.md"
    summary_md.write_text(summary_text, encoding="utf-8")
    paths["summary"] = summary_md

    return paths
