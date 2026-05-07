from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

from jinja2 import Template

from asr_router.glossary import Glossary
from asr_router.meeting.transcribe import SegmentTranscript


_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts/review.j2"


@dataclass(frozen=True)
class ReviewedSegment:
    speaker_id: str
    speaker_role: str
    start: float
    end: float
    text_corrected: str
    changes: list[dict]
    confidence: str
    notes: str
    text_original: str


def _render_prompt(
    segments: list[SegmentTranscript],
    glossary: Glossary,
    prev_context: str,
) -> str:
    tpl = Template(_PROMPT_PATH.read_text(encoding="utf-8"))
    segs_json = json.dumps(
        [
            {
                "speaker": s.speaker, "start": s.start, "end": s.end,
                "text": s.text, "lang": s.lang, "event": s.event,
            }
            for s in segments
        ],
        ensure_ascii=False,
        indent=2,
    )
    return tpl.render(
        glossary=glossary.to_prompt_text(),
        prev_context=prev_context or "(无)",
        segments_json=segs_json,
    )


def _strip_code_fence(text: str) -> str:
    t = text.strip()
    if not t.startswith("```"):
        return t
    # remove opening fence (with optional language tag)
    t = t.split("```", 1)[1]
    if t.lower().startswith("json"):
        t = t[4:]
    # remove trailing fence
    if t.rstrip().endswith("```"):
        t = t.rstrip()[:-3]
    return t.strip()


def _parse_response(text: str) -> dict:
    """Parse JSON output from gemma-4. Tolerates code fences and preamble text."""
    candidate = _strip_code_fence(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Find the first '{' and try to parse from there, balancing braces.
    start = candidate.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(candidate)):
        c = candidate[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(candidate[start:i + 1])
    raise ValueError(f"Unbalanced JSON in response: {text[:200]!r}")


def review_segments(
    raw: list[SegmentTranscript],
    *,
    glossary: Glossary,
    omlx,
    model: str,
    window: int = 4,
    batch: int = 12,
    temperature: float = 0.1,
    timeout_sec: float | None = 180.0,
) -> tuple[list[ReviewedSegment], dict[str, str]]:
    """Run gemma-4 contextual review across `raw` segments.

    Batches `raw` into chunks of size `batch`. Each call carries the last
    `window` reviewed segments as context (read-only "prev_context"). Returns
    (reviewed_segments, speaker_role_map). `text_original` is preserved for diff.
    """
    out: list[ReviewedSegment] = []
    role_map: dict[str, str] = {}
    prev_context = ""

    for i in range(0, len(raw), batch):
        batch_segs = raw[i:i + batch]
        prompt = _render_prompt(batch_segs, glossary, prev_context)

        # Up to 2 attempts: first normal, second with strictly-JSON suffix
        parsed = None
        last_err: Exception | None = None
        for attempt in (1, 2):
            attempt_prompt = prompt
            if attempt == 2:
                attempt_prompt = prompt + (
                    "\n\n再次强调：严格输出合法 JSON 对象，禁止任何额外文字、"
                    "代码栅栏、注释；所有字符串都用双引号包围；不要在 JSON 内"
                    "使用未转义的引号或换行；segments 数组长度必须等于输入数组长度。"
                )
            try:
                resp = omlx.chat(
                    model=model,
                    messages=[{"role": "user", "content": attempt_prompt}],
                    temperature=temperature,
                    timeout=timeout_sec,
                )
                parsed = _parse_response(resp)
                break
            except (json.JSONDecodeError, ValueError, Exception) as e:
                last_err = e
                if attempt == 2:
                    # Both attempts failed — fall back to passthrough for this batch
                    print(
                        f"[review] batch {i // batch} of {(len(raw) + batch - 1) // batch}: "
                        f"both attempts produced malformed JSON; passing through raw text. "
                        f"err: {type(e).__name__}: {str(e)[:200]}",
                        flush=True,
                    )
                    parsed = {
                        "segments": [None] * len(batch_segs),
                        "speaker_role_map": {},
                        "_passthrough_reason": (
                            f"reviewer JSON malformed after 2 attempts: "
                            f"{type(e).__name__}: {str(e)[:120]}"
                        ),
                    }

        rev_list = parsed.get("segments", [])
        passthrough_note = parsed.get("_passthrough_reason")
        if len(rev_list) != len(batch_segs):
            # Fall back to original text for any missing slots so we don't lose data
            rev_list = list(rev_list) + [None] * (len(batch_segs) - len(rev_list))

        for orig, rev in zip(batch_segs, rev_list):
            if rev is None:
                fallback_note = passthrough_note or "reviewer omitted this segment; using original"
                out.append(
                    ReviewedSegment(
                        speaker_id=orig.speaker, speaker_role=orig.speaker,
                        start=orig.start, end=orig.end,
                        text_corrected=orig.text, changes=[],
                        confidence="low",
                        notes=fallback_note,
                        text_original=orig.text,
                    )
                )
                continue
            out.append(
                ReviewedSegment(
                    speaker_id=rev.get("speaker_id", orig.speaker),
                    speaker_role=rev.get("speaker_role", orig.speaker),
                    start=float(rev.get("start", orig.start)),
                    end=float(rev.get("end", orig.end)),
                    text_corrected=rev.get("text_corrected", orig.text),
                    changes=rev.get("changes", []) or [],
                    confidence=rev.get("confidence", "medium"),
                    notes=rev.get("notes", "") or "",
                    text_original=orig.text,
                )
            )

        for sid, role in (parsed.get("speaker_role_map") or {}).items():
            role_map.setdefault(sid, role)

        prev_context = "\n".join(
            f"[{r.speaker_role}] {r.text_corrected}" for r in out[-window:]
        )

    return out, role_map
