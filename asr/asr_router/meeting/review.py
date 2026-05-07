from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
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


def _attempt_batch(
    omlx,
    *,
    model: str,
    base_prompt: str,
    temperature: float,
    timeout_sec: float | None,
    label: str,
) -> tuple[dict | None, str | None]:
    """Run a single batch through gemma-4 with one strict-JSON retry.

    Returns (parsed_dict, passthrough_note). On success passthrough_note is
    None. On both-attempt failure parsed_dict is None and passthrough_note
    is a human-readable diagnostic.
    """
    last_err: Exception | None = None
    for attempt in (1, 2):
        prompt = base_prompt
        if attempt == 2:
            prompt = base_prompt + (
                "\n\n再次强调：严格输出合法 JSON 对象，禁止任何额外文字、"
                "代码栅栏、注释；所有字符串都用双引号包围；不要在 JSON 内"
                "使用未转义的引号或换行；segments 数组长度必须等于输入数组长度。"
            )
        try:
            resp = omlx.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                timeout=timeout_sec,
            )
            return _parse_response(resp), None
        except Exception as e:  # noqa: BLE001 — covers JSONDecodeError + ValueError + httpx errors
            last_err = e
    note = (
        f"reviewer JSON malformed after 2 attempts: "
        f"{type(last_err).__name__}: {str(last_err)[:120]}"
    )
    print(
        f"[review] {label}: both attempts produced malformed JSON; "
        f"passing through raw text. err: {note}",
        flush=True,
    )
    return None, note


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
    parallel_batches: int = 1,
) -> tuple[list[ReviewedSegment], dict[str, str]]:
    """Run gemma-4 contextual review across `raw` segments.

    Strategy: split `raw` into batches of size `batch`, then process them in
    waves of `parallel_batches` concurrent gemma-4 calls. Within a wave,
    every batch sees the same `prev_context` (last `window` reviewed
    segments from the previous wave). Wave size 1 = fully sequential
    (preserves the original sliding-context behavior); larger waves trade
    some context continuity for `~parallel_batches`× speedup, since oMLX
    handles concurrent chat requests up to its scheduler.max_concurrent_requests.

    `text_original` is preserved on each ReviewedSegment for downstream diff.
    """
    out: list[ReviewedSegment] = []
    role_map: dict[str, str] = {}
    prev_context = ""

    if parallel_batches < 1:
        parallel_batches = 1

    all_batches = [raw[i:i + batch] for i in range(0, len(raw), batch)]
    total_batches = len(all_batches)

    for wave_start in range(0, total_batches, parallel_batches):
        wave = all_batches[wave_start: wave_start + parallel_batches]
        prompts = [_render_prompt(b, glossary, prev_context) for b in wave]
        labels = [
            f"batch {wave_start + k + 1}/{total_batches}"
            for k in range(len(wave))
        ]

        if parallel_batches == 1 or len(wave) == 1:
            results = [
                _attempt_batch(
                    omlx, model=model, base_prompt=prompts[0],
                    temperature=temperature, timeout_sec=timeout_sec,
                    label=labels[0],
                )
            ]
        else:
            with ThreadPoolExecutor(max_workers=len(wave)) as ex:
                futures = [
                    ex.submit(
                        _attempt_batch,
                        omlx,
                        model=model,
                        base_prompt=prompts[k],
                        temperature=temperature,
                        timeout_sec=timeout_sec,
                        label=labels[k],
                    )
                    for k in range(len(wave))
                ]
                results = [f.result() for f in futures]

        for batch_segs, (parsed, passthrough_note) in zip(wave, results):
            if parsed is None:
                parsed = {"segments": [None] * len(batch_segs), "speaker_role_map": {}}
            rev_list = parsed.get("segments", [])
            if len(rev_list) != len(batch_segs):
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

        # Context for the NEXT wave: last `window` reviewed segments from
        # everything produced so far.
        prev_context = "\n".join(
            f"[{r.speaker_role}] {r.text_corrected}" for r in out[-window:]
        )

    return out, role_map
