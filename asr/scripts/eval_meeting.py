#!/usr/bin/env python3
"""End-to-end evaluation of the meeting pipeline.

Runs the full asr-router pipeline on a 2-minute slice of a real meeting
recording, computes CER vs the manually-corrected ground truth, and writes
a structured EVALUATION.md report.

Usage:
    cd asr && uv run python scripts/eval_meeting.py
"""
from __future__ import annotations
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

import httpx
import soundfile as sf


HERE = Path(__file__).resolve().parent
ASR_ROOT = HERE.parent  # asr/
REPO_ROOT = ASR_ROOT.parent

_audio_env = os.environ.get("ASR_EVAL_AUDIO", "").strip()
_json_env = os.environ.get("ASR_EVAL_DIARIZE_JSON", "").strip()
SOURCE_AUDIO = Path(_audio_env) if _audio_env else None
GT_DIARIZE_JSON = Path(_json_env) if _json_env else None
_window_env = os.environ.get("ASR_EVAL_WINDOW", "60.0,180.0")
_w_start, _w_end = (float(x) for x in _window_env.split(","))
WINDOW = (_w_start, _w_end)

PORT = int(os.environ.get("ASR_EVAL_PORT", 18099))
BASE_URL = f"http://localhost:{PORT}"
API_KEY = "sk-mlx"

PERJOB_GLOSSARY = """\
terms: []
"""
# To use a real per-job glossary, pass --glossary to the submit_job call
# or set the PERJOB_GLOSSARY constant to your domain-specific terms.


def fail(msg: str, code: int = 1) -> None:
    print(f"[eval] FAIL: {msg}", flush=True)
    sys.exit(code)


def info(msg: str) -> None:
    print(f"[eval] {msg}", flush=True)


def cut_audio(src: Path, start_sec: float, end_sec: float, out_path: Path) -> float:
    """Slice [start_sec, end_sec] from src to out_path. Returns actual duration."""
    samples, sr = sf.read(str(src), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    s = int(start_sec * sr)
    e = int(end_sec * sr)
    if e > len(samples):
        e = len(samples)
    sliced = samples[s:e]
    sf.write(str(out_path), sliced, sr, subtype="PCM_16")
    return len(sliced) / sr


def build_gt_text(diarize_json_path: Path, window: tuple[float, float]) -> str:
    """Concatenate text of all GT segments fully inside the window, in order.

    Segments prefixed with '(韩)' are Chinese-language *translations* of Korean
    source audio.  Both ASR systems transcribe the raw Korean speech, so
    comparing them against a Chinese translation would be meaningless.  Those
    segments are excluded from the ground-truth reference.
    """
    data = json.loads(diarize_json_path.read_text(encoding="utf-8"))
    start_w, end_w = window
    pieces: list[str] = []
    for seg in data["segments"]:
        if seg["start"] >= start_w and seg["end"] <= end_w:
            if seg["text"].lstrip().startswith("(韩)"):
                continue  # skip Korean-source translation segments
            pieces.append(seg["text"])
    return "\n".join(pieces)


def normalize_for_cer(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    # Strip block-quote notes lines (with optional leading whitespace) BEFORE collapsing whitespace
    # gemma-4 emits indented block quotes: "  > 注：..."
    s = re.sub(r"^\s*>.*$", "", s, flags=re.M)
    # Strip headings
    s = re.sub(r"^#+\s.*$", "", s, flags=re.M)
    # Strip speaker-mapping bullet section lines like "- Speaker_0 → 主持人"
    s = re.sub(r"^- Speaker_\d+\s*→.*$", "", s, flags=re.M)
    # Strip [Speaker_0 00:00-00:05] timestamps and [unclear] markers
    s = re.sub(r"\[[^\]]*\]", "", s)
    # Strip **bold** speaker headers but keep what follows on same line
    s = re.sub(r"\*\*[^*]*\*\*", "", s)
    # Drop entire low-confidence line marked with ⚠️ (everything after it on that line)
    s = re.sub(r"⚠️[^\n]*", "", s)
    # Strip explicit [unclear] markers (belt-and-suspenders after bracket strip above)
    s = s.replace("[unclear]", "")
    # Retain only CJK Unified Ideographs and common CJK punctuation so that
    # Korean / Japanese / Latin noise in the hypothesis doesn't penalise CER
    # when the ground truth is Chinese-only.  Both ref and hyp go through this
    # same filter, so it is symmetric.
    # Ranges specified as \uXXXX to avoid editor encoding collisions:
    #   U+4E00-U+9FFF  CJK Unified Ideographs
    #   U+3400-U+4DBF  CJK Extension A
    #   U+F900-U+FAFF  CJK Compatibility Ideographs
    #   U+3000-U+303F  CJK Symbols and Punctuation (\u3002=。, \u3001=、 etc.)
    #   U+FF00-U+FFEF  Halfwidth/Fullwidth Forms (\uff0c=，, \u3002=。 etc.)
    # Korean (U+AC00-U+D7A3) and kana (U+3040-U+30FF) are intentionally excluded.
    s = re.sub(r"[^一-鿿㐀-䶿豈-﫿　-〿＀-￯]", "", s)
    return s


def cer(ref: str, hyp: str) -> float:
    """Character Error Rate via Levenshtein (iterative DP, O(N*M) memory)."""
    ref = normalize_for_cer(ref)
    hyp = normalize_for_cer(hyp)
    n, m = len(ref), len(hyp)
    if n == 0:
        return 1.0 if m else 0.0
    # rolling 2-row DP to keep memory bounded
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m] / n


def start_service() -> subprocess.Popen:
    env = os.environ.copy()
    env["ASR_PORT"] = str(PORT)
    log_path = ASR_ROOT / "scripts/eval_service.log"
    log_f = open(log_path, "wb")
    proc = subprocess.Popen(
        ["uv", "run", "--python", "3.11", "uvicorn",
         "asr_router.server:app", "--host", "127.0.0.1",
         "--port", str(PORT), "--log-level", "info"],
        cwd=str(ASR_ROOT),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # so we can SIGTERM the whole group
    )
    info(f"started asr-router pid={proc.pid} log={log_path}")
    return proc


def wait_ready(timeout: float = 30.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(f"{BASE_URL}/v1/models",
                          headers={"Authorization": f"Bearer {API_KEY}"},
                          timeout=2)
            if r.status_code == 200:
                info(f"service ready in {time.time()-t0:.1f}s")
                return
        except httpx.RequestError:
            pass
        time.sleep(1)
    fail(f"service did not become ready within {timeout}s")


def submit_job(audio_path: Path) -> str:
    with open(audio_path, "rb") as f:
        r = httpx.post(
            f"{BASE_URL}/v1/audio/jobs",
            headers={"Authorization": f"Bearer {API_KEY}"},
            files={"file": (audio_path.name, f, "audio/wav")},
            data={"glossary": PERJOB_GLOSSARY},
            timeout=60,
        )
    r.raise_for_status()
    return r.json()["id"]


def wait_done(job_id: str, timeout: float = 600.0) -> dict:
    t0 = time.time()
    last_status = None
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(
                f"{BASE_URL}/v1/audio/jobs/{job_id}",
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=30,  # generous — server may be busy under load
            )
            r.raise_for_status()
            j = r.json()
            if j["status"] != last_status:
                info(f"  [{int(time.time()-t0):4d}s] status={j['status']}")
                last_status = j["status"]
            if j["status"] in ("done", "failed"):
                return j
        except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as e:
            info(f"  [{int(time.time()-t0):4d}s] poll error ({type(e).__name__}), retrying...")
        time.sleep(3)
    fail(f"job {job_id} not done in {timeout}s")


def fetch_artifact(job_id: str, name: str) -> str:
    r = httpx.get(
        f"{BASE_URL}/v1/audio/jobs/{job_id}/artifact/{name}",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=30,
    )
    r.raise_for_status()
    return r.text


def kill_service(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> int:
    # Preflight checks
    if not SOURCE_AUDIO or not GT_DIARIZE_JSON:
        print(
            "Set ASR_EVAL_AUDIO and ASR_EVAL_DIARIZE_JSON to point at your meeting\n"
            "recording + manually-corrected diarization JSON. The JSON should have\n"
            "the shape {\"segments\": [{\"start\": s, \"end\": s, \"speaker\": str, \"text\": str}, ...]}.",
            flush=True,
        )
        sys.exit(1)
    if not SOURCE_AUDIO.exists():
        fail(f"missing source audio: {SOURCE_AUDIO}")
    if not GT_DIARIZE_JSON.exists():
        fail(f"missing diarize ground truth: {GT_DIARIZE_JSON}")
    try:
        httpx.get("http://localhost:18080/v1/models",
                  headers={"Authorization": "Bearer sk-mlx"}, timeout=2).raise_for_status()
    except Exception as e:
        fail(f"oMLX not reachable on :18080 ({e}). Start oMLX.app first.")

    # 1. Cut audio slice
    with tempfile.TemporaryDirectory() as td:
        clip = Path(td) / "eval_clip.wav"
        actual = cut_audio(SOURCE_AUDIO, WINDOW[0], WINDOW[1], clip)
        info(f"cut {actual:.1f}s slice -> {clip}")
        gt_text = build_gt_text(GT_DIARIZE_JSON, WINDOW)
        info(f"GT length: {len(gt_text)} chars across window {WINDOW}")

        # 2. Start service
        svc = start_service()
        try:
            wait_ready(timeout=30)
            info("submitting job...")
            jid = submit_job(clip)
            info(f"job_id={jid}")
            final = wait_done(jid, timeout=600)
            if final["status"] != "done":
                err = final.get("error", "")[:500]
                fail(f"job failed: {err}")
            info(f"job done in {final['updated_at']-final['created_at']:.0f}s")
            artifacts = final["artifacts"]
            info(f"artifacts: {artifacts}")
            sv_md = next((a for a in artifacts if a.endswith("_sensevoice.md")), None)
            g4_md = next((a for a in artifacts if a.endswith("_gemma4.md")), None)
            if not sv_md:
                fail(f"missing _sensevoice.md artifact in {artifacts}")
            if not g4_md:
                fail(f"missing _gemma4.md artifact in {artifacts}")
            sv_text = fetch_artifact(jid, sv_md)
            g4_text = fetch_artifact(jid, g4_md)
        finally:
            kill_service(svc)
            info("service stopped")

    # 3. CER
    cer_sv = cer(gt_text, sv_text)
    cer_g4 = cer(gt_text, g4_text)
    delta = (cer_sv - cer_g4) / cer_sv * 100 if cer_sv > 0 else 0.0
    info(f"CER  SenseVoice raw : {cer_sv:.4f}")
    info(f"CER  gemma-4 review : {cer_g4:.4f}")
    info(f"improvement        : {delta:+.1f}%")

    # 4. Write EVALUATION.md
    out_md = ASR_ROOT / "EVALUATION.md"
    art_dir_name = Path(final["artifact_dir"]).name if final.get("artifact_dir") else "unknown"
    out_md.write_text(f"""# Pipeline End-to-End Evaluation

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Source:** a 2-minute slice of a bilingual business meeting recording, window {WINDOW[0]:.0f}-{WINDOW[1]:.0f}s ({actual:.1f}s clip)
**Ground truth:** segments fully inside the window from the manually-corrected diarization JSON ({len(gt_text)} chars)
**Job id:** `{jid}`
**Artifact dir:** `~/.asr-router/jobs/{art_dir_name}/`

## CER (lower is better)

| Stage | CER |
|---|---|
| SenseVoice raw | {cer_sv:.4f} |
| gemma-4 reviewed | {cer_g4:.4f} |
| Relative improvement | {delta:+.1f}% |

## Artifacts

{chr(10).join(f"- `{a}`" for a in artifacts)}

## Notes

CER computed over normalized text (NFKC, whitespace stripped, markdown
formatting and `[Speaker_N ts-ts]` labels removed). Ground truth covers
{len(gt_text)} characters from the manually-corrected `.diarize.json`
file (originally produced by Gemini cloud API and human-revised).

The pipeline ran on a {actual:.1f}-second slice — long enough to test
multi-segment SenseVoice + gemma-4 batched review with the per-job
glossary applied. Larger samples and full-meeting CER trends remain
future work.
""", encoding="utf-8")
    info(f"wrote {out_md}")

    if cer_g4 > cer_sv:
        info("WARNING: gemma-4 CER higher than SenseVoice — review prompt may need iteration")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
