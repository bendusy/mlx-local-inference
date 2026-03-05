#!/usr/bin/env python3
"""
Transcribe daemon: watches ~/transcribe/ for audio files.
Two-phase processing to avoid MLX memory contention:
  Phase 1: Transcribe ALL pending files (Qwen3-ASR, with ffmpeg chunking)
  Phase 2: Unload ASR model, then LLM-correct all pending files
"""

import json
import os
import re
import sys
import time
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx


# ============================================================================
# Configuration
# ============================================================================

WATCH_DIR = Path.home() / "transcribe"
DONE_DIR = WATCH_DIR / "done"
POLL_INTERVAL = 15  # seconds

ASR_API = "http://127.0.0.1:8788/v1"     # mlx-audio server (Qwen3-ASR)
LLM_API = "http://127.0.0.1:8787/v1"     # mlx-openai-server (Qwen3.5-35B)
ASR_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
LLM_MODEL = "qwen3.5-35b"

CHUNK_MINUTES = 10
MAX_WORKERS = 1  # mlx-audio server is single-worker, serialize requests

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}

LLM_CORRECTION_PROMPT = (
    "你是一个专业的语音转录校对编辑。录音内容包含普通话和粤语，可能混合使用。\n"
    "校对规则：\n"
    "1. 修正同音字错误（如 的/得/地，在/再，他/她）\n"
    "2. 粤语内容保留粤语用字（嘅、唔、咁、喺、冇、佢、啲、嘢），不要转换成普通话\n"
    "3. 添加正确的标点符号和分段\n"
    "4. 去除语气词和重复（呃、嗯、就是说、然后然后）\n"
    "5. 修正领域专有名词\n"
    "6. 保持原始语言，不要翻译\n"
    "只返回校对后的文本，不要添加任何说明。"
)


# ============================================================================
# Logging
# ============================================================================

def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ============================================================================
# Audio splitting
# ============================================================================

def get_audio_duration(audio_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def split_audio(audio_path: Path, output_dir: Path, duration: float) -> list[Path]:
    """Split audio into chunks (no overlap — Qwen3-ASR handles context well)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_seconds = CHUNK_MINUTES * 60
    total_chunks = int((duration + chunk_seconds - 1) / chunk_seconds)

    chunks = []
    for i in range(total_chunks):
        start = i * chunk_seconds
        chunk_file = output_dir / f"chunk_{i:03d}.wav"

        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", str(start),
            "-t", str(chunk_seconds),
            "-i", str(audio_path),
            "-filter:a", "silenceremove=stop_periods=-1:stop_duration=0.5:stop_threshold=-30dB,atempo=1.5",
            "-ar", "16000", "-ac", "1",
            str(chunk_file)
        ]
        subprocess.run(cmd, check=True)
        chunks.append(chunk_file)

    return chunks


# ============================================================================
# Transcription (Qwen3-ASR via mlx-audio server)
# ============================================================================

def transcribe_chunk(audio_file: Path) -> str:
    """Transcribe a single audio chunk."""
    with open(audio_file, 'rb') as f:
        files = {'file': (audio_file.name, f, 'audio/wav')}
        data = {'model': ASR_MODEL, 'language': 'zh'}

        with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
            response = client.post(
                f"{ASR_API}/audio/transcriptions",
                files=files, data=data
            )
            response.raise_for_status()
            text = response.text.strip()
            try:
                j = json.loads(text)
                if isinstance(j, dict) and "text" in j:
                    return j["text"].strip()
            except (json.JSONDecodeError, KeyError):
                pass
            return text


def transcribe_file(audio_path: Path, chunks_dir: Path) -> str:
    """Transcribe audio file. Short files direct, long files split + parallel."""
    duration = get_audio_duration(audio_path)
    log(f"  Duration: {duration/60:.1f} min")

    # Always process via ffmpeg to apply 1.5x speedup
    # if duration <= CHUNK_MINUTES * 60:
    #     log(f"  Short file, transcribing directly...")
    #     return transcribe_chunk(audio_path)

    total_chunks = int((duration + CHUNK_MINUTES * 60 - 1) / (CHUNK_MINUTES * 60))
    log(f"  Splitting into {total_chunks} chunks...")
    chunk_files = split_audio(audio_path, chunks_dir, duration)

    results = {}

    def worker(idx: int, chunk_file: Path) -> tuple[int, str]:
        start_time = time.time()
        text = transcribe_chunk(chunk_file)
        elapsed = time.time() - start_time
        # Save individual chunk transcription
        chunk_txt = chunks_dir / f"chunk_{idx:03d}.txt"
        chunk_txt.write_text(text, encoding='utf-8')
        log(f"  Chunk {idx:03d} done ({elapsed:.1f}s)")
        return idx, text

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(worker, i, cf): i
            for i, cf in enumerate(chunk_files)
        }
        for future in as_completed(futures):
            idx, text = future.result()
            results[idx] = text

    merged = []
    for i in range(total_chunks):
        if i in results:
            merged.append(results[i])
    return "\n\n".join(merged)


def unload_asr_model():
    """Unload ASR model from mlx-audio server to free memory for LLM."""
    try:
        with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
            response = client.request(
                "DELETE", f"{ASR_API}/models",
                params={"model_name": ASR_MODEL}
            )
            if response.status_code == 200:
                log(f"Unloaded ASR model to free memory for LLM")
            else:
                log(f"ASR unload returned {response.status_code}")
    except Exception as e:
        log(f"Failed to unload ASR model: {e}")


# ============================================================================
# LLM Correction
# ============================================================================

def correct_text(text: str) -> str:
    """Correct transcription text using LLM. Process in ~2000 char chunks."""
    lines = text.split('\n')
    chunks = []
    current_chunk = []
    current_len = 0

    for line in lines:
        if current_len + len(line) > 2000 and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = [line]
            current_len = len(line)
        else:
            current_chunk.append(line)
            current_len += len(line)

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    if not chunks:
        return text

    log(f"  Correcting {len(chunks)} text chunks...")

    corrected_parts = []
    for i, chunk in enumerate(chunks):
        try:
            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": LLM_CORRECTION_PROMPT},
                    {"role": "user", "content": chunk}
                ],
                "temperature": 0.3,
                "max_tokens": 4096,
            }

            with httpx.Client(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                response = client.post(
                    f"{LLM_API}/chat/completions", json=payload
                )
                response.raise_for_status()
                result = response.json()
                corrected = result["choices"][0]["message"]["content"].strip()
                corrected = re.sub(r'<think>.*?</think>\s*', '', corrected, flags=re.DOTALL)
                corrected_parts.append(corrected)
                log(f"  Correction chunk {i+1}/{len(chunks)} done")
        except Exception as e:
            log(f"  Correction chunk {i+1}/{len(chunks)} failed: {e}")
            corrected_parts.append(chunk)

    return "\n\n".join(corrected_parts)


# ============================================================================
# File helpers
# ============================================================================

_file_sizes: dict[str, int] = {}


def is_file_stable(filepath: Path) -> bool:
    key = str(filepath)
    try:
        current_size = filepath.stat().st_size
    except OSError:
        _file_sizes.pop(key, None)
        return False
    if current_size == 0:
        return False
    prev_size = _file_sizes.get(key)
    _file_sizes[key] = current_size
    if prev_size is None:
        return False
    return prev_size == current_size


def processing_marker(audio_path: Path) -> Path:
    return audio_path.with_suffix(audio_path.suffix + ".processing")


def raw_md_path(audio_path: Path) -> Path:
    return audio_path.parent / f"{audio_path.stem}_raw.md"


def corrected_md_path(audio_path: Path) -> Path:
    return audio_path.parent / f"{audio_path.stem}_corrected.md"


def audio_files() -> list[Path]:
    result = []
    for entry in sorted(WATCH_DIR.iterdir()):
        if entry.is_file() and entry.suffix.lower() in AUDIO_EXTENSIONS:
            result.append(entry)
    return result


def cleanup_stale_markers():
    for marker in WATCH_DIR.glob("*.processing"):
        log(f"Cleanup: removing stale marker {marker.name}")
        marker.unlink()


# ============================================================================
# Two-phase scan
# ============================================================================

def scan_and_process():
    """Two-phase processing to avoid MLX memory contention.

    Phase 1: Transcribe all pending audio files (ASR model loaded).
    Phase 2: Unload ASR, then LLM-correct + move to done.
    """
    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)

    files = audio_files()
    if not files:
        return

    # --- Phase 1: Transcribe all pending files ---
    transcribed_any = False
    for audio_path in files:
        if processing_marker(audio_path).exists():
            continue
        if raw_md_path(audio_path).exists():
            continue
        if not is_file_stable(audio_path):
            log(f"Waiting for file to stabilize: {audio_path.name}")
            continue

        stem = audio_path.stem
        chunks_dir = WATCH_DIR / f".chunks_{stem}"
        marker = processing_marker(audio_path)
        marker.write_text(str(os.getpid()), encoding='utf-8')
        try:
            log(f"[ASR] Transcribing: {audio_path.name}")
            start = time.time()
            raw_text = transcribe_file(audio_path, chunks_dir)
            raw_out = raw_md_path(audio_path)
            raw_out.write_text(raw_text, encoding='utf-8')
            log(f"[ASR] Wrote {raw_out.name} ({time.time()-start:.1f}s)")
            transcribed_any = True
        except Exception as e:
            log(f"[ASR] ERROR {audio_path.name}: {e}")
        finally:
            if marker.exists():
                marker.unlink()
            # Move chunks to done/ for reference
            if chunks_dir.exists():
                dest_chunks = DONE_DIR / f"chunks_{stem}"
                if dest_chunks.exists():
                    shutil.rmtree(dest_chunks, ignore_errors=True)
                try:
                    shutil.move(str(chunks_dir), str(dest_chunks))
                except Exception:
                    shutil.rmtree(chunks_dir, ignore_errors=True)

    # --- Phase 2: Correct all files with _raw.md but no _corrected.md ---
    pending_correction = []
    for audio_path in audio_files():
        if raw_md_path(audio_path).exists() and not corrected_md_path(audio_path).exists():
            pending_correction.append(audio_path)

    if not pending_correction:
        # Move any fully-done files
        for audio_path in audio_files():
            if raw_md_path(audio_path).exists() and corrected_md_path(audio_path).exists():
                _move_to_done(audio_path)
        return

    # Unload ASR model before LLM work
    if transcribed_any:
        unload_asr_model()
        time.sleep(2)

    for audio_path in pending_correction:
        raw_text = raw_md_path(audio_path).read_text(encoding='utf-8')
        corrected_out = corrected_md_path(audio_path)

        marker = processing_marker(audio_path)
        marker.write_text(str(os.getpid()), encoding='utf-8')
        try:
            log(f"[LLM] Correcting: {audio_path.name}")
            corrected = correct_text(raw_text)
            corrected_out.write_text(corrected, encoding='utf-8')
            log(f"[LLM] Wrote {corrected_out.name}")
        except Exception as e:
            log(f"[LLM] ERROR {audio_path.name}: {e}")
        finally:
            if marker.exists():
                marker.unlink()

        _move_to_done(audio_path)


def _move_to_done(audio_path: Path):
    # Keep files in staging area for 7 days before moving to done/
    now = time.time()
    if now - audio_path.stat().st_mtime < 7 * 24 * 3600:
        return

    DONE_DIR.mkdir(parents=True, exist_ok=True)
    dest = DONE_DIR / audio_path.name
    if dest.exists():
        base, ext = audio_path.stem, audio_path.suffix
        i = 1
        while dest.exists():
            dest = DONE_DIR / f"{base}_{i}{ext}"
            i += 1
    try:
        shutil.move(str(audio_path), str(dest))
        log(f"Moved to done/{dest.name}")
    except Exception as e:
        log(f"Move failed for {audio_path.name}: {e}")


# ============================================================================
# Main loop
# ============================================================================

def main():
    log("Transcribe daemon starting (two-phase, chunked)")
    log(f"Watching: {WATCH_DIR}")
    log(f"ASR: {ASR_API} ({ASR_MODEL})")
    log(f"LLM: {LLM_API} ({LLM_MODEL})")
    log(f"Chunk: {CHUNK_MINUTES}min, Workers: {MAX_WORKERS}")
    log(f"Poll interval: {POLL_INTERVAL}s")

    WATCH_DIR.mkdir(parents=True, exist_ok=True)
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_stale_markers()

    while True:
        try:
            scan_and_process()
        except Exception as e:
            log(f"Scan error: {e}")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
