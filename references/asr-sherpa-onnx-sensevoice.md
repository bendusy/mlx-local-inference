# SenseVoice via sherpa-onnx — Decision Log

Research and benchmark notes for the local fast-ASR component used by the asr-router.

## What We Use

**SenseVoice 2024-07-17 int8** (`iic/SenseVoiceSmall`, 228 MB)

- Languages: zh, en, yue, ja, ko
- RTF: ~0.010-0.013 (60-100ms decode for 5-7s audio)
- Exposes `lang`, `emotion`, `event` metadata per segment — enables auto-routing without a separate LID model

## Why Not the 2025-09-09 Build

The `WSYue-ASR/sensevoice_small_yue` (2025-09-09) checkpoint is a **Cantonese specialist**, not a general multilingual upgrade. Its LID head is over-fitted to Cantonese: it labels en/ja/ko audio as `<|yue|>`, breaking the asr-router's auto-routing logic. Do not use for general-purpose transcription. Stick with 2024-07-17 for production.

If a future build ships a genuinely multilingual upgrade, re-run the 5-wav benchmark below before switching.

## Why Not FireRedASR2

Tested `fire-red-asr2-ctc-zh_en-int8` (2026-02-25, 496 MB) against the same 5 test wavs:

| Metric | SenseVoice 2024-07-17 | FireRedASR2 |
|--------|----------------------|-------------|
| Decode time (avg 5-7s clip) | ~60-90ms | ~600ms |
| Speed ratio | 1× | ~10× slower |
| English ITN | Punctuated, mixed-case | ALL CAPS, no punctuation |
| Memory | 228 MB | 496 MB |

FireRedASR2's claimed SOTA numbers come from AISHELL / WenetSpeech long-form Mandarin benchmarks. Short mixed-language clips cannot reproduce that advantage. Keep as a future option if the workload shifts to long-form Mandarin dialects; not in production.

## Auto-Routing via SenseVoice Tags

SenseVoice exposes per-utterance tags in the transcript text:
- Language: `<|zh|>` `<|yue|>` `<|en|>` `<|ja|>` `<|ko|>`
- Emotion: `<|NEUTRAL|>` `<|HAPPY|>` `<|SAD|>` `<|ANGRY|>`
- Event: `<|Speech|>` `<|BGM|>` `<|Applause|>` `<|Laughter|>`

The asr-router reads `r.lang` and `r.event` to decide whether to upgrade to oMLX Qwen3-ASR (see `routing.yaml`). No separate language-ID model needed.

## Install

```bash
bash asr/scripts/install_models.sh
```

Downloads (in order):
1. SenseVoice 2024-07-17 int8 — fast ASR
2. silero-vad — VAD pre-filter
3. pyannote-segmentation-3-0 — speaker segmentation
4. 3D-Speaker eres2net — speaker embedding for diarization

## Benchmark — 5-Wav Suite (SenseVoice bundled test_wavs)

| File | Duration | Decode | RTF |
|------|----------|--------|-----|
| zh.wav | 5.59s | 60-100ms | ~0.011 |
| en.wav | 7.15s | 70-90ms | ~0.010 |
| yue.wav | 5.15s | 50-90ms | ~0.011 |
| ja.wav | 7.20s | 90-100ms | ~0.013 |
| ko.wav | 4.61s | 40-50ms | ~0.010 |

Hardware: Apple M4. Results vary by load; RTF < 0.015 is consistent.

## References

- sherpa-onnx: https://github.com/k2-fsa/sherpa-onnx
- SenseVoice: https://github.com/FunAudioLLM/SenseVoice
- SenseVoice sherpa-onnx guide: https://k2-fsa.github.io/sherpa/onnx/sense-voice/index.html
- FireRedASR2 paper: https://arxiv.org/abs/2603.10420
