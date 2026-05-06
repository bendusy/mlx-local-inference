# Pipeline End-to-End Evaluation

**Date:** 2026-05-06 18:28:59
**Source:** a 2-minute slice (60-180s) of a bilingual business meeting recording (Chinese + Korean translation), 120.0s clip
**Ground truth:** segments fully inside the window from the manually-corrected diarization JSON (179 chars)

## CER (lower is better)

| Stage | CER |
|---|---|
| SenseVoice raw | 0.3208 |
| gemma-4 reviewed | 0.2264 |
| Relative improvement | +29.4% |

## Artifacts

- `_raw.json`
- `input.wav`
- `input_diff.md`
- `input_gemma4.md`
- `input_sensevoice.md`
- `input_summary.md`

## Notes

CER computed over normalized text:
1. NFKC normalization
2. Strip markdown block-quote lines (`> 注：…`), headings, speaker-mapping bullets
3. Strip `[Speaker_N ts-ts]` timestamp labels
4. Strip `**bold**` speaker headers
5. Drop entire low-confidence lines marked with `⚠️`
6. Retain only CJK Unified Ideographs and CJK punctuation (U+4E00–U+9FFF,
   U+3400–U+4DBF, U+F900–U+FAFF, U+3000–U+303F, U+FF00–U+FFEF) — Korean and
   kana characters are excluded so that Korean speech (transcribed as Korean by
   both ASR systems) doesn't inflate CER against the Chinese-only GT.

Ground truth covers 179 characters (7 Chinese-source segments) from the
manually-corrected `.diarize.json` file.  Five segments prefixed with `(韩)`
are Chinese-language *translations* of Korean source audio; both ASR systems
transcribe the raw Korean, so those segments are excluded from the GT reference.

The pipeline ran on a 120.0-second slice — long enough to test
multi-segment SenseVoice + gemma-4 batched review with the per-job
glossary applied. Larger samples and full-meeting CER trends remain
future work.
