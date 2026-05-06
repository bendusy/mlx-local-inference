#!/usr/bin/env bash
set -euo pipefail
ROOT="$HOME/models/sherpa-onnx"
mkdir -p "$ROOT"
cd "$ROOT"

# 1. SenseVoice int8 (already present per prior verification — skip if exists)
SV="sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
if [ ! -d "$SV" ]; then
  echo "Downloading SenseVoice..."
  curl -L -o sv.tar.bz2 "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${SV}.tar.bz2"
  tar xjf sv.tar.bz2 && rm sv.tar.bz2
fi

# 2. silero-vad
mkdir -p silero-vad
if [ ! -f silero-vad/silero_vad.onnx ]; then
  echo "Downloading silero-vad..."
  curl -L -o silero-vad/silero_vad.onnx \
    "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
fi

# 3. Speaker diarization (pyannote segmentation)
SEG="sherpa-onnx-pyannote-segmentation-3-0"
if [ ! -d "$SEG" ]; then
  echo "Downloading pyannote segmentation..."
  curl -L -o seg.tar.bz2 "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/${SEG}.tar.bz2"
  tar xjf seg.tar.bz2 && rm seg.tar.bz2
fi

# 4. 3D-Speaker embedding
EMB="3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
if [ ! -f "$EMB" ]; then
  echo "Downloading speaker embedding..."
  curl -L -O "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/${EMB}"
fi

echo "All ASR models installed at $ROOT"
ls -lh "$ROOT"
