# 🧠 MLX 本地推理技术栈 (MLX Local Inference Stack)

让你的 Apple Silicon Mac 具备听、看、读、说、思考的能力 —— 全本地运行。

---

## 🚀 硬件模型选择建议 (统一内存)

根据你的 Mac 内存配置选择合适的梯队。本项目优先保障 **ASR (语音转文字)** 的实时性，以确保在飞书、Discord 等 IM 端的极致沟通体验。

### 🟢 32GB 内存梯队
- **思考/视觉 (Think/Vision):** `Qwen3.5-9B-MLX-4bit` (MoE 架构)
- **听觉 (ASR):** `Qwen3-ASR-1.7B-8bit` (**常驻保活**)
- **策略:** 利用 MoE 实现极速推理 (50 t/s)，同时保持 ASR 常驻以实现即时语音沟通。

### 🟡 16GB 内存梯队
- **思考:** `Gemma-3-12B-it-4bit` 或 `Qwen3-14B-4bit`
- **听觉 (ASR):** `Qwen3-ASR-1.7B-8bit` (**常驻保活**)
- **策略:** 性能均衡。优先保证 ASR 驻留内存，LLM 采用按需加载。

### ⚪ 8GB 内存梯队
- **思考:** `Qwen3-7B-4bit`
- **听觉 (ASR):** `Qwen3-ASR-1.7B-4bit` (按需加载)

---

## 🛠️ 极简执行 (通过 `uv`)

为了确保适用性并避免依赖混乱，建议所有组件均通过 `uv` 运行。

### 👂 听觉 — 即时 ASR (高优先级)
针对飞书/Discord 语音消息交互优化。
```bash
uv run --python 3.11 --with mlx-audio python -m mlx_audio.stt.generate \
  --model ~/models/Qwen3-ASR-1.7B-8bit \
  --audio "voice_message.ogg" \
  --output-path /tmp/asr_result \
  --language zh
```

### 🧠 思考 — 本地 LLM
```bash
uv run --with mlx-lm python -m mlx_lm.generate \
  --model ~/models/Qwen3.5-9B-MLX-4bit \
  --prompt "请分析以下请求..."
```

---

## 🏗️ 架构设计

**混合路径：** oMLX 负责 LLM/VLM（高性能 API 模式），Python 库通过 `uv` 负责 Embedding/ASR。

- **oMLX (localhost:8000/v1):** 负责主推理引擎。
- **UV / Python:** 负责 ASR 和嵌入向量计算，轻量且解耦。

---

## 📌 为什么选择此方案？

你的 Mac 拥有强大的统一内存，但大多数 AI 工作流仍依赖云端。本项目将你的 Mac 转化为全自守、高私密的 AI 工作站，重点优化了 **IM 交互场景** 下的语音识别鲁棒性。

---
*Created by Linus Torvalds (AI CTO) @ OpenClaw*
