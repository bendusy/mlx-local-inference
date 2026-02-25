<p align="center">
  <h1 align="center">🧠 MLX 本地推理全家桶</h1>
  <p align="center">
    在 Apple Silicon Mac 上运行完整本地 AI 推理 — LLM · 语音识别 · 向量化 · OCR · 语音合成 · 自动转录
  </p>
  <p align="center">
    <a href="https://clawhub.ai/skills/mlx-local-inference"><img src="https://img.shields.io/badge/ClawHub-mlx--local--inference-FF5A36?style=flat-square" alt="ClawHub"></a>
    <a href="#"><img src="https://img.shields.io/badge/平台-macOS%20Apple%20Silicon-000?style=flat-square&logo=apple&logoColor=white" alt="Platform"></a>
    <a href="#"><img src="https://img.shields.io/badge/运行时-MLX-blue?style=flat-square" alt="MLX"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/许可证-MIT-green?style=flat-square" alt="License"></a>
  </p>
  <p align="center">
    <a href="README.md">English</a> · <b>中文</b>
  </p>
</p>

---

一个 [OpenClaw](https://github.com/openclaw/openclaw) 技能包，通过 [MLX](https://github.com/ml-explore/mlx) 在 Apple Silicon Mac 上提供完整的本地 AI 推理能力。不依赖云端、不产生 API 费用、数据完全本地化。

## 功能概览

| 能力 | 模型 | 端口 | 说明 |
|:-----|:-----|:-----|:-----|
| **LLM 对话** | Qwen3-14B, Gemma3-12B | 8787 | 流式输出、思维链推理 |
| **语音识别** | Qwen3-ASR, Whisper-v3-turbo | 8788 / 8787 | 粤语/普通话强项 + 99 种语言 |
| **文本向量化** | Qwen3-Embedding 0.6B / 4B | 8787 | RAG、语义搜索、文档索引 |
| **OCR** | PaddleOCR-VL-1.5 | CLI | 中英文场景文字、票据、文档 |
| **语音合成** | Qwen3-TTS-1.7B | 8788 / CLI | 支持自定义音色克隆 |
| **自动转录** | ASR + LLM 联合 | 守护进程 | 文件监听、自动转录 + 智能纠错 |

所有服务均提供 **OpenAI 兼容 API**，可直接使用 `openai` Python SDK、`curl` 或任何兼容客户端调用。

## 环境要求

- **硬件**：Apple Silicon Mac（M1 / M2 / M3 / M4）
- **系统**：macOS 14+
- **内存**：推荐 32GB+（多模型同时运行）
- **Python**：3.10+

## 安装

### 作为 OpenClaw Skill 安装

```bash
clawhub install mlx-local-inference
```

### 独立安装

```bash
# 克隆仓库
git clone https://github.com/bendusy/mlx-local-inference.git
cd mlx-local-inference

# 安装 Python 依赖
pip install mlx mlx-lm mlx-audio mlx-vlm openai
```

### 下载模型

模型在首次调用时自动下载，也可以预先拉取：

```bash
# LLM
huggingface-cli download Qwen/Qwen3-14B-MLX-4bit
huggingface-cli download mlx-community/gemma-3-text-12b-it-4bit

# 语音识别
huggingface-cli download mlx-community/Qwen3-ASR-1.7B-8bit
huggingface-cli download mlx-community/whisper-large-v3-turbo

# 向量化
huggingface-cli download mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ

# OCR
huggingface-cli download mlx-community/PaddleOCR-VL-1.5-6bit

# 语音合成（可选）
huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
```

## 使用示例

### LLM 对话

```bash
curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-14b",
    "messages": [{"role": "user", "content": "用一句话解释量子计算"}]
  }'
```

<details>
<summary>Python 示例</summary>

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8787/v1", api_key="unused")
response = client.chat.completions.create(
    model="qwen3-14b",
    messages=[{"role": "user", "content": "你好"}],
    temperature=0.7,
    max_tokens=2048,
)
print(response.choices[0].message.content)
```

</details>

> **提示：** Qwen3 会输出 `<think>...</think>` 思维链标签，按需过滤：
> ```python
> import re
> text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
> ```

### 语音识别

```bash
# Qwen3-ASR — 粤语/普通话首选
curl http://localhost:8788/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=mlx-community/Qwen3-ASR-1.7B-8bit \
  -F language=zh

# Whisper — 多语言（99 种）
curl http://localhost:8787/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=whisper-large-v3-turbo
```

支持格式：`wav`、`mp3`、`m4a`、`flac`、`ogg`、`webm`

长音频建议先切分为 10 分钟片段：

```bash
ffmpeg -y -ss 0 -t 600 -i long.wav -ar 16000 -ac 1 chunk_000.wav
```

### 文本向量化

```bash
# 单条
curl http://localhost:8787/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-0.6b", "input": "要向量化的文本"}'

# 批量
curl http://localhost:8787/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-4b", "input": ["文本一", "文本二"]}'
```

### OCR 文字识别

```bash
python -m mlx_vlm.generate \
  --model mlx-community/PaddleOCR-VL-1.5-6bit \
  --image photo.jpg \
  --prompt "OCR:" \
  --max-tokens 512 \
  --temp 0.0
```

> Prompt 必须为 `OCR:`，temperature 设 0 确保确定性输出。

### 语音合成

```bash
curl http://localhost:8788/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit","input":"你好世界"}' \
  -o speech.wav
```

### 自动转录守护进程

将音频文件放入 `~/transcribe/` 目录，守护进程自动处理：

1. **Qwen3-ASR 转录** → `文件名_raw.md`
2. **Qwen3-14B 智能校对** → `文件名_corrected.md`
3. **归档** → `~/transcribe/done/`

校对规则：同音字纠错、保留粤语用字（嘅/唔/咁/喺/冇/佢）、补全标点、去除语气词和重复。

## 架构

```
┌──────────────────────────────────────────────┐
│           Apple Silicon Mac (MLX)            │
├──────────────────┬───────────────────────────┤
│    端口 8787      │       端口 8788            │
│    (局域网可访问)  │       (仅本机)             │
│                  │                           │
│  · Qwen3-14B    │  · Qwen3-ASR              │
│  · Gemma3-12B   │  · Qwen3-TTS              │
│  · Whisper      │                           │
│  · Embedding    │                           │
│    0.6B / 4B    │                           │
├──────────────────┴───────────────────────────┤
│  OCR: PaddleOCR-VL           (CLI, 按需调用)  │
│  转录守护进程          (文件监听, ASR→LLM 校对) │
└──────────────────────────────────────────────┘
```

## 模型选型

### LLM

| 场景 | 推荐 |
|:-----|:-----|
| 中文 / 粤语 | `qwen3-14b` |
| 英文 / 代码 | `gemma-3-12b` |
| 深度推理 | `qwen3-14b`（think 模式） |
| 快速问答 | `gemma-3-12b` |

### 语音识别

| 场景 | 推荐 |
|:-----|:-----|
| 粤语 / 普通话 | Qwen3-ASR |
| 多语言（99 种） | Whisper |

### 向量化

| 场景 | 推荐 |
|:-----|:-----|
| 快速检索 / 低延迟 | `qwen3-embedding-0.6b` |
| 高精度语义匹配 | `qwen3-embedding-4b` |

## 服务管理

```bash
# 主服务（LLM + Whisper + Embedding）
launchctl kickstart -k gui/$(id -u)/com.mlx-server

# ASR + TTS 服务
launchctl kickstart -k gui/$(id -u)/com.mlx-audio-server

# 转录守护进程
launchctl kickstart gui/$(id -u)/com.mlx-transcribe-daemon
```

## 目录结构

```
mlx-local-inference/
├── SKILL.md              # OpenClaw 技能定义
├── README.md             # English
├── README_CN.md          # 中文说明（本文件）
├── LICENSE               # MIT
└── references/           # 各模型详细参考文档
    ├── asr-qwen3.md
    ├── asr-whisper.md
    ├── embedding-qwen3.md
    ├── llm-qwen3-14b.md
    ├── llm-gemma3-12b.md
    ├── llm-models-reference.md
    ├── ocr.md
    ├── transcribe-daemon.md
    └── tts-qwen3.md
```

## 贡献

欢迎提 Issue 和 PR。各模型的详细技术文档见 `references/` 目录。

## 许可证

[MIT](LICENSE)
