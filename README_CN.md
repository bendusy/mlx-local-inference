<p align="center">
  <h1 align="center">🧠 MLX 本地推理全家桶</h1>
  <p align="center">
    让你的 Apple Silicon Mac 学会听、看、读、说、想 — 完全本地化。
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

## 为什么做这个

你的 M 系列 Mac 有强大的 Neural Engine 和统一内存，但大多数 AI 工作流仍然把每个请求发到云端 — 慢、贵、还有隐私顾虑。

**MLX Local Inference Stack** 把你的 Mac 变成一台自给自足的 AI 工作站。我们逐一测试并精选了各模态下表现最好的 MLX 模型 — 语音识别、文字生成、OCR、语音合成、向量嵌入 — 打包成一套开箱即用的本地推理方案。装上之后，你的 Mac 就拥有了**听、看、读、说、想**的完整能力，全程离线。

搭配 [OpenClaw](https://github.com/openclaw/openclaw) 等 AI 智能体使用效果尤佳：语音转录、文档识别、文本纠错、语义搜索、语音播报……这些以前依赖云端 API 的环节，现在全部在本地完成，交互更快、成本为零、数据不出机。

## 你的 Mac 获得了什么

| 能力 | 做什么 | 精选模型 |
|:-----|:-------|:---------|
| 👂 **听** | 转录 99 种语言的语音，粤语/普通话表现尤强 | Qwen3-ASR-1.7B · Whisper-v3-turbo |
| 👁️ **看** | 从照片、截图、票据、文档中提取文字 | PaddleOCR-VL-1.5 |
| 🧠 **想** | 对话、推理、写代码、翻译、总结 | Qwen3-14B · Gemma3-12B |
| 🗣️ **说** | 生成自然语音，支持自定义音色克隆 | Qwen3-TTS-1.7B |
| 📐 **理解** | 文本向量化，语义搜索、RAG、文档索引 | Qwen3-Embedding 0.6B · 4B |
| 📝 **转录** | 丢入音频文件，自动获得校对过的文稿 | ASR + LLM 校对流水线 |

每个模型都经过实测筛选，在 Apple Silicon 上取得质量、速度、内存占用的最佳平衡。这不是六个拼凑的工具，而是一套**完整的本地 AI 运行时**。

## 整体架构

```
                        ┌─────────────────┐
                        │    你的智能体    │
                        │ (OpenClaw 等)   │
                        └────────┬────────┘
                                 │ OpenAI 兼容 API
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
          ┌────────────┐  ┌───────────┐  ┌────────────┐
          │  端口 8787  │  │ 端口 8788 │  │   CLI      │
          │            │  │           │  │            │
          │ · LLM      │  │ · ASR     │  │ · OCR      │
          │ · Whisper  │  │ · TTS     │  │            │
          │ · Embed    │  │           │  │            │
          └────────────┘  └───────────┘  └────────────┘
                 │               │               │
                 └───────────────┴───────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │   Apple Silicon (MLX)   │
                    │     统一内存 · GPU       │
                    └─────────────────────────┘
```

所有服务提供 **OpenAI 兼容 API**，任何支持 OpenAI 协议的工具、SDK、智能体都能直接调用，无需适配器。

## 环境要求

- Apple Silicon Mac（M1 / M2 / M3 / M4）
- macOS 14+
- Python 3.10+
- 推荐 32GB+ 内存（16GB 可运行，但无法同时加载多模型）

## 安装

### 作为 OpenClaw Skill

```bash
clawhub install mlx-local-inference
```

### 直接克隆

```bash
git clone https://github.com/bendusy/mlx-local-inference.git
cd mlx-local-inference
pip install mlx mlx-lm mlx-audio mlx-vlm openai
```

模型在首次使用时自动下载。如需预先拉取：

<details>
<summary>预下载全部模型</summary>

```bash
huggingface-cli download Qwen/Qwen3-14B-MLX-4bit
huggingface-cli download mlx-community/gemma-3-text-12b-it-4bit
huggingface-cli download mlx-community/Qwen3-ASR-1.7B-8bit
huggingface-cli download mlx-community/whisper-large-v3-turbo
huggingface-cli download mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
huggingface-cli download mlx-community/PaddleOCR-VL-1.5-6bit
huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
```

</details>

## 使用示例

### 🧠 想 — LLM 对话

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
r = client.chat.completions.create(
    model="qwen3-14b",
    messages=[{"role": "user", "content": "你好"}],
)
print(r.choices[0].message.content)
```

</details>

内置两个 LLM：**Qwen3-14B**（中文最强、自带思维链推理）和 **Gemma3-12B**（英文/代码快速响应），按任务选择即可。

### 👂 听 — 语音识别

```bash
# 粤语 / 普通话 → Qwen3-ASR
curl http://localhost:8788/v1/audio/transcriptions \
  -F file=@audio.wav -F model=mlx-community/Qwen3-ASR-1.7B-8bit -F language=zh

# 99 种语言 → Whisper
curl http://localhost:8787/v1/audio/transcriptions \
  -F file=@audio.wav -F model=whisper-large-v3-turbo
```

支持格式：`wav`、`mp3`、`m4a`、`flac`、`ogg`、`webm`

### 👁️ 看 — OCR 文字识别

```bash
python -m mlx_vlm.generate \
  --model mlx-community/PaddleOCR-VL-1.5-6bit \
  --image document.jpg --prompt "OCR:" --max-tokens 512 --temp 0.0
```

### 🗣️ 说 — 语音合成

```bash
curl http://localhost:8788/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit","input":"你好世界"}' \
  -o speech.wav
```

### 📐 理解 — 文本向量化

```bash
curl http://localhost:8787/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-0.6b", "input": ["文档一", "文档二"]}'
```

两个规格：**0.6B** 快速检索，**4B** 高精度语义匹配。

### 📝 转录 — 自动流水线

把音频文件丢进 `~/transcribe/`，其余的交给守护进程：

1. Qwen3-ASR 转录 → `文件名_raw.md`
2. Qwen3-14B 纠错、补标点、保留粤语用字 → `文件名_corrected.md`
3. 归档至 `~/transcribe/done/`

不用敲命令，丢文件就行。

## 模型选型依据

每个模型都经过实机测试，在质量与效率之间取最佳平衡：

| 模态 | 模型 | 为什么选它 |
|:-----|:-----|:-----------|
| LLM（中文） | Qwen3-14B 4bit | 同尺寸最强中英双语，内置思维链 |
| LLM（英文） | Gemma3-12B 4bit | 响应快、代码生成强、内存占用低 |
| ASR（中文） | Qwen3-ASR-1.7B 8bit | 粤语/普通话准确率最高，按需加载 |
| ASR（多语言） | Whisper-v3-turbo | 99 种语言，常驻内存，久经考验 |
| Embedding（快） | Qwen3-Embedding-0.6B 4bit | 低延迟，日常检索够用 |
| Embedding（精） | Qwen3-Embedding-4B 4bit | 高精度语义匹配 |
| OCR | PaddleOCR-VL-1.5 6bit | ~185 token/s，3.3 GB，精度速度比最优 |
| TTS | Qwen3-TTS-1.7B 8bit | 自定义音色克隆，约 2 GB |

## 服务管理

```bash
# 主服务（LLM + Whisper + Embedding）
launchctl kickstart -k gui/$(id -u)/com.mlx-server

# ASR + TTS 服务
launchctl kickstart -k gui/$(id -u)/com.mlx-audio-server

# 自动转录守护进程
launchctl kickstart gui/$(id -u)/com.mlx-transcribe-daemon
```

## 目录结构

```
mlx-local-inference/
├── SKILL.md              # OpenClaw 技能定义
├── README.md             # English
├── README_CN.md          # 中文（本文件）
├── LICENSE
└── references/           # 各模型详细技术文档
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
