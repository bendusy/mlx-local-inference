<p align="center">
  <h1 align="center">🧠 MLX 本地推理全家桶</h1>
  <p align="center">
    让你的 Apple Silicon Mac 学会听、看、读、说、想 — 完全本地化。
  </p>
  <p align="center">
    <a href="https://clawhub.ai/skills/mlx-local-inference"><img src="https://img.shields.io/badge/ClawHub-mlx--local--inference-FF5A36?style=flat-square" alt="ClawHub"></a>
    <a href="#"><img src="https://img.shields.io/badge/平台-macOS%20Apple%20Silicon-000?style=flat-square&logo=apple&logoColor=white" alt="Platform"></a>
    <a href="#"><img src="https://img.shields.io/badge/运行时-MLX--VLM-blue?style=flat-square" alt="MLX-VLM"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/许可证-MIT-green?style=flat-square" alt="License"></a>
  </p>
  <p align="center">
    <a href="README.md">English</a> · <b>中文</b>
  </p>
</p>

---

## 一句话安装

```bash
clawhub install mlx-local-inference
```

或直接克隆：

```bash
git clone https://github.com/bendusy/mlx-local-inference.git
```

## 为什么做这个

你的 M 系列 Mac 有强大的 Neural Engine 和统一内存，但大多数 AI 工作流仍然把每个请求发到云端。**MLX Local Inference Stack** 把你的 Mac 变成一台自给自足的 AI 工作站，专为**内存效率**设计，**16 GB 机器也能流畅运行**。

## 内存占用方案

| 方案 | 空闲内存占用 | 常驻模型 |
|:-----|:------------|:---------|
| **16 GB** | ~3 GB | Embedding (0.6B) + ASR (1.7B) |
| **32 GB** | ~3 GB | 同上 — LLM/VLM 按需加载 |

**核心原则：** 不调用就不加载。模型首次使用时从缓存加载，空闲后自动卸载。常驻的只有轻量级 API 服务本身。

## 能力一览

| 能力 | 模型 | 内存 | 加载策略 |
|:-----|:-----|:-----|:---------|
| 📐 **向量化** | Qwen3-Embedding-0.6B | ~1 GB | **常驻加载** |
| 👂 **语音识别** | Qwen3-ASR-1.7B | ~1.5 GB | **常驻加载** |
| 🧠 **推理/对话** | Qwen3.5-35B-A3B (32GB) / Qwen3-14B (16GB) | 20 GB / 9 GB | **按需加载** |
| 👁️ **OCR** | PaddleOCR-VL-1.5 | ~3.3 GB | **按需加载** |
| 🗣️ **语音合成** | Qwen3-TTS-1.7B | ~2 GB | **按需加载（默认不启用）** |

## 自动下载缺失模型

首次调用时，服务器会自动检测并下载缺失的模型：

```
[mlx-server] 未找到模型: mlx-community/Qwen3-ASR-1.7B-8bit
[mlx-server] 正在下载... (1.7 GB，快速网络约 2 分钟)
[mlx-server] 下载完成，正在加载模型...
```

也可以提前一次性下载所有默认模型：

```bash
python ~/.mlx-server/download_models.py
```

或单独下载：

```bash
huggingface-cli download mlx-community/Qwen3-ASR-1.7B-8bit
huggingface-cli download mlx-community/qwen3-embedding-0.6b-4bit
```

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
          │  常驻服务   │  │  常驻服务  │  │  按需调用   │
          │            │  │           │  │            │
          │ · Embed ✅  │  │ · ASR ✅  │  │ · OCR      │
          │ · LLM/VLM  │  │ · TTS     │  │            │
          │   (按需)    │  │  (按需)   │  │            │
          └────────────┘  └───────────┘  └────────────┘
```

✅ = 启动时常驻 | 其余 = 首次调用时加载，空闲后自动卸载

## 使用示例

### 📐 向量化 — 文本嵌入（常驻）

```bash
curl http://localhost:8787/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-0.6b", "input": "你好世界"}'
```

### 👂 语音识别（常驻）

```bash
# 粤语 / 普通话 / 中英混合
curl http://localhost:8788/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=mlx-community/Qwen3-ASR-1.7B-8bit \
  -F language=zh
```

支持格式：`wav`、`mp3`、`m4a`、`flac`、`ogg`、`webm`

### 🧠 推理/对话 — LLM / 视觉语言（按需，mlx-vlm）

```bash
# 纯文本
curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-35b", "messages": [{"role": "user", "content": "你好"}]}'

# 图文混合（视觉语言）
curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "text", "text": "这张图片里有什么？"}
      ]
    }]
  }'
```

> **16 GB 提示：** 使用 `qwen3-14b` 替代 `qwen3.5-35b`。14B 模型约占 9 GB，与 Embedding + ASR 共存没有问题。

### 👁️ OCR — 图像文字识别（按需）

```bash
python -m mlx_vlm.generate \
  --model mlx-community/PaddleOCR-VL-1.5-6bit \
  --image document.jpg --prompt "OCR:" --max-tokens 512 --temp 0.0
```

### 🗣️ 语音合成（按需，默认不加载）

```bash
curl http://localhost:8788/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-TTS", "input": "你好世界"}' \
  -o speech.wav
```

TTS 首次调用时加载，空闲超时后自动卸载（默认 5 分钟）。

### 📝 自动转录流水线

把音频文件丢进 `~/transcribe/`，守护进程自动处理：

1. Qwen3-ASR 转录 → `文件名_raw.md`
2. LLM 纠错、补标点 → `文件名_corrected.md`
3. 归档至 `~/transcribe/done/`

## 按内存选模型

### 16 GB Mac

| 用途 | 模型 | 内存 |
|:-----|:-----|:-----|
| 向量化 | `qwen3-embedding-0.6b` | ~1 GB |
| 语音识别 | `Qwen3-ASR-1.7B-8bit` | ~1.5 GB |
| LLM（按需） | `Qwen3-14B-4bit` | ~9 GB |
| OCR（按需） | `PaddleOCR-VL-1.5-6bit` | ~3.3 GB |
| TTS（可选） | `Qwen3-TTS-1.7B-8bit` | ~2 GB |

> ⚠️ 16 GB 机器上，避免同时运行 LLM + OCR。转录守护进程会自动在两个阶段之间卸载模型。

### 32 GB Mac

| 用途 | 模型 | 内存 |
|:-----|:-----|:-----|
| 向量化 | `qwen3-embedding-0.6b` | ~1 GB |
| 语音识别 | `Qwen3-ASR-1.7B-8bit` | ~1.5 GB |
| LLM/VLM（按需） | `Qwen3.5-35B-A3B-4bit` | ~20 GB |
| OCR（按需） | `PaddleOCR-VL-1.5-6bit` | ~3.3 GB |
| TTS（可选） | `Qwen3-TTS-1.7B-8bit` | ~2 GB |

## 服务管理

```bash
# 重启主服务（Embedding + 按需 LLM/VLM）
launchctl kickstart -k gui/$(id -u)/com.mlx-server

# 重启 ASR 服务（ASR 常驻 + TTS 按需）
launchctl kickstart -k gui/$(id -u)/com.mlx-audio-server

# 重启转录守护进程
launchctl kickstart gui/$(id -u)/com.mlx-transcribe-daemon

# 查看日志
tail -f ~/.mlx-server/logs/server.log
tail -f ~/.mlx-server/logs/mlx-audio-server.err.log
```

## 升级模型

```bash
# 1. 下载新模型
huggingface-cli download mlx-community/<新模型名>

# 2. 更新配置 (~/.mlx-server/config.yaml)
# 3. 重启服务
launchctl kickstart -k gui/$(id -u)/com.mlx-server
```

## 环境要求

- Apple Silicon Mac（M1 / M2 / M3 / M4）
- macOS 14+
- Python 3.10+
- **最低 16 GB 内存**（32 GB 推荐，可运行 35B 模型）
- mlx-vlm >= 0.3.12

## 目录结构

```
mlx-local-inference/
├── SKILL.md              # OpenClaw 技能定义
├── README.md             # English
├── README_CN.md          # 中文（本文件）
├── LICENSE
└── references/           # 各模型详细技术文档
```

## 贡献

欢迎提 Issue 和 PR。

## 许可证

[MIT](LICENSE)
