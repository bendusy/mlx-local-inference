---
name: mlx-local-inference
description: >
  Full local AI inference stack on Apple Silicon Macs via MLX.
  Default framework: mlx-vlm (unified vision-language + text).
  Memory-efficient design: only Embedding (0.6B) and ASR (1.7B) are always loaded (~3 GB idle).
  LLM/VLM, OCR loaded on-demand via lazy proxy. TTS not loaded by default.
  Missing models are auto-downloaded on first call.
  Works on 16 GB Macs. 32 GB recommended for 35B models.
  Use when the user needs local AI: text generation, vision-language, speech recognition,
  embeddings, OCR, TTS — without cloud API calls.
metadata: { "openclaw": { "os": ["darwin"], "requires": { "anyBins": ["python3"] } } }
---

# MLX Local Inference Stack

Apple Silicon Mac 上的全本地 AI 推理栈。**mlx-vlm** 统一处理所有视觉语言和文本生成任务。专为 **16 GB 机器**设计——只有必要模型常驻内存。

## 内存策略

| 常驻 | 按需加载 | 默认不加载 |
|:-----|:---------|:-----------|
| Embedding 0.6B (~1 GB) | LLM/VLM (9–20 GB) | TTS (~2 GB) |
| ASR 1.7B (~1.5 GB) | OCR (~3.3 GB) | |
| **空闲总计: ~3 GB** | | |

**核心原则：** 不调用就不加载。首次请求时自动加载（需等待几秒到几十秒），空闲超时后自动卸载释放内存。

## 按需加载原理

服务器采用 **懒加载代理（lazy proxy）** 模式：

1. 启动时注册所有模型配置，但**不占用内存**
2. 首次请求到来时，代理透明地加载模型——调用方只需等待
3. **空闲 watchdog** 监控每个模型，超过配置时间无请求则自动卸载

```
请求到来
   │
   ▼
模型已加载？──否──▶ 立即加载（调用方等待）──▶ 处理请求
   │是                                            │
   └─────────────────────────────────────────────┘
                                                  │
                                         [空闲超时]
                                                  │
                                             卸载 ✓
```

对调用方完全透明，无需任何客户端改动。

## 自动下载

首次调用时自动检测并下载缺失模型：

```
[mlx-server] Model not found: mlx-community/Qwen3-ASR-1.7B-8bit
[mlx-server] Downloading... (1.7 GB, ~2 min on fast connection)
[mlx-server] Download complete. Loading model...
```

一次性预下载所有默认模型：

```bash
python ~/.mlx-server/download_models.py
```

或下载指定模型：

```bash
huggingface-cli download mlx-community/Qwen3-ASR-1.7B-8bit
huggingface-cli download mlx-community/qwen3-embedding-0.6b-4bit
```

## 架构

```
                        ┌─────────────────┐
                        │   你的 Agent    │
                        │  (OpenClaw 等)  │
                        └────────┬────────┘
                                 │ OpenAI 兼容 API
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
          ┌────────────┐  ┌───────────┐  ┌────────────┐
          │  端口 8787 │  │ 端口 8788 │  │    CLI     │
          │  常驻服务  │  │  常驻服务 │  │  按需调用  │
          │            │  │           │  │            │
          │ · Embed ✅  │  │ · ASR ✅  │  │ · OCR      │
          │ · LLM/VLM  │  │ · TTS     │  │            │
          │   (懒加载) │  │  (懒加载) │  │            │
          └────────────┘  └───────────┘  └────────────┘
```

✅ = 启动时加载 | 懒加载 = 首次请求时加载，空闲后卸载

## 使用方法

### 1. Embedding（常驻，~1 GB）

```bash
curl http://localhost:8787/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-embedding-0.6b", "input": "要向量化的文本"}'
```

### 2. ASR 语音识别（常驻，~1.5 GB）

```bash
# 中文 / 粤语 / 混合
curl http://localhost:8788/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=mlx-community/Qwen3-ASR-1.7B-8bit \
  -F language=zh
```

支持格式：`wav`, `mp3`, `m4a`, `flac`, `ogg`, `webm`

### 3. LLM / 视觉语言（按需，via mlx-vlm）

#### 按内存选模型

| 内存 | 模型 | 占用 |
|------|------|------|
| 16 GB | `qwen3-14b` | ~9 GB |
| 32 GB | `qwen3.5-35b` | ~20 GB |

```bash
# 文本
curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-35b", "messages": [{"role": "user", "content": "你好"}]}'

# 视觉（图片+文本）
curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-35b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "text", "text": "图片里有什么？"}
      ]
    }]
  }'
```

> **16 GB 提示：** 用 `qwen3-14b` 代替 `qwen3.5-35b`，占用 ~9 GB，留有足够余量给 Embedding + ASR。

### 4. OCR（按需，~3.3 GB）

```bash
python -m mlx_vlm.generate \
  --model mlx-community/PaddleOCR-VL-1.5-6bit \
  --image document.jpg \
  --prompt "OCR:" \
  --max-tokens 512 --temp 0.0
```

### 5. TTS 语音合成（按需，默认不加载）

```bash
curl http://localhost:8788/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-TTS", "input": "你好世界"}' \
  -o speech.wav
```

首次调用时加载，空闲 5 分钟后自动卸载。

### 6. 转录 Daemon — 自动流水线

将音频文件拖入 `~/transcribe/`，自动处理：

1. ASR 转录 → `filename_raw.md`
2. LLM 纠错加标点 → `filename_corrected.md`
3. 归档到 `~/transcribe/done/`

> 16 GB 机器上：daemon 会在加载 LLM 前先卸载 ASR，避免内存冲突。

## Admin API

服务器提供轻量级管理接口，用于手动控制模型加载状态：

```bash
# 列出所有模型及加载状态
curl http://localhost:8787/v1/admin/models

# 手动卸载模型
curl -X POST http://localhost:8787/v1/admin/models/qwen3.5-35b/unload

# 手动加载模型
curl -X POST http://localhost:8787/v1/admin/models/qwen3.5-35b/load

# 查看队列状态
curl http://localhost:8787/v1/admin/models/qwen3.5-35b/stats
```

## 服务管理

```bash
# 重启主服务（Embedding + 按需 LLM/VLM）
launchctl kickstart -k gui/$(id -u)/com.mlx-server

# 重启 ASR 服务（ASR 常驻 + TTS 按需）
launchctl kickstart -k gui/$(id -u)/com.mlx-audio-server

# 重启转录 daemon
launchctl kickstart gui/$(id -u)/com.mlx-transcribe-daemon

# 查看日志
tail -f ~/.mlx-server/logs/server.log
tail -f ~/.mlx-server/logs/mlx-audio-server.err.log
```

## 环境要求

- Apple Silicon Mac（M1 / M2 / M3 / M4）
- macOS 14+
- Python 3.10+
- **最低 16 GB 内存**（35B 模型推荐 32 GB）
- mlx-vlm >= 0.3.12

## License

[MIT](LICENSE)
