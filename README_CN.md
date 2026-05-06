<p align="center">
  <h1 align="center">MLX 本地推理技术栈</h1>
  <p align="center">
    oMLX 网关 + asr-router 辅助服务 —— 在 Apple Silicon 上全本地运行 AI。
  </p>
  <p align="center">
    <a href="https://clawhub.ai/skills/mlx-local-inference"><img src="https://img.shields.io/badge/ClawHub-mlx--local--inference-FF5A36?style=flat-square" alt="ClawHub"></a>
    <a href="#"><img src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-000?style=flat-square&logo=apple&logoColor=white" alt="Platform"></a>
    <a href="#"><img src="https://img.shields.io/badge/gateway-oMLX-blue?style=flat-square" alt="oMLX"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"></a>
  </p>
  <p align="center">
    English · <a href="README_CN.md"><b>中文</b></a>
  </p>
</p>

---

本机所有本地 AI 推理由两个进程提供：**oMLX**（GUI 应用）通过连续批处理和 SSD 模型缓存处理 LLM/VLM/嵌入/OCR/高质量 ASR 推理；**asr-router** 是一个 FastAPI 辅助服务，封装了 sherpa-onnx SenseVoice 用于百毫秒内的 IM 转写，并提供异步四阶段会议流水线（VAD+说话人分割 → SenseVoice → gemma-4 上下文校对 → 渲染 5 份命名产物）。两者均在独立端口暴露 OpenAI 兼容的 REST API。

## 架构

```
┌────────────────────────────────────────────────────────────────────┐
│                  Apple Silicon (Mac M 系列)                        │
│                                                                    │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐│
│  │  oMLX 网关      :18080/v1    │  │  asr-router    :18081/v1     ││
│  │  密钥: sk-mlx                │  │  密钥: sk-mlx                ││
│  │                              │  │                              ││
│  │  • Qwen3.5-35B-A3B  (LLM)    │  │  IM 模式 (Whisper-API):      ││
│  │  • gemma-4-26b      (LLM)    │  │   sherpa-onnx SenseVoice     ││
│  │  • Qwen3.5-9B       (LLM)    │  │   ↔ oMLX Qwen3-ASR (自动)    ││
│  │  • supergemma4-26b  (VLM)    │  │                              ││
│  │  • PaddleOCR-VL-1.5 (OCR)    │  │  会议模式 (异步任务):        ││
│  │  • Qwen3-Embedding-0.6B      │  │   VAD+说话人分割 →           ││
│  │  • Qwen3-ASR-1.7B   (ASR-Q)  │  │   SenseVoice →               ││
│  │                              │  │   gemma-4 校对 (oMLX) →      ││
│  │  连续批处理，SSD 缓存        │  │   渲染 5 份产物              ││
│  │  (由 oMLX.app 管理)          │  │                              ││
│  └──────────────────────────────┘  └──────────────────────────────┘│
└────────────────────────────────────────────────────────────────────┘
```

## 接入端点

| 服务 | URL | API 密钥 |
|------|-----|----------|
| oMLX 网关 | `http://localhost:18080/v1` | `sk-mlx` |
| asr-router | `http://localhost:18081/v1` | `sk-mlx` |

oMLX 配置由 GUI 应用管理；权威配置文件位于 `~/.omlx/settings.json`。

## 当前模型清单

| 能力 | 模型 ID | 大小 | 宿主 |
|-----|---------|------|------|
| LLM（旗舰） | `Qwen3.5-35B-A3B-4bit` | ~18 GB | oMLX |
| LLM（快速，OpenClaw 默认） | `gemma-4-26b-a4b-it-4bit` | ~14 GB | oMLX |
| LLM（小型） | `Qwen3.5-9B-MLX-4bit` | ~5.8 GB | oMLX |
| VLM | `supergemma4-26b-abliterated-multimodal-mlx-4bit` | ~14 GB | oMLX |
| OCR（VLM） | `PaddleOCR-VL-1.5-6bit` | ~3.3 GB | oMLX |
| 嵌入向量 | `Qwen3-Embedding-0.6B-4bit-DWQ` | ~1 GB | oMLX |
| ASR（高质量） | `Qwen3-ASR-1.7B-8bit` | ~1.5 GB | oMLX |
| ASR（快速，本地） | sherpa-onnx SenseVoice int8 | 228 MB | asr-router |

SenseVoice 模型路径：`~/models/sherpa-onnx/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/`  
支持语言：中文 / 英文 / 粤语 / 日文 / 韩文。解码延迟：5–7 秒音频约 60–90 ms（RTF ≈ 0.01）。

## 快速上手

### LLM 文本生成

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:18080/v1", api_key="sk-mlx")
resp = client.chat.completions.create(
    model="Qwen3.5-35B-A3B-4bit",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)
```

### VLM 图像理解

```python
import base64
from openai import OpenAI
client = OpenAI(base_url="http://localhost:18080/v1", api_key="sk-mlx")
img_b64 = base64.b64encode(open("photo.jpg", "rb").read()).decode()
resp = client.chat.completions.create(
    model="supergemma4-26b-abliterated-multimodal-mlx-4bit",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
    ]}],
)
print(resp.choices[0].message.content)
```

### 嵌入向量

```bash
curl -s http://localhost:18080/v1/embeddings \
  -H "Authorization: Bearer sk-mlx" \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen3-Embedding-0.6B-4bit-DWQ", "input": "Hello"}' \
  | jq .data[0].embedding
```

### OCR（PaddleOCR-VL 通过对话接口）

```python
import base64
from openai import OpenAI
client = OpenAI(base_url="http://localhost:18080/v1", api_key="sk-mlx")
img_b64 = base64.b64encode(open("doc.jpg", "rb").read()).decode()
resp = client.chat.completions.create(
    model="PaddleOCR-VL-1.5-6bit",
    messages=[{"role": "user", "content": [
        {"type": "text", "text": "OCR this document."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
    ]}],
)
print(resp.choices[0].message.content)
```

### ASR — IM 模式（asr-router 自动路由）

```bash
curl -s http://localhost:18081/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-mlx" \
  -F "file=@voice.wav" \
  -F "model=auto"
# 返回 Whisper 兼容 JSON + x_route (sense_voice|omlx) + x_tags (语言/事件/情感)
```

Python 调用：

```python
import httpx

with open("voice.wav", "rb") as f:
    r = httpx.post(
        "http://localhost:18081/v1/audio/transcriptions",
        headers={"Authorization": "Bearer sk-mlx"},
        files={"file": ("voice.wav", f, "audio/wav")},
        data={"model": "auto"},
    )
print(r.json())  # {"text": "...", "x_route": "sense_voice", "x_tags": {...}}
```

### ASR — 会议模式（异步任务流水线）

```bash
# 1. 提交任务
JOB=$(curl -s http://localhost:18081/v1/audio/jobs \
  -H "Authorization: Bearer sk-mlx" \
  -F "file=@meeting.wav" \
  -F 'glossary=terms:
  - term: Alpha Group
    aliases: [Alpa Group]' \
  | jq -r .id)

# 2. 轮询状态
while true; do
  S=$(curl -s "http://localhost:18081/v1/audio/jobs/$JOB" \
       -H "Authorization: Bearer sk-mlx" | jq -r .status)
  echo "$S"; [ "$S" = "done" ] || [ "$S" = "failed" ] && break
  sleep 5
done

# 3. 获取产物
curl -s "http://localhost:18081/v1/audio/jobs/$JOB/artifact/meeting_gemma4.md" \
  -H "Authorization: Bearer sk-mlx"
```

## ASR 路由模块

`asr/` 目录包含一个独立的 FastAPI 服务，与 oMLX 并行运行，在 18081 端口暴露两个端点：同步 Whisper 兼容转写接口（`POST /v1/audio/transcriptions`）和异步会议流水线（`POST /v1/audio/jobs`）。

**IM 模式**在两个后端之间自动路由。30 秒以内且无模糊事件标签的短音频直接交给 sherpa-onnx SenseVoice 处理（解码延迟 60–90 ms，RTF ≈ 0.01）；较长音频、情感/事件标签不确定的音频，或显式指定 `quality=high` 的请求，则转发至 oMLX 的 Qwen3-ASR-1.7B-8bit 以获得更高精度。响应为 Whisper 兼容 JSON，扩展了 `x_route`（实际使用的后端）和 `x_tags`（SenseVoice 输出的语言、事件、情感）字段。

**会议模式**异步执行四阶段流水线：第一阶段 VAD + 说话人分割；第二阶段 SenseVoice 转写每段音频，附带语言和说话人标签；第三阶段 oMLX 上的 gemma-4-26b 进行上下文校对，应用按任务提交的词汇表修正专有名词、领域术语及跨语言谐音词；第四阶段将校对后的文本渲染为 5 份命名产物（SenseVoice 原始输出、gemma-4 校对 Markdown、说话人时间轴 JSON、分段 SRT 字幕、摘要）。产物通过 `GET /v1/audio/jobs/{id}/artifact/{name}` 获取。

按任务词汇表以 YAML 格式通过 `glossary` multipart 字段在提交时传入。基于项目联系人数据预填的默认词汇表位于 `asr/glossary/default.yaml`。完整 API 规范、路由逻辑和配置说明请参阅 [`asr/README.md`](asr/README.md)。

**实测效果：** gemma-4 上下文校对将字符错误率（CER）从 32.08% 降至 22.64%，在实际双语会议录音上实现 **29.4% 的相对 CER 降幅**（配合按任务词汇表）。方法论和完整结果见 [`asr/EVALUATION.md`](asr/EVALUATION.md)。

## 硬件要求

- Apple Silicon Mac（M1 / M2 / M3 / M4 系列）
- 加载 26B 模型时推荐 **16 GB 统一内存**（最低）；并发使用 LLM 和 VLM 时建议 32 GB
- 35B MoE 旗舰模型（`Qwen3.5-35B-A3B-4bit`，权重约 18 GB）至少需要 24 GB 空闲统一内存；oMLX 通过按需加载和 SSD 缓存机制在空闲时卸载模型
- SenseVoice（228 MB int8）和 Qwen3-Embedding（1 GB）持续驻留内存；两个小模型的常驻占用约 1.3 GB

## 当前状态

| 组件 | 状态 |
|------|------|
| oMLX 网关 | 运行中 — 已加载 7 个模型（oMLX 6 个 + asr-router 中的 SenseVoice） |
| asr-router IM 模式 | 已实现 — SenseVoice ↔ Qwen3-ASR 自动路由 |
| asr-router 会议流水线 | 已实现 — 四阶段：VAD/分割 → SenseVoice → gemma-4 校对 → 渲染 |
| gemma-4 上下文校对 | 已验证 — 在实际双语会议录音上 CER 降低 29.4% |
| 按任务词汇表 | 可用 — 在任务提交时以 YAML 传入 |

## 致谢

- [oMLX](https://github.com/jundot/oMLX) — 本地 MLX 推理 GUI 及 OpenAI 兼容网关（`brew tap jundot/omlx`）
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — SenseVoice 的快速 ONNX 运行时
- [SenseVoice / FunAudioLLM](https://github.com/FunAudioLLM/SenseVoice) — 支持情感与事件检测的多语言 ASR
- [mlx-community on Hugging Face](https://huggingface.co/mlx-community) — 量化 MLX 模型权重

## 许可证

[MIT](LICENSE)
