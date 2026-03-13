# MLX Local Inference Stack — 混合架构

Apple Silicon Mac 上的全本地 AI 推理栈。使用 **oMLX** 处理 LLM/VLM 推理，使用 **Python 库**（mlx-lm、mlx-whisper、mlx-vlm）处理 Embedding、ASR、OCR。

## 安装

```bash
# 1. 克隆仓库
git clone https://github.com/bendusy/mlx-local-inference.git
cd mlx-local-inference

# 2. 安装 Python 库
pip install mlx-lm mlx-whisper mlx-vlm huggingface_hub

# 3. 下载模型到 ~/models/（oMLX 兼容结构）
python3 -c "
from huggingface_hub import snapshot_download
import os

models = [
    'mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ',
    'mlx-community/Qwen3-ASR-1.7B-8bit',
    'mlx-community/PaddleOCR-VL-1.5-6bit',
    'mlx-community/Qwen3-14B-4bit'
]

for repo_id in models:
    model_name = repo_id.split('/')[-1]
    local_dir = os.path.expanduser(f'~/models/{model_name}')
    print(f'正在下载 {model_name}...')
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
"

# 4. 安装 oMLX（用于 LLM/VLM）
brew install omlx
# 或：pip install omlx

# 5. 启动 oMLX 服务器
omlx serve --model-dir ~/models --port 8000
```

## 为什么存在

你的 M 系列 Mac 拥有强大的统一内存——但大多数 AI 工作流仍然把每个请求发到云端。**MLX Local Inference Stack** 把你的 Mac 变成完全自给自足的 AI 工作站，内存高效设计可在 **16 GB 机器**上运行。

## 你的 Mac 获得的能力

| 能力 | 模型 | 内存 | 实现方式 |
|:-----|:-----|:-----|:---------|
| 📐 **Embed** | `Qwen/Qwen3-Embedding-4B` | ~4 GB | **mlx-lm** (Python) |
| 👂 **Hear** | `mlx-community/whisper-large-v3-turbo` | ~1.5 GB | **mlx-whisper** (Python) |
| 🧠 **Think** | `mlx-community/Qwen3-14B-4bit` | ~8 GB | **oMLX** (OpenAI API) |
| 👁️ **See (OCR)** | `mlx-community/PaddleOCR-VL-1.5-6bit` | ~3.3 GB | **mlx-vlm** (Python) |

## 架构

```
┌─────────────────────────────────────────┐
│          你的 Agent / 应用               │
└─────────┬───────────────────────────────┘
          │
          ├─ LLM/VLM ──────────────────────┐
          │  (OpenAI 兼容 API)              │
          │                                 ▼
          │                    ┌────────────────────────┐
          │                    │ oMLX (omlx serve)      │
          │                    │ localhost:8000/v1      │
          │                    └────────────────────────┘
          │
          ├─ Embedding ────────────────────┐
          │  (Python 库)                    │
          │                                 ▼
          │                    ┌────────────────────────┐
          │                    │ mlx_lm.generate()      │
          │                    └────────────────────────┘
          │
          ├─ ASR ─────────────────────────┐
          │  (Python 库)                   │
          │                                ▼
          │                    ┌────────────────────────┐
          │                    │ mlx_whisper.transcribe │
          │                    └────────────────────────┘
          │
          └─ OCR ─────────────────────────┐
             (Python 库)                   │
                                           ▼
                               ┌────────────────────────┐
                               │ mlx_vlm.generate()     │
                               └────────────────────────┘
```

## 使用方法

### 1. Embedding（通过 mlx-lm）

```python
from mlx_lm import load, generate

model, tokenizer = load("~/models/Qwen3-Embedding-4B")
text = "要向量化的文本"
inputs = tokenizer(text, return_tensors="np")
embeddings = model(**inputs).last_hidden_state.mean(axis=1)
print(embeddings.shape)  # (1, hidden_size)
```

### 2. ASR 语音识别（通过 mlx-whisper）

```python
import mlx_whisper

result = mlx_whisper.transcribe(
    "audio.wav",
    path_or_hf_repo="~/models/whisper-large-v3-turbo",
    language="zh"
)
print(result["text"])
```

支持格式：`wav`, `mp3`, `m4a`, `flac`, `ogg`, `webm`

### 3. OCR（通过 mlx-vlm）

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_image

model, processor = load("~/models/PaddleOCR-VL-1.5-6bit")
image = load_image("receipt.jpg")

prompt = apply_chat_template(
    processor, config, "OCR:", num_images=1
)

output = generate(model, processor, image, prompt, max_tokens=500, temp=0.0)
print(output)
```

### 4. LLM（通过 oMLX）

启动 oMLX 服务器：

```bash
omlx serve --model ~/models/Qwen3-14B-4bit --port 8000
```

然后使用 OpenAI 客户端：

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="Qwen3-14B-4bit",
    messages=[{"role": "user", "content": "你好"}]
)
print(response.choices[0].message.content)
```

## 服务管理

### oMLX 服务器（仅用于 LLM/VLM）

```bash
# 启动服务器
omlx serve --model ~/models/Qwen3-14B-4bit --port 8000

# 列出可用模型
omlx list

# 查看帮助
omlx serve --help
```

### Python 库（Embedding/ASR/OCR）

直接在 Python 代码中导入使用，无需运行服务器。

## 注意事项

- **所有模型统一存储在 `~/models/`，使用 oMLX 兼容结构**（如 `~/models/Qwen3-14B-4bit/`）
- **oMLX 仅用于 LLM/VLM 推理**（通过 OpenAI 兼容 API）
- **Embedding、ASR、OCR 使用 Python 库**（mlx-lm、mlx-whisper、mlx-vlm）
- **面向未来：** 当 oMLX 支持 Embedding/ASR 时，可立即切换，无需重新下载模型

## 项目结构

```
mlx-local-inference/
├── SKILL.md          # Agent 技能定义
├── README.md         # 英文文档
├── README_CN.md      # 中文文档
└── references/       # 参考文档
    ├── omlx.md
    └── ...
```

## License

[MIT](LICENSE)
