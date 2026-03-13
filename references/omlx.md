# omlx API Reference

omlx 是 OpenAI-compatible 本地推理服务器，统一管理所有 MLX 模型。

## 端点

Base URL: `http://localhost:8000/v1`

| 端点 | 功能 |
|------|------|
| `GET /v1/models` | 列出已注册模型 |
| `POST /v1/chat/completions` | LLM / VLM / OCR |
| `POST /v1/embeddings` | 文本向量化 |
| `POST /v1/audio/transcriptions` | 语音转文字 |

## 模型目录结构

```
~/models/
├── Qwen3-Embedding-0.6B-4bit-DWQ/
│   ├── config.json
│   ├── tokenizer.json
│   └── *.safetensors
├── Qwen3-ASR-1.7B-8bit/
├── PaddleOCR-VL-1.5-6bit/
└── Qwen3-14B-4bit/          # 可选，按需下载
```

模型名即目录名（不含 org 前缀），omlx 自动扫描 `~/models/` 下所有含 `config.json` 的子目录。

## 配置文件

`~/.omlx/settings.json`:
```json
{
  "model_dir": "/Users/ben/models",
  "port": 8000,
  "host": "0.0.0.0",
  "max_model_memory": 12,
  "hot_cache_max_size": 2
}
```

## CLI 常用命令

```bash
# 启动服务
omlx serve --model-dir ~/models --port 8000

# 列出已发现模型
curl http://localhost:8000/v1/models

# 手动下载模型（真实文件，不用 symlinks）
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='mlx-community/Qwen3-14B-4bit',
    local_dir='$HOME/models/Qwen3-14B-4bit',
    local_dir_use_symlinks=False
)
"
```

## 注意事项

- 模型名在 API 调用时**不含** `mlx-community/` 前缀，直接用目录名
- `local_dir_use_symlinks=False` 必须设置，否则 omlx 无法发现模型（符号链接指向 HF cache blobs，移动后会断）
- OCR 调用时 prompt 必须是 `"OCR:"`，temperature 必须是 `0.0`
