<p align="center">
  <h1 align="center">🧠 MLX Local Inference Stack</h1>
  <p align="center">
    Give your Apple Silicon Mac the power to hear, see, read, speak, think — all locally.
  </p>
  <p align="center">
    <a href="https://clawhub.ai/skills/mlx-local-inference"><img src="https://img.shields.io/badge/ClawHub-mlx--local--inference-FF5A36?style=flat-square" alt="ClawHub"></a>
    <a href="#"><img src="https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-000?style=flat-square&logo=apple&logoColor=white" alt="Platform"></a>
    <a href="#"><img src="https://img.shields.io/badge/gateway-oMLX-blue?style=flat-square" alt="oMLX"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License"></a>
  </p>
  <p align="center">
    <a href="README_CN.md"><b>中文</b></a> · English
  </p>
</p>

---

## Installation

```bash
# Clone repository
git clone https://github.com/bendusy/mlx-local-inference.git
cd mlx-local-inference

# Install Python libraries
pip install mlx-lm mlx-vlm mlx-whisper

# Install oMLX (for LLM/VLM)
brew install omlx
# or: pip install omlx

# Start oMLX server
omlx serve --model-dir ~/models --port 8000
```

## Why This Exists

Your M-series Mac has powerful unified memory — yet most AI workflows still send every request to the cloud. **MLX Local Inference Stack** turns your Mac into a fully self-contained AI workstation, with a memory-efficient design that works on **16 GB machines**.

## What Your Mac Gains

| Ability | Model (oMLX id) | Memory | Load Strategy |
|:--------|:----------------|:-------|:--------------|
| 📐 **Embed** | `Qwen3-Embedding-0.6B-4bit-DWQ` | ~1 GB | **On-demand** |
| 👂 **Hear** | `Qwen3-ASR-1.7B-8bit` | ~1.5 GB | **On-demand** |
| 🧠 **Think** | `Qwen3-14B-4bit` / `Qwen3.5-35B-A3B-4bit` | ~8–22 GB | **On-demand** |
| 👁️ **See (OCR)** | `PaddleOCR-VL-1.5-6bit` | ~3.3 GB | **On-demand** |

## Architecture

**Hybrid approach:** oMLX for LLM/VLM (high performance), Python libraries for Embedding/ASR/OCR (simplicity).

```
┌─────────────────────────────────────┐
│  oMLX (localhost:8000/v1)           │
│  - LLM (Qwen3-14B, etc.)            │
│  - VLM (vision-language models)     │
│  - Continuous batching + SSD cache  │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  Python Libraries (direct call)     │
│  - mlx-lm: Embedding                │
│  - mlx-vlm: OCR (PaddleOCR-VL)      │
│  - mlx-whisper: ASR (Qwen3-ASR)     │
└─────────────────────────────────────┘
```

## What Your Mac Gains

| Ability | Implementation | Model | Memory |
|:--------|:--------------|:------|:-------|
| 💬 **Think** | oMLX API | `Qwen3-14B-4bit` | ~8 GB |
| 👁️ **See (VLM)** | oMLX API | Any mlx-vlm model | varies |
| 📐 **Embed** | mlx-lm (Python) | `Qwen3-Embedding-0.6B-4bit-DWQ` | ~1 GB |
| 👂 **Hear** | mlx-whisper (Python) | `Qwen3-ASR-1.7B-8bit` | ~1.5 GB |
| 👁️ **Read (OCR)** | mlx-vlm (Python) | `PaddleOCR-VL-1.5-6bit` | ~3.3 GB |

## Usage

### 💬 LLM — Text Generation (via oMLX API)

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")

response = client.chat.completions.create(
    model="Qwen3-14B-4bit",
    messages=[{"role": "user", "content": "Hello"}]
)
print(response.choices[0].message.content)
```

### 📐 Embed — Text Vectorization (via mlx-lm)

```python
from mlx_lm import load

model, tokenizer = load("mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")
inputs = tokenizer("text to embed", return_tensors="np")
embeddings = model(**inputs).last_hidden_state.mean(axis=1)
```

### 👂 Hear — Speech Recognition (via mlx-whisper)

```python
import mlx_whisper

result = mlx_whisper.transcribe(
    "audio.wav",
    path_or_hf_repo="mlx-community/Qwen3-ASR-1.7B-8bit"
)
print(result["text"])
```

### 👁️ Read — OCR (via mlx-vlm)

```python
from mlx_vlm import load, generate
from mlx_vlm.utils import load_image

model, processor = load("mlx-community/PaddleOCR-VL-1.5-6bit")
image = load_image("document.jpg")

output = generate(model, processor, image, "OCR:", max_tokens=512, temp=0.0)
print(output)
```

## Service Management (oMLX)

```bash
# List discovered models
curl http://localhost:8000/v1/models

# Restart service
launchctl kickstart -k gui/$(id -u)/com.omlx-server

# Logs
tail -f /tmp/omlx-server.log
```

## Notes

- oMLX is used **only** for LLM/VLM (chat/completions).
- Embedding/ASR/OCR are handled by Python libraries because oMLX does not support Qwen3-Embedding or Qwen3-ASR.
- Models can still be stored in `~/models/` for consistency, but Python libs can load directly from HuggingFace.


## Project Structure

```
mlx-local-inference/
├── SKILL.md
├── README.md
├── README_CN.md
├── references/
└── ...
```

## License

[MIT](LICENSE)
