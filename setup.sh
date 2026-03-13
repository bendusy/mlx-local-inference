#!/usr/bin/env bash
# mlx-local-inference setup — omlx unified gateway
# Models stored permanently in ~/models, served via omlx at localhost:8000/v1
set -euo pipefail

MODELS_DIR="$HOME/models"
OMLX_CONFIG_DIR="$HOME/.omlx"
PLIST_DST="$HOME/Library/LaunchAgents/com.omlx-server.plist"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== mlx-local-inference omlx setup ==="

# 1. Install omlx
if ! command -v omlx &>/dev/null; then
  echo "Installing omlx..."
  if command -v brew &>/dev/null; then
    brew tap jundot/omlx && brew install omlx
  else
    pip install omlx
  fi
else
  echo "✓ omlx already installed"
fi

# 2. Create permanent model storage
mkdir -p "$MODELS_DIR"

# 3. Download models (skip if already present)
download_model() {
  local model_id="$1"
  local model_name="${model_id##*/}"
  local local_dir="$MODELS_DIR/$model_name"
  if [ -f "$local_dir/config.json" ]; then
    echo "✓ $model_name already exists, skipping"
  else
    echo "Downloading $model_id..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$model_id',
    local_dir='$local_dir',
    local_dir_use_symlinks=False
)
print('✓ Done')
"
  fi
}

download_model "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
download_model "mlx-community/Qwen3-ASR-1.7B-8bit"
download_model "mlx-community/PaddleOCR-VL-1.5-6bit"

# 4. Write omlx config
mkdir -p "$OMLX_CONFIG_DIR"
cat > "$OMLX_CONFIG_DIR/settings.json" <<EOF
{
  "model_dir": "$MODELS_DIR",
  "port": 8000,
  "host": "0.0.0.0",
  "max_model_memory": 12,
  "hot_cache_max_size": 2
}
EOF
echo "✓ omlx config written to $OMLX_CONFIG_DIR/settings.json"

# 5. Install launchd service
if [ -f "$SCRIPT_DIR/com.omlx-server.plist" ]; then
  sed "s|__HOME__|$HOME|g" "$SCRIPT_DIR/com.omlx-server.plist" > "$PLIST_DST"
  launchctl unload "$PLIST_DST" 2>/dev/null || true
  launchctl load "$PLIST_DST"
  echo "✓ launchd service registered"
fi

echo ""
echo "=== Setup complete ==="
echo "omlx endpoint: http://localhost:8000/v1"
echo "Models dir:    $MODELS_DIR"
echo ""
echo "Test: curl http://localhost:8000/v1/models"
