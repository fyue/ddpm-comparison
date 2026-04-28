#!/usr/bin/env bash
# =============================================================================
# setup.sh — diffusion-sampler conda 環境のセットアップスクリプト
#
# 使い方:
#   chmod +x setup.sh
#   ./setup.sh
#
# 動作環境:
#   - macOS (Apple Silicon M1/M2/M3/M4/M5)  → MPS バックエンドで動作
#   - macOS (Intel)                          → CPU で動作
#   - Linux / Windows WSL2 (CUDA GPU あり)  → CUDA で動作
# =============================================================================

set -euo pipefail

ENV_NAME="diffusion-sampler"

echo "=== [1/3] conda 環境を作成します: ${ENV_NAME} ==="
conda env create -f environment.yml

echo ""
echo "=== [2/3] 環境をアクティベートします ==="
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo ""
echo "=== [3/3] セットアップ完了 ==="
echo ""
echo "次のコマンドで環境をアクティベートしてください:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "PyTorch デバイス確認:"
python - <<'EOF'
import torch
if torch.cuda.is_available():
    print(f"  デバイス: CUDA ({torch.cuda.get_device_name(0)})")
elif torch.backends.mps.is_available():
    print("  デバイス: MPS (Apple Silicon)")
else:
    print("  デバイス: CPU")
print(f"  PyTorch バージョン: {torch.__version__}")
EOF
