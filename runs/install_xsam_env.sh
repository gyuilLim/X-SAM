#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-xsam}"

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.10 -y
fi
conda activate "${ENV_NAME}"

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
conda install gcc=11 gxx=11 -c conda-forge -y
pip install git+https://github.com/InternLM/xtuner.git@v0.2.0
pip install -r "${ROOT_DIR}/xsam/requirements/deepspeed.txt"
pip install -r "${ROOT_DIR}/xsam/requirements/xsam.txt"
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -e "${ROOT_DIR}/xsam"
