#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-xsam}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.10 -y
fi
conda activate "${ENV_NAME}"

pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${TORCH_INDEX_URL}"
conda install gcc=11 gxx=11 -c conda-forge -y
pip install git+https://github.com/InternLM/xtuner.git@v0.2.0
pip install -r "${ROOT_DIR}/xsam/requirements/deepspeed.txt"
pip install -r "${ROOT_DIR}/xsam/requirements/xsam.txt"
if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
fi
if [[ -f "${ROOT_DIR}/xsam/setup.py" || -f "${ROOT_DIR}/xsam/pyproject.toml" ]]; then
  pip install -e "${ROOT_DIR}/xsam"
else
  echo "Skip editable install: ${ROOT_DIR}/xsam has no setup.py or pyproject.toml; run scripts set PYTHONPATH."
fi
