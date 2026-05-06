#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="xsam/configs/xsam/s3_refseg_clean_finetune/xsam_phi3_refcocop_freeze_e1.py"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/wkdrs/s3_refseg_clean_finetune/xsam_phi3_refcocop_freeze_e1}"

export root_dir="${ROOT_DIR}"
export GPU_PER_NODE="${GPU_PER_NODE:-1}"
export MASTER_PORT="${MASTER_PORT:-29621}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "${ROOT_DIR}"
mkdir -p "${WORK_DIR}"

export CODE_DIR="${ROOT_DIR}/xsam/"
export DATA_DIR="${ROOT_DIR}/datas/"
export INIT_DIR="${ROOT_DIR}/inits/"
export WORK_DIR="${ROOT_DIR}/wkdrs/"
export LMUData="${DATA_DIR}/LMUData"
export HF_HOME="${INIT_DIR}/huggingface"
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export XTUNER_DATASET_TIMEOUT=120
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
elif [[ -d "/home/vision/anaconda3/envs/sam/lib" ]]; then
  export LD_LIBRARY_PATH="/home/vision/anaconda3/envs/sam/lib:${LD_LIBRARY_PATH:-}"
fi

cd "${ROOT_DIR}/xsam"
PYTHONPATH="${ROOT_DIR}/xsam:${PYTHONPATH:-}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m torch.distributed.run \
  --master_addr="${MASTER_ADDR:-localhost}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${GPU_PER_NODE}" \
  xsam/tools/train.py \
  "${CONFIG}" \
  --work-dir "${WORK_DIR}" \
  --resume auto \
  --launcher pytorch \
  --seed 1024
