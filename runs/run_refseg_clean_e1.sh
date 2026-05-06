#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATASET="${1:-}"
MODES="${MODES:-train,segeval}"
GPU_PER_NODE="${GPU_PER_NODE:-1}"
MASTER_PORT="${MASTER_PORT:-29610}"

case "${DATASET}" in
  refcoco)
    CONFIG="xsam/configs/xsam/s3_refseg_clean_finetune/xsam_phi3_refcoco_clean_e1.py"
    WORK_DIR="${WORK_DIR:-${ROOT_DIR}/wkdrs/s3_refseg_clean_finetune/xsam_phi3_refcoco_clean_e1}"
    ;;
  refcoco+|refcocop)
    CONFIG="xsam/configs/xsam/s3_refseg_clean_finetune/xsam_phi3_refcocop_clean_e1.py"
    WORK_DIR="${WORK_DIR:-${ROOT_DIR}/wkdrs/s3_refseg_clean_finetune/xsam_phi3_refcocop_clean_e1}"
    ;;
  refcocog)
    CONFIG="xsam/configs/xsam/s3_refseg_clean_finetune/xsam_phi3_refcocog_clean_e1.py"
    WORK_DIR="${WORK_DIR:-${ROOT_DIR}/wkdrs/s3_refseg_clean_finetune/xsam_phi3_refcocog_clean_e1}"
    ;;
  *)
    echo "Usage: $0 {refcoco|refcoco+|refcocog}" >&2
    exit 2
    ;;
esac

export root_dir="${ROOT_DIR}"
export GPU_PER_NODE
export MASTER_PORT
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "${ROOT_DIR}"
bash runs/run.sh --modes "${MODES}" --config "${CONFIG}" --work-dir "${WORK_DIR}"

