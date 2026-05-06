#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

required_paths=(
  "${ROOT_DIR}/datas/refseg_data/refcoco/refs(unc).p"
  "${ROOT_DIR}/datas/refseg_data/refcoco/instances.json"
  "${ROOT_DIR}/datas/refseg_data/refcoco+/refs(unc).p"
  "${ROOT_DIR}/datas/refseg_data/refcoco+/instances.json"
  "${ROOT_DIR}/datas/refseg_data/refcocog/refs(umd).p"
  "${ROOT_DIR}/datas/refseg_data/refcocog/instances.json"
  "${ROOT_DIR}/datas/refseg_data/images/train2014"
  "${ROOT_DIR}/inits/Phi-3-mini-4k-instruct"
  "${ROOT_DIR}/inits/siglip2-so400m-patch14-384"
  "${ROOT_DIR}/inits/sam-vit-large"
  "${ROOT_DIR}/inits/mask2former-swin-large-coco-panoptic"
  "${ROOT_DIR}/wkdrs/s2_align_pretrain/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_e1_gpu16_align_pretrain/pytorch_model.bin"
)

missing=0
for path in "${required_paths[@]}"; do
  if [ ! -e "${path}" ]; then
    echo "MISSING ${path}"
    missing=1
  else
    echo "OK      ${path}"
  fi
done

exit "${missing}"
