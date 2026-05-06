#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
INITS = ROOT / "inits"
WKDRS = ROOT / "wkdrs"


def download_snapshot(repo_id: str, local_name: str, allow_patterns: list[str] | None = None) -> None:
    local_dir = INITS / local_name
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[download] {repo_id} -> {local_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        resume_download=True,
    )


def download_xsam_file(repo_path: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[skip] {dst}")
        return
    print(f"[download] hao9610/X-SAM:{repo_path} -> {dst}")
    src = hf_hub_download(repo_id="hao9610/X-SAM", filename=repo_path, resume_download=True)
    tmp = Path(src)
    if tmp.resolve() != dst.resolve():
        dst.write_bytes(tmp.read_bytes())


def main() -> None:
    INITS.mkdir(parents=True, exist_ok=True)
    WKDRS.mkdir(parents=True, exist_ok=True)

    download_snapshot("microsoft/Phi-3-mini-4k-instruct", "Phi-3-mini-4k-instruct")
    download_snapshot("google/siglip2-so400m-patch14-384", "siglip2-so400m-patch14-384")
    download_snapshot("facebook/sam-vit-large", "sam-vit-large", allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin"])
    download_snapshot(
        "facebook/mask2former-swin-large-coco-panoptic",
        "mask2former-swin-large-coco-panoptic",
        allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin"],
    )

    download_xsam_file(
        "s2_align_pretrain/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_e1_gpu16_align_pretrain/pytorch_model.bin",
        WKDRS
        / "s2_align_pretrain"
        / "xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_e1_gpu16_align_pretrain"
        / "pytorch_model.bin",
    )


if __name__ == "__main__":
    main()
