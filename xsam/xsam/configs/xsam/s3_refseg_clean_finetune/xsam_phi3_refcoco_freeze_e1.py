import os

import torch
from mmengine.dataset import DefaultSampler
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipImageProcessor, SiglipVisionModel
from xtuner.utils import PROMPT_TEMPLATE

from xsam.dataset.collate_fns.xsam_collate_fn import xsam_collate_fn
from xsam.dataset.map_fns.dataset_map_fn_factory import dataset_map_fn_factory
from xsam.dataset.map_fns.dataset_map_fns.refseg_map_fn import refseg_map_fn
from xsam.dataset.map_fns.template_map_fn_factory import template_map_fn_factory
from xsam.dataset.process_fns.postprocess_fns.refseg_process_fn import refseg_postprocess_fn
from xsam.dataset.processors.image_processing_sam import SamImageProcessor
from xsam.dataset.refseg_dataset import RefSegDataset
from xsam.engine.hooks.model_info_hook import ModelInfoHook
from xsam.engine.hooks.pt_checkpoint_hook import PTCheckpointHook
from xsam.engine.runner import TrainLoop
from xsam.model.xsam import XSamModel
from xsam.model.segmentors.x_segmentor import XSegmentor
from xsam.model.segmentors.mask2former import Mask2FormerConfig, Mask2FormerModel
from xsam.model.segmentors.sam import SamModel

code_dir = os.getenv("CODE_DIR", "./xsam/")
data_dir = os.getenv("DATA_DIR", "./datas/")
init_dir = os.getenv("INIT_DIR", "./inits/")
work_dir = os.getenv("WORK_DIR", "./wkdrs/")

llm_name_or_path = init_dir + "Phi-3-mini-4k-instruct"
visual_encoder_name_or_path = init_dir + "siglip2-so400m-patch14-384"
seg_encoder_name_or_path = init_dir + "sam-vit-large"
seg_decoder_name_or_path = init_dir + "mask2former-swin-large-coco-panoptic"

s1_pretrained_pth = work_dir + "s1_seg_finetune/xsam_sam_large_m2f_e36_gpu16_seg_finetune/pytorch_model.bin"
s2_pretrained_pth = (
    work_dir
    + "s2_align_pretrain/xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_e1_gpu16_align_pretrain/pytorch_model.bin"
)

prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = int(4096 - (384 / 14) ** 2 - 1024)

batch_size = 1
accumulative_counts = 8
dataloader_num_workers = 2
max_epochs = 1
optim_type = AdamW
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1
warmup_ratio = 0.03
save_steps = 2000
save_total_limit = 2
logging_interval = 10

special_tokens = ["<SEG>", "<p>", "</p>"]
cond_type = "phrase"
ignore_label = 255

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side="right",
)

image_processor = dict(
    type=SiglipImageProcessor.from_pretrained,
    pretrained_model_name_or_path=visual_encoder_name_or_path,
    trust_remote_code=True,
)

extra_image_processor = dict(
    type=SamImageProcessor.from_pretrained,
    pretrained_model_name_or_path=seg_encoder_name_or_path,
    trust_remote_code=True,
    ignore_index=0,
)

model = dict(
    type=XSamModel,
    freeze_llm=True,
    freeze_visual_encoder=True,
    freeze_segmentor_encoder=True,
    use_dual_encoder=True,
    use_vision_sampler=True,
    connector_type="conv",
    cond_type=cond_type,
    seg_select_layers=[6, 12, 18, 24],
    connector_hidden_dim=512,
    connector_scale_factor=[4, 2, 1, 0.5],
    sampler_input_feat="extra_pixel_values",
    special_tokens=special_tokens,
    s1_pretrained_pth=s1_pretrained_pth,
    s2_pretrained_pth=s2_pretrained_pth,
    tokenizer=tokenizer,
    postprocess_fn=refseg_postprocess_fn,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ),
    visual_encoder=dict(
        type=SiglipVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path,
        torch_dtype=torch.bfloat16,
    ),
    segmentor=dict(
        type=XSegmentor,
        encoder=dict(
            type=SamModel.from_pretrained,
            pretrained_model_name_or_path=seg_encoder_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ),
        decoder=dict(
            type=Mask2FormerModel._from_config,
            config=dict(
                type=Mask2FormerConfig.from_pretrained,
                pretrained_model_name_or_path=seg_decoder_name_or_path,
                use_backbone=False,
                feature_channels=[512, 1024, 2048],
                num_feature_levels=3,
                trust_remote_code=True,
            ),
            torch_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        reinit_decoder=True,
        open_cls=True,
    ),
)

refseg_data_root = data_dir + "refseg_data/"
refcoco_refseg_dataset = dict(
    type=RefSegDataset,
    data_root=refseg_data_root,
    image_folder=refseg_data_root + "images/train2014",
    dataset="refcoco",
    data_split="train",
    tokenizer=tokenizer,
    task_name="refseg",
    data_name="refcoco_refseg",
    cond_type=cond_type,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    image_processor=image_processor,
    postprocess_fn=refseg_postprocess_fn,
    dataset_map_fn=dict(
        type=dataset_map_fn_factory,
        fn=refseg_map_fn,
        cond_type=cond_type,
    ),
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    use_variant_cat=True,
    use_random_cat=True,
    max_length=max_length,
    pad_image_to_square=False,
    ignore_label=ignore_label,
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=refcoco_refseg_dataset,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=xsam_collate_fn),
)

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale=dict(enabled=False),
    dtype="bfloat16",
    paramwise_cfg=dict(
        bypass_duplicate=True,
        custom_keys={
            "segmentor.encoder": dict(lr_mult=0.1, decay_mult=1.0),
            "visual_encoder": dict(lr_mult=0.1, decay_mult=1.0),
        },
    ),
)

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

custom_hooks = [
    dict(
        type=ModelInfoHook,
        module_names=["llm", "visual_encoder", "projector", "connector", "segmentor"],
        display_params=True,
    ),
    dict(type=PTCheckpointHook, clean_pth=False),
]

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=logging_interval),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=save_steps, max_keep_ckpts=save_total_limit),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

visualizer = None
log_level = "INFO"
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
log_processor = dict(
    by_epoch=False,
    window_size=1,
    mean_pattern=r".*(loss|time|data_time|grad_norm|tflops).*",
)

val_evaluators = []
