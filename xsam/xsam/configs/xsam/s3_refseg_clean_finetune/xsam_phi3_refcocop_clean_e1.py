from pathlib import Path

_base = Path(__file__).resolve().parents[1] / "s3_mixed_finetune" / "xsam_phi3_mini_4k_instruct_siglip2_so400m_p14_384_sam_large_m2f_gpu16_mixed_finetune.py"
exec(compile(_base.read_text(), str(_base), "exec"))

s1_pretrained_pth = None
model["s1_pretrained_pth"] = None

batch_size = 1
accumulative_counts = 4
dataloader_num_workers = 2
max_epochs = 1
save_steps = 2000
logging_interval = 10
evaluation_freq = 10**12

train_datasets = dict(
    type=ConcatDataset,
    oversample_ratio=1.0,
    datasets=[refcocop_refseg_dataset],
)
train_dataloader.update(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_datasets,
    persistent_workers=True,
)
train_dataloader["sampler"].update(
    per_device_batch_size=batch_size * accumulative_counts,
)

val_datasets = [
    dataset
    for dataset in val_datasets
    if dataset.get("data_name") in {
        "refcoco+_val_refseg",
        "refcoco+_testA_refseg",
        "refcoco+_testB_refseg",
    }
]
test_datasets = val_datasets
vis_datasets = val_datasets

evaluators = [
    dict(type=RefSegEvaluator, distributed=True, data_name="refcoco+_val_refseg"),
    dict(type=RefSegEvaluator, distributed=True, data_name="refcoco+_testA_refseg"),
    dict(type=RefSegEvaluator, distributed=True, data_name="refcoco+_testB_refseg"),
]

custom_hooks = [
    dict(
        type=ModelInfoHook,
        module_names=["llm", "visual_encoder", "projector", "connector", "segmentor"],
        display_params=True,
    ),
    dict(type=DatasetInfoHook, tokenizer=tokenizer, special_tokens=special_tokens),
    dict(type=PTCheckpointHook, clean_pth=False),
]

default_hooks["checkpoint"].update(interval=save_steps, max_keep_ckpts=save_total_limit)
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)
