from copy import deepcopy
from pathlib import Path

_base = Path(__file__).with_name("xsam_phi3_refcoco_freeze_e1.py")
exec(compile(_base.read_text(), str(_base), "exec"))

refcocop_refseg_dataset = deepcopy(refcoco_refseg_dataset)
refcocop_refseg_dataset.update(
    dataset="refcoco+",
    data_name="refcoco+_refseg",
)

train_dataloader["dataset"] = refcocop_refseg_dataset
