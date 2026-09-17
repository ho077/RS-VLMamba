"""RefSegRS 指代分割数据集。

标注：``{DATAINFO_ROOT}/refsegrs_{split}.jsonl``
图像：``configs.paths.IMAGE_ROOTS['refsegrs']``（jsonl 中 file_name 为 *.tif）
"""

from .base import JsonlReferDataset


class ReferDataset(JsonlReferDataset):
    dataset_name = 'refsegrs'

    # 语义类别（与 RSRefSeg2 的 RefSegDataset 保持一致）
    target_cls = (
        "road", "vehicle", "car", "van", "building", "truck",
        "trailer", "bus", "road marking", "bikeway", "sidewalk",
        "tree", "low vegetation", "impervious surface",
    )
