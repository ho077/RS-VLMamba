"""RRSIS-D 指代分割数据集。

标注：``{DATAINFO_ROOT}/rrsisd_{split}.jsonl``
图像：``configs.paths.IMAGE_ROOTS['rrsisd']``（jsonl 中 file_name 为 *.jpg）

与 RefSegRS 的差别：
    - 每行额外带 category_id / category_name / bbox / area；
    - segmentation 字段是“实例列表”，取第一个实例作为目标掩码。
"""

from .base import JsonlReferDataset


class ReferDataset(JsonlReferDataset):
    dataset_name = 'rrsisd'

    # 语义类别（与 RSRefSeg2 / RRSIS-D 的类别体系保持一致）
    target_cls = (
        "airplane", "airport", "golf field", "expressway service area",
        "baseball field", "stadium", "ground track field", "storage tank",
        "basketball court", "chimney", "tennis court", "overpass",
        "train station", "ship", "expressway toll station", "dam",
        "harbor", "bridge", "vehicle", "windmill",
    )
