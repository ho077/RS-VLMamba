"""数据集入口。

统一从 RSRefSeg2 的 ``datainfo`` jsonl 标注构建数据集，
不再依赖 COCO 风格的 REFER 目录结构。

用法：
    from data import build_dataset
    ds = build_dataset('rrsisd', 'train', transform, args)
"""

from . import refsegrs
from . import rrsisd

#: 数据集注册表：名称 -> Dataset 类
DATASET_REGISTRY = {
    'refsegrs': refsegrs.ReferDataset,
    'rrsisd': rrsisd.ReferDataset,
}


def build_dataset(dataset, split, transform, args=None, eval_mode=False):
    """按名称构建数据集。

    Args:
        dataset: ``refsegrs`` 或 ``rrsisd``
        split:   ``train`` / ``val`` / ``test``
        transform: 需要同时处理 image 与 target 的 transforms.Compose
        args:    含 img_size / max_tokens / bert_tokenizer 等字段的配置对象
        eval_mode: 评估模式下保留全部指代句（不随机采样）
    """
    if dataset not in DATASET_REGISTRY:
        raise KeyError(f'未知数据集 {dataset!r}，可选：{sorted(DATASET_REGISTRY)}')
    return DATASET_REGISTRY[dataset](
        args=args,
        split=split,
        image_transforms=transform,
        target_transforms=None,
        eval_mode=eval_mode,
    )


__all__ = ['DATASET_REGISTRY', 'build_dataset', 'refsegrs', 'rrsisd']
