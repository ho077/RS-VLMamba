"""路径配置 —— 全仓库唯一集中管理路径的地方。

设计原则：
    1. 代码中不再出现任何硬编码的绝对路径；
    2. 所有路径都可以通过环境变量覆盖，便于换机器/换数据集；
    3. 数据集标注统一来自 RSRefSeg2 的 ``datainfo`` 目录（jsonl 格式），
       支持 RefSegRS 与 RRSIS-D 两个数据集。

默认值按本机目录布局给出，迁移到其它机器时只需修改本文件或设置环境变量。
"""

import os

# --------------------------------------------------------------------------- #
# 1. 标注文件（*.jsonl）根目录
#    文件命名约定：{dataset}_{split}.jsonl，例如 rrsisd_train.jsonl
# --------------------------------------------------------------------------- #
DATAINFO_ROOT = os.environ.get(
    "PVLMAMBA_DATAINFO_ROOT",
    r"D:\code\RRISS\RSRefSeg2\datainfo",
)

ANNOTATION_TEMPLATE = "{dataset}_{split}.jsonl"

# --------------------------------------------------------------------------- #
# 2. 各数据集图像根目录
#    jsonl 中的 file_name 是相对该目录的文件名
# --------------------------------------------------------------------------- #
IMAGE_ROOTS = {
    "refsegrs": os.environ.get(
        "PVLMAMBA_REFSEGRS_IMAGES",
        r"D:\Datasets\RefSegRS\images",
    ),
    "rrsisd": os.environ.get(
        "PVLMAMBA_RRSISD_IMAGES",
        r"D:\Datasets\RRSIS-D\images\rrsisd\JPEGImages",
    ),
}

# 支持的切分
SPLITS = ("train", "val", "test")

# --------------------------------------------------------------------------- #
# 3. 预训练权重
#    留空表示由 backbone 工厂自动下载到缓存目录（PVLMAMBA_CACHE_DIR）
# --------------------------------------------------------------------------- #
PRETRAINED_SWIN_WEIGHTS = os.environ.get("PVLMAMBA_SWIN_WEIGHTS", "")
PRETRAINED_MAMBAVISION_WEIGHTS = os.environ.get("PVLMAMBA_MAMBAVISION_WEIGHTS", "")

# 权重/模型下载缓存目录
CACHE_DIR = os.environ.get(
    "PVLMAMBA_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "pvlmamba"),
)

# --------------------------------------------------------------------------- #
# 4. 文本编码器（HuggingFace 模型标识，非本机路径）
# --------------------------------------------------------------------------- #
BERT_TOKENIZER = os.environ.get("PVLMAMBA_BERT_TOKENIZER", "bert-base-uncased")
BERT_WEIGHTS = os.environ.get("PVLMAMBA_BERT_WEIGHTS", "bert-base-uncased")

# --------------------------------------------------------------------------- #
# 5. 输出
# --------------------------------------------------------------------------- #
CHECKPOINT_DIR = os.environ.get("PVLMAMBA_CKPT_DIR", "outputs/checkpoints")
VISUAL_DIR = os.environ.get("PVLMAMBA_VISUAL_DIR", "outputs/visual")


def annotation_file(dataset: str, split: str) -> str:
    """返回某数据集某个切分的标注文件绝对路径。"""
    return os.path.join(DATAINFO_ROOT, ANNOTATION_TEMPLATE.format(dataset=dataset, split=split))


def image_root(dataset: str) -> str:
    """返回某数据集的图像根目录。"""
    if dataset not in IMAGE_ROOTS:
        raise KeyError(f"未知数据集 {dataset!r}，可选：{sorted(IMAGE_ROOTS)}")
    return IMAGE_ROOTS[dataset]
