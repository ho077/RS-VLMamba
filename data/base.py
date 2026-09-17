"""指代遥感图像分割数据集的公共实现。

数据来源统一为 RSRefSeg2 的 ``datainfo`` 目录下的 jsonl 标注：

    {DATAINFO_ROOT}/{dataset}_{split}.jsonl

每行一条样本，字段约定：
    refsegrs: split, image_id, sent, file_name, segmentation(dict: COCO RLE)
    rrsisd  : split, image_id, sent, file_name, category_id, category_name,
              ann_id, bbox, area, segmentation(list[dict] | dict)

本模块负责：
    1. 读取 jsonl 与图像；
    2. 将 RLE 标注解码为 {0, 1} 二值掩码（优先 pycocotools，缺失时回落纯 numpy 实现）；
    3. 用 BERT tokenizer 编码指代文本并做定长 padding；
    4. 统一返回 (image, target, input_ids, attention_mask, save_prefix, refer_text)。
"""

import json
import os
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from bert.tokenization_bert import BertTokenizer
from configs import paths


# --------------------------------------------------------------------------- #
# RLE 解码
# --------------------------------------------------------------------------- #
def _leb128_counts(buf):
    """解码 COCO 压缩 RLE 的变长字节流，得到交替的 0/1 游程长度。"""
    cnts, p, n = [], 0, len(buf)
    while p < n:
        x, k, more = 0, 0, True
        while more:
            if p >= n:
                raise ValueError('RLE 字节流不完整')
            c = buf[p] - 48          # 编码时每个字节 +48
            p += 1
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            k += 1
            if (not more) and (c & 0x10):
                x |= -1 << (5 * k)
        if len(cnts) > 2:            # 从第 4 个起为差分值
            x += cnts[len(cnts) - 2]
        cnts.append(x)
    return cnts


def _counts_bytes(counts):
    """把 counts 转成字节流。

    标注在 json 中可能是 utf-8 或 latin-1 序列化而来，两种编码都试一遍，
    以“游程总和等于 H*W”为准。
    """
    if not isinstance(counts, str):
        return counts
    for enc in ('utf-8', 'latin-1'):
        try:
            yield counts.encode(enc)
        except UnicodeEncodeError:
            continue


def _decode_rle_pycocotools(rle):
    from pycocotools import mask as mask_utils
    if isinstance(rle['counts'], str):
        for buf in _counts_bytes(rle['counts']):
            try:
                out = mask_utils.decode({'size': rle['size'], 'counts': buf})
                if out is not None:
                    return out
            except Exception:
                continue
        raise ValueError('pycocotools 无法解码该 RLE')
    return mask_utils.decode(rle)


def _decode_rle_numpy(rle):
    """纯 numpy 的 COCO 压缩 RLE 解码（pycocotools 缺失时的回退实现）。"""
    h, w = rle['size']
    expected = h * w
    last_err = None
    if isinstance(rle['counts'], str):
        candidates = _counts_bytes(rle['counts'])
    else:
        candidates = [rle['counts']]

    for buf in candidates:
        try:
            cnts = _leb128_counts(buf)
        except Exception as e:  # 字节流被错误编码时会在这里失败
            last_err = e
            continue
        if sum(c for c in cnts if c > 0) != expected:
            last_err = ValueError(f'游程总和与图像尺寸不符（期望 {expected}）')
            continue

        flat = np.zeros(expected, dtype=np.uint8)
        idx = 0
        for i, c in enumerate(cnts):
            if c > 0:
                if i % 2 == 1:       # 奇数段为前景
                    flat[idx:idx + c] = 1
                idx += c
        return flat.reshape((h, w), order='F')  # COCO RLE 为列优先

    raise ValueError(f'无法解码 RLE：{last_err}')


def decode_segmentation(segmentation):
    """把 jsonl 中的 segmentation 字段解码成 (H, W) 的二值 uint8 掩码。"""
    # rrsisd 的 segmentation 是实例列表，取第一个实例
    if isinstance(segmentation, list):
        segmentation = segmentation[0]
    try:
        return _decode_rle_pycocotools(segmentation)
    except Exception:
        return _decode_rle_numpy(segmentation)


# --------------------------------------------------------------------------- #
# 图像读取
# --------------------------------------------------------------------------- #
def _cv2_available():
    try:
        import cv2  # noqa: F401
        return True
    except ImportError:
        return False


def load_image(path, backend='auto'):
    """读取图像为 RGB 的 PIL.Image。

    某些环境下 Pillow 解码 PackBits 压缩的 TIFF（RefSegRS 使用的格式）会直接
    导致进程崩溃（无法用 try/except 捕获），此时可切换 OpenCV 作为读取后端：

        auto : 能用 cv2 就用 cv2，否则回落 Pillow
        cv2  : 强制 OpenCV
        pil  : 强制 Pillow
    """
    if backend == 'auto':
        backend = 'cv2' if _cv2_available() else 'pil'

    if backend == 'cv2':
        try:
            import cv2
        except ImportError as e:
            raise ImportError(
                '--image_backend cv2 需要安装 opencv-python：pip install opencv-python') from e
        arr = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
        if arr is None:
            raise IOError(f'cv2 无法读取图像：{path}')
        return Image.fromarray(arr[:, :, ::-1])   # BGR -> RGB

    elif backend == 'pil':
        return Image.open(path).convert('RGB')

    raise ValueError(f'未知的图像读取后端：{backend!r}（可选 auto / cv2 / pil）')


# --------------------------------------------------------------------------- #
# 数据集基类
# --------------------------------------------------------------------------- #
class JsonlReferDataset(data.Dataset):
    """基于 jsonl 标注的指代分割数据集基类。

    子类只需覆写 ``dataset_name`` 以及（可选）``extra_fields`` / ``target_cls``。
    """

    dataset_name = None

    def __init__(self,
                 args,
                 image_transforms=None,
                 target_transforms=None,
                 split='train',
                 eval_mode=False):

        if self.dataset_name is None:
            raise NotImplementedError('子类必须定义 dataset_name')

        self.args = args
        self.split = split
        self.eval_mode = eval_mode
        self.image_transforms = image_transforms
        self.target_transform = target_transforms

        self.max_tokens = getattr(args, 'max_tokens', 20)
        self.image_backend = getattr(args, 'image_backend', 'auto')
        self.image_root = getattr(args, 'data_root', None) or paths.image_root(self.dataset_name)
        self.ann_file = getattr(args, 'ann_file', None) or paths.annotation_file(self.dataset_name, split)

        self.samples = self._load_annotations(self.ann_file)

        # 训练时对一部分样本随机遮挡，做鲁棒性增强
        mask_ratio = getattr(args, 'random_mask_ratio', 0.2)
        num_mask = int(len(self.samples) * mask_ratio)
        self.mask_indices = set(random.sample(range(len(self.samples)), num_mask))

        self.tokenizer = BertTokenizer.from_pretrained(
            getattr(args, 'bert_tokenizer', paths.BERT_TOKENIZER))

    # ------------------------------------------------------------------ #
    # 标注读取
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_annotations(ann_file):
        if not os.path.isfile(ann_file):
            raise FileNotFoundError(
                f'标注文件不存在：{ann_file}\n'
                f'请检查 configs/paths.py 中的 DATAINFO_ROOT 或设置环境变量 PVLMAMBA_DATAINFO_ROOT')
        samples = []
        with open(ann_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    # ------------------------------------------------------------------ #
    # 文本编码
    # ------------------------------------------------------------------ #
    def _tokenize(self, sentence):
        input_ids = self.tokenizer.encode(text=sentence, add_special_tokens=True)[:self.max_tokens]
        attention_mask = [0] * self.max_tokens
        padded = [0] * self.max_tokens
        padded[:len(input_ids)] = input_ids
        attention_mask[:len(input_ids)] = [1] * len(input_ids)
        return torch.tensor(padded), torch.tensor(attention_mask)

    # ------------------------------------------------------------------ #
    # Dataset 协议
    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.samples)

    def get_classes(self):
        return self.target_cls

    def __getitem__(self, index):
        sample = self.samples[index]

        img_path = os.path.join(self.image_root, sample['file_name'])
        img = load_image(img_path, self.image_backend)

        seg = decode_segmentation(sample['segmentation'])
        if seg.shape[:2] != (img.size[1], img.size[0]):
            # 掩码与图像尺寸不一致时按图像尺寸做最近邻重采样
            seg_img = Image.fromarray(seg.astype(np.uint8), mode='P')
            seg_img = seg_img.resize((img.size[0], img.size[1]), resample=Image.NEAREST)
            seg = np.asarray(seg_img)
        target = (seg > 0).astype(np.uint8)

        if self.split == 'train' and index in self.mask_indices:
            img = self._add_random_boxes(img)

        target = Image.fromarray(target, mode='P')

        if self.image_transforms is not None:
            img, target = self.image_transforms(img, target)

        sentence = sample['sent']
        save_prefix = f"{sample.get('image_id', index)}_{sentence}"

        input_ids, attention_mask = self._tokenize(sentence)
        if self.eval_mode:
            input_ids = input_ids.unsqueeze(-1)
            attention_mask = attention_mask.unsqueeze(-1)

        return img, target, input_ids, attention_mask, save_prefix, sentence

    # ------------------------------------------------------------------ #
    # 增强
    # ------------------------------------------------------------------ #
    @staticmethod
    def _add_random_boxes(img, min_num=20, max_num=60, size=32):
        w = h = size
        arr = np.asarray(img).copy()
        img_size = arr.shape[1]
        for _ in range(random.randint(min_num, max_num)):
            y = random.randint(0, max(0, img_size - h))
            x = random.randint(0, max(0, img_size - w))
            arr[y:y + h, x:x + w] = 0
        return Image.fromarray(arr.astype('uint8'), 'RGB')
