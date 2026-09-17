"""PVLMamba 评估入口（单卡）。

用法示例（超参与路径均显式给出，脚本本身不含任何硬编码路径）：

    python evaluate.py --dataset rrsisd --split test \
        --resume outputs/checkpoints/model_best_pvlmamba.pth \
        --img_size 480 --device cuda:0

说明：
    - 数据来自 RSRefSeg2 的 datainfo jsonl；
    - 输出整体 IoU / oIoU / precision@k，并在标注含 category_name 时给出类别级 IoU；
    - 解码器当前为接口骨架，前向计算尚未实现。
"""

import datetime
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data

import transforms as T
import utils
from args import parse_args
from bert.modeling_bert import BertModel
from data import build_dataset
from lib import segmentation


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_transform(args):
    return T.Compose([
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def compute_iou(pred_mask, gt_mask):
    """二值掩码 IoU（pred/gt 均为 {0,1} 的 numpy 数组）。"""
    pred = pred_mask == 1
    gt = gt_mask == 1
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(pred, gt).sum()) / float(union)


def build_model(args, device):
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
    if not args.resume or not os.path.isfile(args.resume):
        raise SystemExit(
            f'请通过 --resume 指定已训练好的权重文件（当前为：{args.resume!r}）')
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    model.eval()

    bert_model = None
    if args.model != 'lavt_one':
        bert_model = BertModel.from_pretrained(args.ck_bert)
        if args.ddp_trained_weights:
            bert_model.pooler = None
        if 'bert_model' in checkpoint:
            bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = bert_model.to(device)
        bert_model.eval()
    return model, bert_model


@torch.no_grad()
def evaluate(model, data_loader, bert_model, device, category_names=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int64)
    seg_total = 0
    cum_i, cum_u = 0.0, 0.0
    iou_list = []
    per_category = defaultdict(list)

    start_time = time.time()
    sample_idx = 0

    for data in metric_logger.log_every(data_loader, 100, header):
        image, target, sentences, attentions, _, _ = data
        image = image.to(device)
        target = target.to(device)
        sentences = sentences.to(device)
        attentions = attentions.to(device)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        if bert_model is not None:
            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
            embedding = last_hidden_states.permute(0, 2, 1)
            output = model(image, embedding, l_mask=attentions.unsqueeze(-1))
        else:
            output = model(image, sentences, l_mask=attentions)

        pred = output.argmax(1).cpu().numpy()
        gt = target.cpu().numpy()

        for b in range(pred.shape[0]):
            iou = compute_iou(pred[b], gt[b])
            iou_list.append(iou)
            for k, thr in enumerate(eval_seg_iou_list):
                seg_correct[k] += int(iou >= thr)
            seg_total += 1

            pred_bin, gt_bin = (pred[b] == 1), (gt[b] == 1)
            cum_i += float(np.logical_and(pred_bin, gt_bin).sum())
            cum_u += float(np.logical_or(pred_bin, gt_bin).sum())

        if category_names is not None:
            for b in range(pred.shape[0]):
                name = category_names[sample_idx] if sample_idx < len(category_names) else None
                if name:
                    per_category[name].append(compute_iou(pred[b], gt[b]))
                sample_idx += 1

    miou = float(np.mean(iou_list)) if iou_list else 0.0
    oiou = cum_i / cum_u if cum_u > 0 else 0.0

    print('=' * 72)
    print('PVLMamba Evaluation Results')
    print('=' * 72)
    print(f'  Mean IoU   : {miou * 100:.2f}')
    print(f'  Overall IoU: {oiou * 100:.2f}')
    for k, thr in enumerate(eval_seg_iou_list):
        print(f'  precision@{thr}: {seg_correct[k] * 100. / max(seg_total, 1):.2f}')
    print(f'  Samples    : {seg_total}')

    if per_category:
        print('-' * 72)
        print('Per-category mean IoU')
        cat_mious = []
        for name in sorted(per_category):
            v = float(np.mean(per_category[name])) * 100
            cat_mious.append(v)
            print(f'  {name:<28} {v:6.2f}  (n={len(per_category[name])})')
        if cat_mious:
            print(f'  {"mean over categories":<28} {np.mean(cat_mious):6.2f}')

    total_time = time.time() - start_time
    print('-' * 72)
    print(f'Total time: {datetime.timedelta(seconds=int(total_time))}')
    print('=' * 72)

    return {'mIoU': miou * 100, 'oIoU': oiou * 100,
            'precision': {str(t): seg_correct[k] * 100. / max(seg_total, 1)
                          for k, t in enumerate(eval_seg_iou_list)},
            'per_category': {k: float(np.mean(v)) * 100 for k, v in per_category.items()},
            'num_samples': seg_total}


def main(args):
    device = torch.device(args.device)
    transform = get_transform(args)
    dataset = build_dataset(args.dataset, args.split, transform, args, eval_mode=True)
    print(f'Evaluating {args.dataset} split={args.split}, {len(dataset)} samples')
    print(f'  images    : {dataset.image_root}')
    print(f'  annotation: {dataset.ann_file}')

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        sampler=torch.utils.data.SequentialSampler(dataset),
        num_workers=args.workers)

    # 若标注里带类别名，则额外做类别级统计
    category_names = [s.get('category_name') for s in dataset.samples]

    model, bert_model = build_model(args, device)
    results = evaluate(model, data_loader, bert_model, device, category_names)

    if getattr(args, 'save_results', ''):
        os.makedirs(os.path.dirname(os.path.abspath(args.save_results)) or '.', exist_ok=True)
        with open(args.save_results, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'Results written to {args.save_results}')

    return results


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    main(args)
