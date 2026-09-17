"""PVLMamba 训练入口（单卡）。

说明：
    - 训练超参默认是占位值，必须在命令行显式给出（见 configs/training.py）；
    - 实验跟踪后端可选：none / swanlab / wandb，默认 none；
    - 数据来自 RSRefSeg2 的 datainfo jsonl（RefSegRS / RRSIS-D）；
    - 解码器当前为接口骨架，前向计算尚未实现，训练会在 forward 处明确报错。
"""

import datetime
import gc
import operator
import os
import random
import time
from functools import reduce

import numpy as np
import torch
import torch.utils.data

import transforms as T
import utils
from args import parse_args
from bert.modeling_bert import BertModel
from configs import paths
from data import build_dataset
from lib import segmentation
from loss.loss import Loss


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_transform(args):
    return T.Compose([
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def criterion(input, target, weight=0.1):
    return Loss(weight=weight)(input, target)


def IoU(pred, gt):
    pred = pred.argmax(1)
    intersection = torch.sum(torch.mul(pred, gt))
    union = torch.sum(torch.add(pred, gt)) - intersection
    if intersection == 0 or union == 0:
        iou = 0
    else:
        iou = float(intersection) / float(union)
    return iou, intersection, union


class _NoopLogger:
    """不落盘的日志后端，避免在未配置实验跟踪时引入额外依赖。"""

    def log(self, *_args, **_kwargs):
        pass


def build_logger(args):
    backend = (getattr(args, 'log_backend', 'none') or 'none').lower()
    if backend == 'none':
        return _NoopLogger()
    if backend == 'swanlab':
        import swanlab
        swanlab.init(project=args.log_project, name=args.log_run_name)
        return swanlab
    if backend == 'wandb':
        import wandb
        wandb.init(project=args.log_project, name=args.log_run_name)
        return wandb
    raise ValueError(f'未知的日志后端：{backend}（可选 none / swanlab / wandb）')


def check_required_args(args):
    missing = [name for name in ('epochs', 'lr', 'weight_decay', 'batch_size')
               if getattr(args, name, None) is None]
    if missing:
        raise SystemExit(
            '以下训练超参尚未指定：' + ', '.join('--' + m.replace('_', '-') for m in missing) +
            '\n请在命令行显式传入，或先填写 configs/training.py 中的占位配置。')


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #
def evaluate(model, data_loader, bert_model, logger, epoch=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "
    total_its = 0
    acc_ious = 0

    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    total_loss = 0

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            total_its += 1
            image, target, sentences, attentions, _, _ = data
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            sentences = sentences.cuda(non_blocking=True)
            attentions = attentions.cuda(non_blocking=True)

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            if bert_model is not None:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l)
                attentions = attentions.unsqueeze(dim=-1)        # (B, N_l, 1)
                output = model(image, embedding, l_mask=attentions)
            else:
                output = model(image, sentences, l_mask=attentions)

            iou, I, U = IoU(output, target)
            loss = criterion(output, target)
            total_loss += loss.item()
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U
            for n_eval_iou in range(len(eval_seg_iou_list)):
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou_list[n_eval_iou])
            seg_total += 1
        iou = acc_ious / max(total_its, 1)

    mIoU = np.mean(mean_IoU) if mean_IoU else 0.0
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % (
            str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / max(seg_total, 1))
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / max(cum_U, 1))
    print(results_str)

    logger.log({
        'val mIoU': mIoU * 100.,
        'val oiou': cum_I * 100. / max(cum_U, 1),
        'val Loss': total_loss / max(total_its, 1),
    })

    return 100 * iou, 100 * cum_I / max(cum_U, 1)


# --------------------------------------------------------------------------- #
# 训练一个 epoch
# --------------------------------------------------------------------------- #
def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model, logger):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0
    total_its = 0

    for i, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        total_its += 1
        image, target, sentences, attentions, _, _ = data

        image = image.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        sentences = sentences.cuda(non_blocking=True)
        attentions = attentions.cuda(non_blocking=True)

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)

        if bert_model is not None:
            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
            embedding = last_hidden_states.permute(0, 2, 1)
            attentions = attentions.unsqueeze(dim=-1)
            output = model(image, embedding, attentions)
        else:
            output = model(image, sentences, attentions)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        logger.log({'Train Loss': train_loss / total_its})

        del image, target, sentences, attentions, loss, output, data
        if bert_model is not None:
            del last_hidden_states, embedding

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def build_optimizer(single_model, single_bert_model, args):
    """按参数类型分组：骨干的 norm/位置编码不做 weight decay。"""
    backbone_no_decay, backbone_decay = [], []
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    params_to_optimize = [
        {'params': backbone_no_decay, 'weight_decay': 0.0},
        {'params': backbone_decay},
        {'params': [p for p in single_model.classifier.parameters() if p.requires_grad]},
    ]

    # 参与微调的 BERT 层（层数由 --bert_trainable_layers 指定，占位默认 0）
    n_layers = max(0, getattr(args, 'bert_trainable_layers', 0) or 0)
    if n_layers > 0:
        if single_bert_model is not None:
            encoder_layers = single_bert_model.encoder.layer
        else:
            encoder_layers = single_model.text_encoder.encoder.layer
        selected = [p for i in range(min(n_layers, len(encoder_layers)))
                    for p in encoder_layers[i].parameters() if p.requires_grad]
        params_to_optimize.append({'params': selected})

    return torch.optim.AdamW(params_to_optimize, lr=args.lr,
                             weight_decay=args.weight_decay, amsgrad=args.amsgrad)


def main(args):
    check_required_args(args)
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = build_logger(args)

    transform = get_transform(args)
    dataset = build_dataset(args.dataset, 'train', transform, args)
    dataset_test = build_dataset(args.dataset, 'test', transform, args, eval_mode=True)
    num_classes = 2

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers)

    print(args.model)
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
    model.cuda()
    single_model = model

    if args.model != 'lavt_one':
        bert_model = BertModel.from_pretrained(args.ck_bert)
        bert_model.pooler = None
        bert_model.cuda()
        single_bert_model = bert_model
    else:
        bert_model = None
        single_bert_model = None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'], strict=False)
        if args.model != 'lavt_one':
            single_bert_model.load_state_dict(checkpoint['bert_model'])
    else:
        checkpoint = None

    optimizer = build_optimizer(single_model, single_bert_model, args)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9)

    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1
    resume_epoch = checkpoint['epoch'] if checkpoint is not None and 'epoch' in checkpoint else -999
    if checkpoint is not None:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    for epoch in range(max(0, resume_epoch + 1), args.epochs):
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, epoch,
                        args.print_freq, iterations, bert_model, logger)
        iou, overallIoU = evaluate(model, data_loader_test, bert_model, logger, epoch)
        print('Average object IoU {}'.format(iou))
        print('Overall IoU {}'.format(overallIoU))

        dict_to_save = {'model': single_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': args,
                        'lr_scheduler': lr_scheduler.state_dict()}
        if single_bert_model is not None:
            dict_to_save['bert_model'] = single_bert_model.state_dict()

        if best_oIoU < overallIoU:
            print('Better epoch: {}\n'.format(epoch))
            torch.save(dict_to_save, os.path.join(
                args.output_dir, 'model_best_{}.pth'.format(args.model_id)))
            best_oIoU = overallIoU
        torch.save(dict_to_save, os.path.join(
            args.output_dir, 'model_last_{}.pth'.format(args.model_id)))

    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))
    print('Checkpoints saved to {}'.format(os.path.abspath(args.output_dir)))


if __name__ == "__main__":
    main(parse_args())
