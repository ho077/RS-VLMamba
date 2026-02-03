import torch
import torch.utils.data
import utils
import numpy as np
import transforms as T
import random
from bert.modeling_bert import BertModel
import re
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
import datetime
# from lib import segmentation
from lib_RMSIN import segmentation

def seed_everything(seed=567):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_dataset(image_set, transform, args):
    if args.dataset == "rrsisd":
        from data.rrsisd import ReferDataset
    else:
        from data.refsegrs import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=False)
    return ds, ds.target_cls

def extract_target_class_from_text(refer_text, target_cls_set):
    """从指代文本中匹配类别：优先匹配tree，再匹配其他类别"""
    refer_text_lower = refer_text.lower().strip()
    sorted_cls = sorted(target_cls_set, key=lambda x: len(x), reverse=True)
    for cls in sorted_cls:
        if cls.lower() == "tree":  # 已在第一步处理，此处跳过
            continue
        pattern = r'\b' + re.escape(cls.lower()) + r'\b'
        if re.search(pattern, refer_text_lower):
            return cls
    return None

def compute_target_iou(pred_seg, gd_seg):
    """计算指代目标（mask=1区域）的IoU（掩码仅0/1两类）"""
    pred_target = (pred_seg == 1)
    gd_target = (gd_seg == 1)
    
    intersection = np.logical_and(pred_target, gd_target).sum()
    union = np.logical_or(pred_target, gd_target).sum()
    
    has_target = (gd_target.sum() > 0)

    if union == 0 or not has_target:
        return 0.0, has_target
    iou = intersection / union
    return iou, has_target

def evaluate(model, data_loader, bert_model, device, target_cls_set):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    class_metrics = {
        cls: {
            "iou_list": [],
            "sample_count": 0,  # 有效样本数（有目标标注）
            "total_samples": 0   # 该类别总样本数
        } for cls in target_cls_set
    }

    header = 'Test:'
    total_samples = 0
    unclassified_samples = 0
    start_time = time.time()

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            image, target, sentences, attentions, save_prefix, refer_text = data
            
            image = image.to(device)
            target = target.to(device)
            sentences = sentences.to(device)
            attentions = attentions.to(device)
            target = target.to(device)


            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)

            refer_text = refer_text[0]

            if bert_model is not None:
                tensor_embeddings = sentences
                attention_mask = attentions

                if len(tensor_embeddings.shape) == 2:
                    tensor_embeddings = tensor_embeddings.unsqueeze(0)
                if len(attention_mask.shape) == 2:
                    attention_mask = attention_mask.unsqueeze(0)
                
                last_hidden_states = bert_model(tensor_embeddings, attention_mask=attention_mask)[0]
                embedding = last_hidden_states.permute(0, 2, 1)
                output = model(image, embedding, l_mask=attention_mask.unsqueeze(-1))
            else:
                output = model(image, sentences, attentions)
                # output = model(image, sentences, attentions, target_masks, position_masks)

            pred_mask = output.argmax(1).cpu().data.numpy()[0]
            gd_mask = target.cpu().data.numpy()[0]

            target_cls = extract_target_class_from_text(refer_text, target_cls_set)
            total_samples += 1
            
            if target_cls is None:
                unclassified_samples += 1
            else:
                iou, has_target = compute_target_iou(pred_mask, gd_mask)
                metrics = class_metrics[target_cls]
                metrics["total_samples"] += 1

                if has_target:
                    metrics["iou_list"].append(iou)
                    metrics["sample_count"] += 1

            del image, target, sentences, attentions, output, pred_mask, gd_mask
            if bert_model is not None:
                del tensor_embeddings, attention_mask, last_hidden_states, embedding

    print('='*90)
    print('Referential Remote Sensing Segmentation - Per-Class Evaluation Results')
    print('='*90)

    all_class_miou = []
    for cls in target_cls_set:
        metrics = class_metrics[cls]
        print(f'\n【Semantic Class: {cls}】')
        print(f'  Total Samples: {metrics["total_samples"]} | Valid Samples: {metrics["sample_count"]}')
        
        if metrics["sample_count"] == 0:
            print(f'  Mean IoU: 0.00% | No valid samples')
            continue

        miou = np.mean(metrics["iou_list"]) * 100
        all_class_miou.append(miou)
        print(f'  Mean IoU: {miou:.2f}%')

    print('\n' + '='*90)
    print('Overall Evaluation Results')
    print('='*90)
    if all_class_miou:
        overall_miou = np.mean(all_class_miou)
        print(f'Overall Mean IoU (all classes): {overall_miou:.2f}%')
    else:
        print(f'Overall Mean IoU: 0.00%')
    print(f'Total Test Samples: {total_samples}')
    print(f'Unclassified Samples (no matched class): {unclassified_samples}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'\nTotal Test Time: {total_time_str}')
    print(f'Average Time per Sample: {total_time / total_samples:.2f} seconds')

def get_transform(args):
    """仅保留图像预处理，移除可视化相关逻辑"""
    transforms = [
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    return T.Compose(transforms)

def main(args):
    device = torch.device(args.device)
    dataset_test, target_cls_set = get_dataset('test', get_transform(args), args)
    print(f"Evaluating on {len(target_cls_set)} semantic classes: {sorted(target_cls_set)}")
    
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=args.workers
    )

    print(f"Loading Model: {args.model}")
    
    single_model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'], strict=False)
    model = single_model.to(device)

    bert_model = None
    if args.model != 'lavt_one':
        single_bert_model = BertModel.from_pretrained(args.ck_bert)
        if args.ddp_trained_weights:
            single_bert_model.pooler = None
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)

    evaluate(model, data_loader_test, bert_model, device, target_cls_set)

if __name__ == "__main__":
    from args import get_parser
    seed_everything()
    parser = get_parser()
    args = parser.parse_args()
    print(f'Image Size: {args.img_size}x{args.img_size}')
    main(args)