import cv2
import torch
import torch.utils.data
import utils
import numpy as np
import transforms as T
from torchvision.transforms import functional as F
import random
from bert.modeling_bert import BertModel

from model import segmentation

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
import datetime

def get_dataset(image_set, transform, args):
    if args.dataset == "rrsisd":
        from data.rrsisd import ReferDataset
    else:
        from data.refsegrs import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=False
                      )
    num_classes = 2
    return ds, num_classes

def evaluate(model, data_loader, bert_model, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'
    save_dir = ""

    start_time = time.time()

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header):
            # 提取指代文本（适配数据集返回格式，最后一个元素为texts）
            image, target, sentences, attentions,save_prefix, texts = data
            image = image.to(device)
            target = target.to(device)
            sentences = sentences.to(device)
            attentions = attentions.to(device)



            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            target = target.cpu().data.numpy()
            for j in range(sentences.size(0)):
                if bert_model is not None:
                    last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                    embedding = last_hidden_states.permute(0, 2, 1)
                    output = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
                else:
                    # output = model(image, sentences, attentions, target_masks, position_masks)
                    output = model(image, sentences, attentions)
                    
                output = output.cpu()
                output_mask = output.argmax(1).data.numpy()
                I, U = computeIoU(output_mask, target)
                
                save_path = os.path.join(save_dir, str(seg_total+1))
                # 传递指代文本到保存函数
                refer_text = texts[0] if isinstance(texts, (list, tuple)) else str(texts)
                # print(save_prefix[0])
                save_pred_targ_results(output_mask, target, image, output, save_path, save_prefix[0], refer_text)
                
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)

                seg_total += 1

            del image, target, sentences, attentions, output, output_mask
            if bert_model is not None:
                del last_hidden_states, embedding

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total test time {}'.format(total_time_str))
    print('Test time for one image %.2f ' % (total_time / seg_total))

def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]
    return T.Compose(transforms)

def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))
    return I, U



def save_pred_targ_results(output_mask, target, image, output_prob, save_path, name_address, refer_text):

    os.makedirs(save_path, exist_ok=True)
    
    pred = output_mask[0, :, :]
    targ = target[0, :, :]
    
    # 反归一化图像
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1 / s for s in std]
    im = F.normalize(image, mean=inv_mean, std=inv_std)
    im = im[0, :, :, :].cpu().detach().numpy()
    im = im.transpose([1, 2, 0])  # 此时im是RGB格式
    im = np.uint8(im * 255)
    
    # --- 核心修正：统一转换为OpenCV的BGR格式 ---
    im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    # 调暗BGR格式的原图（保持亮度调整逻辑）
    brightness_coeff = 0.7
    im_dark_bgr = np.uint8(im_bgr * brightness_coeff)
    
    # --- 保存指代文本为txt文件 ---
    text_save_path = os.path.join(save_path, name_address + "_refer_text.txt")
    with open(text_save_path, 'w', encoding='utf-8') as f:
        f.write("指代文本：\n")
        f.write(str(refer_text).strip())
    
    # # --- 1. 调暗后的图像 + 半透明红色预测掩码 + 蓝色轮廓（BGR格式） ---
    # pred_red_mask = np.zeros_like(im_bgr)
    # pred_red_mask[:, :] = [0, 0, 255]  # BGR格式的红色（正确）
    # pred_mask_3c = np.repeat(pred[:, :, np.newaxis], 3, axis=-1)
    # pred_masked = np.uint8(pred_red_mask * pred_mask_3c)
    # pred_overlay = cv2.addWeighted(im_dark_bgr, 1.0, pred_masked, 0.5, 0)
    
    # # ===== 新增：为预测掩码添加蓝色轮廓 =====
    # # 1. 将单通道预测掩码转换为8位灰度图（用于轮廓检测）
    # pred_8bit = np.uint8(pred * 255)
    # # 2. 查找轮廓（只找最外层轮廓）
    # pred_contours, _ = cv2.findContours(pred_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # 3. 绘制蓝色轮廓（BGR格式：[255,0,0]是蓝色，厚度和真实标签保持一致）
    # contour_thickness = 2
    # cv2.drawContours(pred_overlay, pred_contours, -1, (255, 0, 0), contour_thickness)
    # # ==========================================
    
    # cv2.imwrite(os.path.join(save_path, name_address + "_pred_mask.png"), pred_overlay)

    # # --- 2. 调暗后的图像 + 半透明红色真实标签掩码 + 蓝色轮廓（BGR格式） ---
    # targ_red_mask = np.zeros_like(im_bgr)
    # targ_red_mask[:, :] = [0, 0, 255]  # BGR格式的红色（正确）
    # targ_mask_3c = np.repeat(targ[:, :, np.newaxis], 3, axis=-1)
    # targ_masked = np.uint8(targ_red_mask * targ_mask_3c)
    # targ_overlay = cv2.addWeighted(im_dark_bgr, 1.0, targ_masked, 0.5, 0)
    
    # # ===== 真实标签掩码的蓝色轮廓（原有逻辑保留） =====
    # # 1. 将单通道掩码转换为8位灰度图（用于轮廓检测）
    # targ_8bit = np.uint8(targ * 255)
    # # 2. 查找轮廓（只找最外层轮廓）
    # contours, _ = cv2.findContours(targ_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # 3. 绘制蓝色轮廓（BGR格式：[255,0,0]是蓝色，厚度设为2，可根据需要调整）
    # contour_thickness = 2
    # cv2.drawContours(targ_overlay, contours, -1, (255, 0, 0), contour_thickness)
    # # ==========================================
    
    # cv2.imwrite(os.path.join(save_path, name_address + "_targ_mask.png"), targ_overlay)
    

    # # --- 3. 保存纯掩码图像（单通道，无需通道转换） ---
    # # 保存预测掩码：pred中1的区域设为255，0的区域设为0
    # pred_pure_mask = np.uint8(pred * 255)
    # cv2.imwrite(os.path.join(save_path, name_address + "_pred_pure_mask.png"), pred_pure_mask)
    # # 保存真实标签掩码：targ中1的区域设为255，0的区域设为0
    # targ_pure_mask = np.uint8(targ * 255)
    # cv2.imwrite(os.path.join(save_path, name_address + "_targ_pure_mask.png"), targ_pure_mask)

    # # --- 4 & 5. 热力图相关（修正通道匹配问题） ---
    if output_prob is not None:
        # 获取正类（类别1）的概率图
        prob_map = torch.softmax(output_prob, dim=1)[0, 1]
        prob_map_np = prob_map.cpu().numpy()
        
        # 归一化到0-255
        prob_min = np.min(prob_map_np)
        prob_max = np.max(prob_map_np)
        if prob_max - prob_min > 1e-6:
            prob_normalized = ((prob_map_np - prob_min) / (prob_max - prob_min) * 255).astype(np.uint8)
        else:
            prob_normalized = (prob_map_np * 255).astype(np.uint8)
        
        # 应用热力图颜色映射（输出为BGR格式）
        heatmap = cv2.applyColorMap(prob_normalized, cv2.COLORMAP_TURBO)
        
        # 调整大小（匹配BGR原图尺寸）
        heatmap = cv2.resize(heatmap, (im_bgr.shape[1], im_bgr.shape[0]))
        
        # 保存纯热力图
        cv2.imwrite(os.path.join(save_path, name_address + "_heatmap.png"), heatmap)
        
        # 保存热力图叠加图（使用BGR格式的调暗原图，通道匹配）
        heatmap_overlay = cv2.addWeighted(im_dark_bgr, 0.5, heatmap, 0.5, 0)
        cv2.imwrite(os.path.join(save_path, name_address + "_heatmap_overlay.png"), heatmap_overlay)

def main(args):
    device = torch.device(args.device)
    dataset_test, _ = get_dataset('test', get_transform(args=args), args)

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    print(args.model)
    single_model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights,args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'], strict=False)
    model = single_model.to(device)
    model.eval()
    if args.model != 'lavt_one':
        model_class = BertModel
        single_bert_model = model_class.from_pretrained(args.ck_bert)
        if args.ddp_trained_weights:
            single_bert_model.pooler = None
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)
    else:
        bert_model = None

    evaluate(model, data_loader_test, bert_model, device=device)

if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)