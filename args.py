"""命令行参数定义。

约定：
    - 所有路径类参数默认留空，交给 ``configs/paths.py`` 统一解析；
    - 训练超参默认从 ``configs/training.py`` 的占位配置读取（已做模糊处理），
      实际实验请在命令行显式指定。
"""

import argparse

from configs import paths
from configs import training as train_cfg


def get_parser():
    parser = argparse.ArgumentParser(description='PVLMamba training / testing / evaluation')

    # ---------------- 数据 ---------------- #
    parser.add_argument('--dataset', default=train_cfg.DATA['dataset'], help='refsegrs | rrsisd')
    parser.add_argument('--img_size', default=train_cfg.DATA['img_size'], type=int,
                        help='input image size')
    parser.add_argument('--max_tokens', default=train_cfg.DATA['max_tokens'], type=int,
                        help='指代文本最大 token 数')
    parser.add_argument('--data_root', default='',
                        help='图像根目录（默认取 configs/paths.py 中对应数据集的设置）')
    parser.add_argument('--ann_file', default='',
                        help='标注 jsonl 路径（默认取 configs/paths.py 拼出的 datainfo 路径）')
    parser.add_argument('--random_mask_ratio', default=0.2, type=float,
                        help='训练时随机遮挡的样本比例')
    parser.add_argument('--image_backend', default='auto', choices=('auto', 'cv2', 'pil'),
                        help='图像读取后端：auto 优先 cv2；某些环境下 Pillow 读 PackBits '
                             '压缩的 TIFF（RefSegRS）会导致进程崩溃，此时请用 cv2')
    parser.add_argument('--split', default='test', help='评估/测试使用的切分')
    parser.add_argument('--splitBy', default='unc',
                        help='兼容旧 COCO 风格数据集的划分方式，jsonl 流程下不生效')

    # ---------------- 文本编码器 ---------------- #
    parser.add_argument('--bert_tokenizer', default=paths.BERT_TOKENIZER,
                        help='BERT tokenizer（HuggingFace 模型标识）')
    parser.add_argument('--ck_bert', default=paths.BERT_WEIGHTS,
                        help='预训练 BERT 权重（HuggingFace 模型标识）')

    # ---------------- 模型 ---------------- #
    parser.add_argument('--model', default=train_cfg.MODEL['arch'],
                        help='lavt_one（文本编码器内置） | lavt（外部文本特征）')
    parser.add_argument('--model_id', default='pvlmamba', help='用于标识本次实验的名称')
    parser.add_argument('--swin_type', default='base',
                        help='tiny, small, base, large（走 lavt 分支时的 Swin 变体）')
    parser.add_argument('--window12', action='store_true',
                        help='以 window size 12 初始化 Swin（加载的权重文件名含 window12 时需要）')
    parser.add_argument('--mha', default=train_cfg.MODEL['mha'],
                        help='融合头注意力头数，形如 4-4-4-4，留空用默认')
    parser.add_argument('--fusion_drop', default=train_cfg.MODEL['fusion_drop'], type=float,
                        help='跨模态融合模块的 dropout')
    parser.add_argument('--pretrained_swin_weights', default=paths.PRETRAINED_SWIN_WEIGHTS,
                        help='Swin 预训练权重路径（留空表示随机初始化）')
    parser.add_argument('--pretrained_mambavision_weights', default=paths.PRETRAINED_MAMBAVISION_WEIGHTS,
                        help='MambaVision 预训练权重路径（留空表示自动下载到缓存目录）')
    parser.add_argument('--ddp_trained_weights', action='store_true',
                        help='加载的权重是否来自 DDP 训练（影响 BERT pooler 的处理）')

    # ---------------- 训练超参（占位，需显式指定） ---------------- #
    parser.add_argument('-b', '--batch-size', default=train_cfg.DATA['batch_size'], type=int,
                        help='训练批大小（占位，需指定）')
    parser.add_argument('--epochs', default=train_cfg.OPTIM['epochs'], type=int,
                        help='训练轮数（占位，需指定）')
    parser.add_argument('--lr', default=train_cfg.OPTIM['lr'], type=float,
                        help='初始学习率（占位，需指定）')
    parser.add_argument('--wd', '--weight-decay', default=train_cfg.OPTIM['weight_decay'],
                        type=float, metavar='W', dest='weight_decay',
                        help='权重衰减（占位，需指定）')
    parser.add_argument('--amsgrad', action='store_true',
                        help='AdamW 是否启用 amsgrad')
    parser.add_argument('--bert_trainable_layers', default=train_cfg.OPTIM['bert_trainable_layers'],
                        type=int, help='参与微调的 BERT 层数（占位）')
    parser.add_argument('--seed', default=train_cfg.SEED, type=int, help='随机种子')

    # ---------------- 运行环境 ---------------- #
    parser.add_argument('--device', default='cuda:0', help='单卡评估时使用的设备')
    parser.add_argument("--local_rank", type=int, default=0, help='DDP 的 local rank')
    parser.add_argument('-j', '--workers', default=train_cfg.DATA['num_workers'], type=int,
                        metavar='N', help='dataloader worker 数')
    parser.add_argument('--pin_mem', action='store_true',
                        help='dataloader 是否使用 pinned memory')

    # ---------------- 输出与日志 ---------------- #
    parser.add_argument('--output-dir', default=paths.CHECKPOINT_DIR,
                        help='checkpoint 保存目录')
    parser.add_argument('--resume', default='', help='从指定 checkpoint 继续/开始评估')
    parser.add_argument('--save_results', default='',
                        help='评估结果(json)输出路径，留空则不落盘')
    parser.add_argument('--visual_dir', default=paths.VISUAL_DIR,
                        help='可视化结果输出目录')
    parser.add_argument('--log_backend', default=train_cfg.LOGGING['backend'],
                        help='none | swanlab | wandb')
    parser.add_argument('--log_project', default=train_cfg.LOGGING['project'],
                        help='实验跟踪项目名（占位）')
    parser.add_argument('--log_run_name', default=train_cfg.LOGGING['run_name'],
                        help='实验跟踪运行名（占位）')
    parser.add_argument('--print-freq', default=train_cfg.LOGGING['print_freq'], type=int,
                        help='日志打印频率')

    return parser


def parse_args(argv=None):
    return get_parser().parse_args(argv)


if __name__ == "__main__":
    parse_args()
