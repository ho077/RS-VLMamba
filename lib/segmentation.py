"""模型工厂。

对外暴露两个入口：
    - ``lavt``     : MultiModalSwinTransformer 骨干 + TCMD 掩码预测头（文本特征外部注入）
    - ``lavt_one`` : MambaVision 骨干 + Mamba 解码器（文本编码器内置，PVLMamba 主分支）

说明：
    - 骨干网络权重路径一律由 ``args`` 注入，本文件不含任何硬编码路径；
    - 解码器的具体实现位于 ``lib/decoder.py``，当前为接口骨架；
    - 结构超参默认取自 ``configs/training.py`` 的 ``MODEL`` 配置（占位值）。
"""

import torch
import torch.nn as nn

from configs import paths
from configs import training as train_cfg

from ._utils import LAVT, LAVTOne
from .backbone import MultiModalSwinTransformer
from .mask_predictor import TCMD
from .decoder import MambaDecoder


__all__ = ['lavt', 'lavt_one']


#: Swin 各变体的结构参数
SWIN_CFG = {
    'tiny': dict(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]),
    'small': dict(embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24]),
    'base': dict(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    'large': dict(embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48]),
}


def _parse_mha(args):
    """解析 --mha 参数，形如 "4-4-4-4"，留空则全 1。"""
    if getattr(args, 'mha', ''):
        return [int(a) for a in args.mha.split('-')]
    return [1, 1, 1, 1]


def _swin_window_size(pretrained, args):
    if 'window12' in str(pretrained) or getattr(args, 'window12', False):
        print('Window size 12!')
        return 12
    return 7


# --------------------------------------------------------------------------- #
# LAVT: 外部文本特征 + Swin 骨干
# --------------------------------------------------------------------------- #
def _segm_lavt(pretrained, args):
    cfg = SWIN_CFG[args.swin_type]
    backbone = MultiModalSwinTransformer(
        embed_dim=cfg['embed_dim'],
        depths=cfg['depths'],
        num_heads=cfg['num_heads'],
        window_size=_swin_window_size(pretrained, args),
        ape=False, drop_path_rate=0.3, patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False,
        num_heads_fusion=_parse_mha(args),
        fusion_drop=args.fusion_drop,
    )
    if pretrained:
        print('Initializing Multi-modal Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize Multi-modal Swin Transformer weights.')
        backbone.init_weights()

    classifier = TCMD(8 * cfg['embed_dim'])
    return LAVT(backbone, classifier)


def lavt(pretrained='', args=None):
    return _segm_lavt(pretrained, args)


# --------------------------------------------------------------------------- #
# LAVT-One: 文本编码器内置 + MambaVision 骨干 + Mamba 解码器
# --------------------------------------------------------------------------- #
def _segm_lavt_one(pretrained, args):
    from .MVbackbone import mamba_vision_L2_512_21k

    cfg = train_cfg.MODEL
    decoder_cfg = dict(cfg['decoder'])

    weights = (getattr(args, 'pretrained_mambavision_weights', '')
               or paths.PRETRAINED_MAMBAVISION_WEIGHTS)
    backbone = mamba_vision_L2_512_21k(
        pretrained=True,
        num_classes=cfg['num_classes'],
        model_path=weights or None,
    )

    # 解码器通道数未显式指定时，直接沿用骨干各阶段的输出通道数
    num_features = getattr(backbone, 'num_features', None)
    dims = decoder_cfg.get('dims') or num_features
    embed_dim = decoder_cfg.get('embed_dim') or (num_features[0] if num_features else None)

    if dims is None or embed_dim is None:
        raise ValueError(
            '无法推断解码器通道数，请在 configs/training.py 的 MODEL["decoder"] 中'
            '显式填写 dims 与 embed_dim')

    classifier = MambaDecoder(
        num_layers=decoder_cfg['num_layers'],
        depths=decoder_cfg['depths'],
        dims=dims,
        d_state=decoder_cfg['d_state'],
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=decoder_cfg['drop_path_rate'],
        norm_layer=nn.LayerNorm,
        use_checkpoint=False,
        embed_dim=embed_dim,
        num_classes=cfg['num_classes'],
    )

    if getattr(classifier, 'is_implemented', True) is False:
        print('[PVLMamba] 注意：MambaDecoder 仅提供接口骨架，前向计算尚未实现'
              '（详见 lib/decoder.py）。')

    return LAVTOne(backbone, classifier, args)


def lavt_one(pretrained='', args=None):
    return _segm_lavt_one(pretrained, args)
