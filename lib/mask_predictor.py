"""掩码预测头 / 跨模态融合头。

包含：
    - Transformer_Fusion: 视觉-语言跨模态 Transformer 融合（轻量包装）
    - TCMD: 基于自适应旋转卷积（ARC）的渐进式多尺度掩码预测头

注意：Mamba 解码器的骨架位于 ``lib/decoder.py``，
其具体实现按要求留空，本文件不涉及解码器内部细节。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

from arc import AdaptiveRotatedConv2d, RountingFunction


__all__ = ['Transformer_Fusion', 'TCMD']


class Transformer_Fusion(nn.Module):
    """以一个 TransformerDecoderLayer 完成视觉特征对语言特征的 cross-attention。"""

    def __init__(self, dim=768, nhead=8, num_layers=1):
        super(Transformer_Fusion, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=dim, nhead=nhead)
        self.transformer_model = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

    def forward(self, v, lan):
        W, H = v.shape[2], v.shape[3]
        v = v.view(v.shape[0], v.shape[1], -1)
        v = v.permute(2, 0, 1)
        l = lan.permute(2, 0, 1)
        v = self.transformer_model(v, l)
        v = v.permute(1, 2, 0)
        v = v.view(v.shape[0], v.shape[1], W, H)
        return v


class TCMD(nn.Module):
    """多尺度渐进式掩码预测头：自 c4 向上融合至 c1，逐级注入语言特征。

    每一级结构为：
        conv3x3 -> BN -> ReLU -> AdaptiveRotatedConv2d -> BN -> ReLU
        随后与语言特征做 Transformer cross-attention 融合
    """

    def __init__(self, c4_dims, factor=2):
        super(TCMD, self).__init__()

        lan_size = 768
        hidden_size = lan_size
        c4_size = c4_dims
        c3_size = c4_dims // (factor ** 1)
        c2_size = c4_dims // (factor ** 2)
        c1_size = c4_dims // (factor ** 3)

        self.adpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv1_4 = nn.Conv2d(c4_size + c3_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU()
        routing_function1 = RountingFunction(in_channels=hidden_size, kernel_number=1)
        self.conv2_4 = AdaptiveRotatedConv2d(in_channels=hidden_size, out_channels=hidden_size,
                                             kernel_size=3, padding=1, rounting_func=routing_function1,
                                             bias=False, kernel_number=1)
        self.bn2_4 = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU()

        self.transformer_fusion1 = Transformer_Fusion(dim=768, nhead=8, num_layers=1)

        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU()
        routing_function2 = RountingFunction(in_channels=hidden_size, kernel_number=1)
        self.conv2_3 = AdaptiveRotatedConv2d(in_channels=hidden_size, out_channels=hidden_size,
                                             kernel_size=3, padding=1, rounting_func=routing_function2,
                                             bias=False, kernel_number=1)
        self.bn2_3 = nn.BatchNorm2d(hidden_size)
        self.relu2_3 = nn.ReLU()
        self.transformer_fusion2 = Transformer_Fusion(dim=768, nhead=8, num_layers=1)

        self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU()
        routing_function3 = RountingFunction(in_channels=hidden_size, kernel_number=1)
        self.conv2_2 = AdaptiveRotatedConv2d(in_channels=hidden_size, out_channels=hidden_size,
                                             kernel_size=3, padding=1, rounting_func=routing_function3,
                                             bias=False, kernel_number=1)
        self.bn2_2 = nn.BatchNorm2d(hidden_size)
        self.relu2_2 = nn.ReLU()

        self.conv1_1 = nn.Conv2d(hidden_size, 2, 1)

    def forward(self, lan, x_c4, x_c3, x_c2, x_c1):
        # 融合 Y4 与 Y3
        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(input=x_c4, size=(x_c3.size(-2), x_c3.size(-1)),
                                 mode='bilinear', align_corners=True)
        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.relu1_4(self.bn1_4(self.conv1_4(x)))
        x = self.relu2_4(self.bn2_4(self.conv2_4(x)))
        x = self.transformer_fusion1(x, lan)

        # 融合 Y2
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(input=x, size=(x_c2.size(-2), x_c2.size(-1)),
                              mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c2], dim=1)
        x = self.relu1_3(self.bn1_3(self.conv1_3(x)))
        x = self.relu2_3(self.bn2_3(self.conv2_3(x)))
        x = self.transformer_fusion2(x, lan)

        # 融合 Y1
        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(input=x, size=(x_c1.size(-2), x_c1.size(-1)),
                              mode='bilinear', align_corners=True)
        x = torch.cat([x, x_c1], dim=1)
        x = self.relu1_2(self.bn1_2(self.conv1_2(x)))
        x = self.relu2_2(self.bn2_2(self.conv2_2(x)))

        return self.conv1_1(x)
