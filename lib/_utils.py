"""LAVT / LAVT-One 两种整体结构的封装。

- ``LAVT``:     文本由外部 BERT 编码后以特征形式输入
- ``LAVTOne``:  文本编码器内置于模型内部（lavt_one 分支）

两者共享同一套 backbone + classifier(掩码头) 接口，区别仅在文本特征来自哪里。
解码器的具体实现见 ``lib/decoder.py``（当前为接口骨架，待补全）。
"""

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat

from bert.modeling_bert import BertModel


class _LAVTSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_LAVTSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask):
        input_shape = x.shape[-2:]
        # 注意：Swin 版骨干的 forward 签名为 (x, l, l_mask, extended_attention_mask)，
        # 走 lavt 分支时请按其签名补齐参数。
        features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features

        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVT(_LAVTSimpleDecode):
    pass


###############################################
# LAVT One: put BERT inside the overall model #
###############################################
class _LAVTOneSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_LAVTOneSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None

    def forward(self, x, text, l_mask):

        input_shape = x.shape[-2:]

        l_shape = text.size()
        token_type_ids = torch.zeros(l_shape, dtype=torch.long, device=text.device)
        extended_attention_mask = self.text_encoder.get_extended_attention_mask(l_mask, l_shape, text.device)

        embedding_output = self.text_encoder.embeddings(
            input_ids=text, token_type_ids=token_type_ids
        )
        l_feats = embedding_output.permute(0, 2, 1)  # (B, 768, N_l)

        l_mask = l_mask.unsqueeze(dim=-1)  # (B, N_l, 1)

        l_1, features = self.backbone(x, l_feats, l_mask, extended_attention_mask)
        x_c1, x_c2, x_c3, x_c4 = features

        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x


class LAVTOne(_LAVTOneSimpleDecode):
    pass
