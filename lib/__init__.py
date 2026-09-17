"""PVLMamba 模型库：骨干网络、跨模态融合头与解码器骨架。

子模块按需导入，避免在缺少 mamba_ssm / mmengine 等可选依赖时整包导入失败：

    from lib import segmentation
"""

__all__ = ["segmentation", "decoder", "mask_predictor", "MVbackbone", "backbone"]
