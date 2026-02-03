import torch
import torch.nn.functional as F
import torch.nn as nn

class DiceLoss:
    "Dice loss for segmentation"

    def __init__(self,
                 axis: int = 1,  # Class axis
                 smooth: float = 1e-6,  # Helps with numerical stabilities in the IoU division
                 reduction: str = "sum",  # PyTorch reduction to apply to the output
                 square_in_union: bool = False  # Squares predictions to increase slope of gradients
                 ):
        self.axis = axis
        self.smooth = smooth
        self.reduction = reduction
        self.square_in_union = square_in_union

    def __call__(self, pred, targ):
        "One-hot encodes targ, then runs IoU calculation then takes 1-dice value"

        targ = self._one_hot(targ, pred.shape[self.axis])

        assert pred.shape == targ.shape, 'input and target dimensions differ, DiceLoss expects non one-hot targs'
        pred = self.activation(pred)
        sum_dims = list(range(2, len(pred.shape)))
        inter = torch.sum(pred * targ, dim=sum_dims)
        union = (torch.sum(pred ** 2 + targ, dim=sum_dims) if self.square_in_union
                 else torch.sum(pred + targ, dim=sum_dims))
        dice_score = (2. * inter + self.smooth) / (union + self.smooth)
        loss = 1 - dice_score
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

    @staticmethod
    def _one_hot(
            x,  # Non one-hot encoded targs
            classes: int,  # The number of classes
            axis: int = 1  # The axis to stack for encoding (class dimension)
    ):
        "Creates one binary mask per class"
        return torch.stack([torch.where(x == c, 1, 0) for c in range(classes)], axis=axis)
    # @staticmethod
    # def _one_hot(x, classes: int, axis: int = 1, smoothing: float = 0.1):  # 增加smoothing参数
    #         return torch.stack([torch.where(x == c, 1, 0) for c in range(classes)], axis=axis)

    def activation(self, x):
        "Activation function applied to model output"
        return F.softmax(x, dim=self.axis)

    def decodes(self, x):
        "Converts model output to target format"
        return x.argmax(dim=self.axis)



# class Loss():
#     def __init__(self, weight=0.1):
#         self.dice_loss = DiceLoss()
#         self.ce_loss = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.9, 1.1]).cuda())
#         self.weight = weight

#     def __call__(self, pred, targ):
#         dice_loss = self.dice_loss(pred, targ)
#         ce_loss = self.ce_loss(pred, targ)
#         return (1 - self.weight) * ce_loss + self.weight * dice_loss



class Loss():
    def __init__(self, weight=0.1, ohem_threshold=0.7, ohem_min_kept=50000):
        self.dice_loss = DiceLoss()
        self.ce_loss = OhemCrossEntropyLoss(
            threshold=ohem_threshold,
            min_kept=ohem_min_kept,
            weight=torch.FloatTensor([0.9, 1.1]).cuda()
        )
        self.weight = weight

    def __call__(self, pred, targ):
        dice_loss = self.dice_loss(pred, targ)
        ce_loss = self.ce_loss(pred, targ)
        return (1 - self.weight) * ce_loss + self.weight * dice_loss


class OhemCrossEntropyLoss(nn.Module):
    """
    Online Hard Example Mining Cross Entropy Loss
    只对损失最大的前一部分样本进行梯度回传，关注难例
    """
    
    def __init__(self, 
                 threshold: float = 0.7,  # 困难样本阈值比例
                 min_kept: int = 1,      # 最少保留样本数
                 ignore_index: int = -100,  # 忽略的标签索引
                 weight: torch.Tensor = None):
        super(OhemCrossEntropyLoss, self).__init__()
        self.threshold = threshold
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.weight = weight
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, 
            ignore_index=ignore_index,
            reduction='none'
        )
    
    def forward(self, pred, target):
        """
        Args:
            pred: [N, C, H, W] 模型预测值
            target: [N, H, W] 真实标签
        Returns:
            loss: 加权后的损失值
        """
        pixel_losses = self.criterion(pred, target)  # [N, H, W]
        if self.ignore_index != -100:
            valid_mask = target != self.ignore_index
            pixel_losses = pixel_losses[valid_mask]
        else:
            valid_mask = None
            pixel_losses = pixel_losses.contiguous().view(-1)
        if pixel_losses.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        losses = pixel_losses.contiguous().view(-1)
        n = losses.numel()
        
        min_kept = max(self.min_kept, int(n * self.threshold))
        
        if n > min_kept:
            threshold_loss = losses.kthvalue(n - min_kept + 1).values
            ohem_mask = losses >= threshold_loss
            losses = losses[ohem_mask]
        # 返回困难样本的平均损失
        return losses.mean()