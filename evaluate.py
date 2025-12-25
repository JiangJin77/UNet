import torch.nn as nn

def dice_loss(pred, target, smooth=1e-6):
    """
    计算Dice Loss
    :param pred: 模型预测结果，形状为 (N, C, H, W)
    :param target: 真实标签，形状为 (N, C, H, W)
    :param smooth: 平滑项，防止分母为零
    :return: Dice Loss 值
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)   
    intersection = (pred_flat * target_flat).sum()
    
    dice_coeff = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    return 1 - dice_coeff


#  combined loss = 0.6 * dice + 0.4 * bce
def combined_loss(pred, target):
    dice = dice_loss(pred, target)
    bce = nn.BCELoss()(pred, target)
    return 0.6 * dice + 0.4 * bce


# 加权Dice Loss
def weighted_dice_loss(pred, target, weight=None, smooth=1e-6):
    """
    计算加权Dice Loss
    :param pred: 模型预测结果，形状为 (N, C, H, W)
    :param target: 真实标签，形状为 (N, C, H, W)
    :param weight: 权重张量，形状与 target 一致，或为 None
    :param smooth: 平滑项，防止分母为零
    :return: 加权Dice Loss 值
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    if weight is not None:
        weight_flat = weight.view(-1)
        intersection = (weight_flat * pred_flat * target_flat).sum()
        union = (weight_flat * pred_flat).sum() + (weight_flat * target_flat).sum()
    else:
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
    
    return 1 - ((2. * intersection + smooth) / (union + smooth))
