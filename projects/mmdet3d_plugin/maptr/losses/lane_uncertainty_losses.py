"""
Lane Detection Uncertainty Losses for MapTR
车道线检测的不确定性损失函数，包含负样本利用功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Any, Optional


class UncertaintyLoss(nn.Module):
    """
    基础不确定性损失模块，提供通用的不确定性损失计算方法
    """

    def __init__(self, weight=1.0, reduction='mean'):
        super(UncertaintyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, predictions, targets, uncertainty_info=None):
        """
        计算基本的不确定性损失

        Args:
            predictions: 模型预测
            targets: 目标标签
            uncertainty_info: 不确定性信息字典

        Returns:
            计算得到的损失
        """
        raise NotImplementedError("子类必须实现forward方法")


class LaneUncertaintyLoss(UncertaintyLoss):
    """
    车道线不确定性损失，适用于车道线分类任务
    基于MC Dropout的不确定性估计和负样本利用
    """

    def __init__(self, weight=1.0, reduction='mean', use_epistemic=True,
                 use_aleatoric=True, use_contrast=True, class_weights=None,
                 uncertainty_weight=0.1, negative_sample_weight=1.0):
        super(LaneUncertaintyLoss, self).__init__(weight, reduction)
        self.use_epistemic = use_epistemic
        self.use_aleatoric = use_aleatoric
        self.use_contrast = use_contrast
        self.uncertainty_weight = uncertainty_weight
        self.negative_sample_weight = negative_sample_weight

        # 损失函数组件
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions, targets, uncertainty_info=None):
        """
        计算车道线不确定性损失

        Args:
            predictions: 模型预测字典，包含'logits'和不确定性信息
            targets: 目标标签 [batch_size, sequence_length]
            uncertainty_info: 不确定性信息字典，包含MC采样结果

        Returns:
            损失字典
        """
        if uncertainty_info is None and isinstance(predictions, dict) and 'all_logits' not in predictions:
            # 如果没有不确定性信息，退化为标准交叉熵损失
            logits = predictions['logits'] if isinstance(predictions, dict) else predictions
            loss = self.ce_loss(logits, targets)
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
            return {'total_loss': loss * self.weight}

        losses = {}
        device = targets.device

        # 获取预测结果
        if isinstance(predictions, dict):
            logits = predictions['logits']
            uncertainty_info = predictions
        else:
            logits = predictions

        # 提取有效掩码（忽略-100等无效标签）
        valid_mask = (targets >= 0)

        # 基本分类损失
        basic_loss = self.ce_loss(logits, targets)
        if self.reduction == 'mean':
            basic_loss = basic_loss[valid_mask].mean()
        elif self.reduction == 'sum':
            basic_loss = basic_loss[valid_mask].sum()
        losses['basic_loss'] = basic_loss

        # 从不确定性信息中提取数据
        if 'all_logits' in uncertainty_info and (self.use_epistemic or self.use_aleatoric or self.use_contrast):
            all_logits = uncertainty_info['all_logits']
            mc_forward_num = all_logits.shape[0]

            # 计算不确定性度量
            all_probs = F.softmax(all_logits, dim=-1)
            mean_probs = torch.mean(all_probs, dim=0)

            if self.use_epistemic:
                # 认知不确定性正则化
                epistemic_uncertainty = torch.var(all_probs, dim=0).mean(dim=-1)
                epistemic_loss = epistemic_uncertainty[valid_mask].mean()
                losses['epistemic_loss'] = epistemic_loss * self.uncertainty_weight

            if self.use_aleatoric:
                # 偶然不确定性正则化
                entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
                aleatoric_loss = entropy[valid_mask].mean()
                losses['aleatoric_loss'] = aleatoric_loss * self.uncertainty_weight

            if self.use_contrast:
                # MC采样结果之间的一致性对比损失
                contrast_loss = 0.0
                for i in range(mc_forward_num):
                    for j in range(i+1, mc_forward_num):
                        # 对不同MC采样结果之间的一致性进行正则化
                        logits_i = all_logits[i]
                        logits_j = all_logits[j]

                        # KL散度计算
                        kl_div = F.kl_div(
                            F.log_softmax(logits_i, dim=-1),
                            F.softmax(logits_j, dim=-1),
                            reduction='none'
                        ).mean(dim=-1)  # 对类别维度求平均

                        # 只在有效区域计算
                        contrast_loss += kl_div[valid_mask].mean()

                # 归一化
                if mc_forward_num > 1:
                    contrast_loss /= (mc_forward_num * (mc_forward_num - 1) / 2)
                    losses['contrast_loss'] = contrast_loss * 0.05

        # 负样本利用损失
        if 'positive_mask' in uncertainty_info and 'negative_mask' in uncertainty_info:
            positive_mask = uncertainty_info['positive_mask']
            negative_mask = uncertainty_info['negative_mask']

            # 如果 mask 为空，则跳过负样本/正样本分支
            if positive_mask is None:
                positive_mask = torch.zeros_like(targets, dtype=torch.bool, device=targets.device)
            if negative_mask is None:
                negative_mask = torch.zeros_like(targets, dtype=torch.bool, device=targets.device)

            # 负样本损失 - 对不确定样本施加更大的惩罚
            if negative_mask.numel() > 0 and negative_mask.any():
                negative_logits = logits[negative_mask]
                negative_targets = targets[negative_mask]

                # 为负样本分配更高的权重
                negative_loss = self.ce_loss(negative_logits, negative_targets)
                if self.reduction == 'mean':
                    negative_loss = negative_loss.mean()
                elif self.reduction == 'sum':
                    negative_loss = negative_loss.sum()

                losses['negative_sample_loss'] = negative_loss * self.negative_sample_weight

            # 正样本可以保持较低的权重或正常权重
            if positive_mask.numel() > 0 and positive_mask.any():
                positive_logits = logits[positive_mask]
                positive_targets = targets[positive_mask]

                positive_loss = self.ce_loss(positive_logits, positive_targets)
                if self.reduction == 'mean':
                    positive_loss = positive_loss.mean()
                elif self.reduction == 'sum':
                    positive_loss = positive_loss.sum()

                losses['positive_sample_loss'] = positive_loss * 0.5  # 正样本权重较低

        # 计算总损失
        total_loss = basic_loss
        for name, loss in losses.items():
            if name != 'basic_loss':
                total_loss += loss

        losses['total_loss'] = total_loss * self.weight
        return losses


def create_lane_uncertainty_loss(config=None):
    """
    创建车道线不确定性损失函数

    Args:
        config: 配置字典

    Returns:
        LaneUncertaintyLoss实例
    """
    if config is None:
        config = {}

    return LaneUncertaintyLoss(
        weight=config.get('weight', 1.0),
        reduction=config.get('reduction', 'mean'),
        use_epistemic=config.get('use_epistemic', True),
        use_aleatoric=config.get('use_aleatoric', True),
        use_contrast=config.get('use_contrast', True),
        uncertainty_weight=config.get('uncertainty_weight', 0.1),
        negative_sample_weight=config.get('negative_sample_weight', 1.0)
    )
