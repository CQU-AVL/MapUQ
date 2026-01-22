"""
Segmentation Uncertainty Losses for MapTR
语义分割的不确定性损失函数，用于边界分类和区域检测
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


class SegmentationUncertaintyLoss(UncertaintyLoss):
    """
    分割不确定性损失，适用于边界分类(edge_cls)
    """

    def __init__(self, weight=1.0, reduction='mean', use_epistemic=True,
                 use_aleatoric=True, use_contrast=True, class_weights=None,
                 epistemic_weight: float = 0.1,
                 aleatoric_weight: float = 0.05,
                 use_dynamic_weight: bool = True,
                 contrast_quantile: float = 0.7,
                 contrast_margin: float = -0.6,
                 contrast_temperature: float = 1.0,
                 contrast_weight: float = 0.05):
        super(SegmentationUncertaintyLoss, self).__init__(weight, reduction)
        self.use_epistemic = use_epistemic
        self.use_aleatoric = use_aleatoric
        self.use_contrast = use_contrast
        self.class_weights = class_weights

        # UFCE-style dynamic weighting hyperparameters
        self.epistemic_weight = epistemic_weight
        self.aleatoric_weight = aleatoric_weight
        self.use_dynamic_weight = use_dynamic_weight

        # High/low confidence contrast settings (UBE V-inspired)
        self.contrast_quantile = contrast_quantile
        self.contrast_margin = contrast_margin
        self.contrast_temperature = contrast_temperature
        self.contrast_weight = contrast_weight

        # 损失函数组件
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions, targets, uncertainty_info=None):
        """
        计算分割不确定性损失

        Args:
            predictions: 模型预测 [batch_size, num_classes, height, width]
            targets: 目标标签 [batch_size, height, width]
            uncertainty_info: 不确定性信息字典，包含MC采样结果

        Returns:
            损失字典
        """
        if uncertainty_info is None:
            # 如果没有提供不确定性信息，退化为标准交叉熵损失
            loss = self.ce_loss(predictions, targets)
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
            return {'total_loss': loss * self.weight}

        losses = {}
        device = predictions.device

        # 提取有效掩码（忽略-100等无效标签）
        valid_mask = (targets >= 0)

        # 基本分类损失（像素级）
        ce_map = self.ce_loss(predictions, targets)  # [B, H, W]

        # UFCE-style dynamic sample weighting using epistemic/aleatoric uncertainty
        weights = torch.ones_like(ce_map)
        if self.use_dynamic_weight:
            # Epistemic weighting: w = 1 + k * epistemic
            if 'epistemic_uncertainty' in uncertainty_info and self.use_epistemic:
                epistemic = uncertainty_info['epistemic_uncertainty'].detach()
                # 对不确定性做简单归一化，提升数值稳定性
                ep_min = epistemic.min()
                ep_max = epistemic.max()
                if (ep_max - ep_min) > 1e-6:
                    epistemic_norm = (epistemic - ep_min) / (ep_max - ep_min + 1e-6)
                else:
                    epistemic_norm = epistemic * 0.0
                weights = weights * (1.0 + self.epistemic_weight * epistemic_norm)

            # Aleatoric weighting：同理，可视需要下调/上调噪声较大的像素权重
            if 'aleatoric_uncertainty' in uncertainty_info and self.use_aleatoric:
                aleatoric = uncertainty_info['aleatoric_uncertainty'].detach()
                al_min = aleatoric.min()
                al_max = aleatoric.max()
                if (al_max - al_min) > 1e-6:
                    aleatoric_norm = (aleatoric - al_min) / (al_max - al_min + 1e-6)
                else:
                    aleatoric_norm = aleatoric * 0.0
                weights = weights * (1.0 + self.aleatoric_weight * aleatoric_norm)

        weighted_ce = ce_map * weights
        if self.reduction == 'mean':
            basic_loss = weighted_ce[valid_mask].mean()
        elif self.reduction == 'sum':
            basic_loss = weighted_ce[valid_mask].sum()
        else:
            basic_loss = weighted_ce
        losses['basic_loss'] = basic_loss

        # 从不确定性信息中提取数据
        if 'epistemic_uncertainty' in uncertainty_info and self.use_epistemic:
            # 认知不确定性正则化
            epistemic = uncertainty_info['epistemic_uncertainty']

            # 使用有效掩码
            epistemic_valid = epistemic[valid_mask]

            # 正则化损失 - 鼓励不确定性降低
            epistemic_loss = epistemic_valid.mean()
            losses['epistemic_loss'] = epistemic_loss * 0.05

        if 'aleatoric_uncertainty' in uncertainty_info and self.use_aleatoric:
            # 偶然不确定性正则化
            aleatoric = uncertainty_info['aleatoric_uncertainty']

            # 使用有效掩码
            aleatoric_valid = aleatoric[valid_mask]

            # 正则化损失 - 鼓励不确定性降低
            aleatoric_loss = aleatoric_valid.mean()
            losses['aleatoric_loss'] = aleatoric_loss * 0.02

        if 'all_logits' in uncertainty_info and self.use_contrast:
            # MC采样结果
            all_logits = uncertainty_info['all_logits']
            mc_forward_num = all_logits.shape[0]

            # 计算MC一致性对比损失（原有项）
            contrast_loss = 0.0
            for i in range(mc_forward_num):
                for j in range(i+1, mc_forward_num):
                    # 对不同MC采样结果之间的一致性进行正则化
                    logits_i = all_logits[i]
                    logits_j = all_logits[j]

                    # KL散度计算
                    kl_div = F.kl_div(
                        F.log_softmax(logits_i, dim=1),
                        F.softmax(logits_j, dim=1),
                        reduction='none'
                    ).sum(dim=1)

                    # 只在有效区域计算
                    contrast_loss += kl_div[valid_mask].mean()

            # 归一化
            if mc_forward_num > 1:
                contrast_loss /= (mc_forward_num * (mc_forward_num - 1) / 2)
                losses['contrast_loss_mc'] = contrast_loss * 0.03

            # 进一步：基于高/低置信度分区的对比（UBE V 思路）
            # 注意：此处不强依赖 MC，不确定性可以来自 predictions 本身
            with torch.no_grad():
                probs = F.softmax(predictions, dim=1)
                confidence, _ = probs.max(dim=1)  # [B, H, W]
                conf_valid = confidence[valid_mask]
                if conf_valid.numel() > 0:
                    thr = torch.quantile(conf_valid, self.contrast_quantile)
                    high_conf_mask = (confidence > thr) & valid_mask
                    low_conf_mask = (confidence <= thr) & valid_mask
                else:
                    high_conf_mask = torch.zeros_like(confidence, dtype=torch.bool)
                    low_conf_mask = torch.zeros_like(confidence, dtype=torch.bool)

            if high_conf_mask.any() and low_conf_mask.any():
                # 取出高/低置信度像素的预测向量
                preds_flat = predictions.permute(0, 2, 3, 1)  # [B, H, W, C]
                high_logits = preds_flat[high_conf_mask]  # [N_high, C]
                low_logits = preds_flat[low_conf_mask]    # [N_low, C]

                if high_logits.numel() > 0 and low_logits.numel() > 0:
                    high_probs = F.softmax(high_logits / self.contrast_temperature, dim=1)
                    low_probs = F.softmax(low_logits / self.contrast_temperature, dim=1)

                    # 用点积相似度构造对比损失：鼓励高低置信度分布有明显区分
                    sim = (high_probs * low_probs).sum(dim=1).mean()
                    contrast_loss_hl = -torch.log(
                        torch.exp(self.contrast_margin) + sim + 1e-8
                    )
                    losses['contrast_loss_hl'] = contrast_loss_hl * self.contrast_weight

        # 计算总损失
        total_loss = basic_loss
        for name, loss in losses.items():
            if name != 'basic_loss':
                total_loss += loss

        losses['total_loss'] = total_loss * self.weight
        return losses


class AreaDetectionUncertaintyLoss(UncertaintyLoss):
    """
    区域检测不确定性损失，适用于斑马线分类(zebra_cls)和箭头分类(arrow_cls)
    """

    def __init__(self, weight=1.0, reduction='mean', margin=-0.6,
                 temperature=1.0, calibration_weight=0.05, class_weights=None,
                 road_element_type='zebra'):
        super(AreaDetectionUncertaintyLoss, self).__init__(weight, reduction)
        self.margin = margin
        self.temperature = temperature
        self.calibration_weight = calibration_weight
        self.class_weights = class_weights
        self.road_element_type = road_element_type

        # 根据路口元素类型设置权重
        if road_element_type == 'zebra':
            self.uncertainty_weight = 0.1
            self.calibration_weight = 0.05
        elif road_element_type == 'arrow':
            self.uncertainty_weight = 0.15
            self.calibration_weight = 0.07
        else:
            self.uncertainty_weight = 0.1
            self.calibration_weight = 0.05

        # 损失函数组件
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights, reduction='none')

    def forward(self, predictions, targets, uncertainty_info=None):
        """
        计算区域检测不确定性损失

        Args:
            predictions: 模型预测 [batch_size, num_classes, height, width]
            targets: 目标标签 [batch_size, height, width]
            uncertainty_info: 不确定性信息字典，包含MC采样结果和不确定性估计

        Returns:
            损失字典
        """
        if uncertainty_info is None:
            # 如果没有提供不确定性信息，退化为标准交叉熵损失
            loss = self.ce_loss(predictions, targets)
            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()
            return {'total_loss': loss * self.weight}

        losses = {}
        device = predictions.device

        # 提取有效掩码（忽略-100等无效标签）
        valid_mask = (targets >= 0)

        # 基本分类损失
        basic_loss = self.ce_loss(predictions, targets)
        if self.reduction == 'mean':
            basic_loss = basic_loss[valid_mask].mean()
        elif self.reduction == 'sum':
            basic_loss = basic_loss[valid_mask].sum()
        losses['basic_loss'] = basic_loss

        # 不确定性信息处理
        if 'mean_probs' in uncertainty_info:
            mean_probs = uncertainty_info['mean_probs']

            # 校准损失 - 确保预测概率与实际准确性一致
            if 'accuracy' in uncertainty_info:
                accuracy = uncertainty_info['accuracy']
                confidence = torch.max(mean_probs, dim=1)[0]

                # 只在有效区域计算
                valid_confidence = confidence[valid_mask]
                valid_accuracy = accuracy[valid_mask]

                if valid_confidence.numel() > 0 and valid_accuracy.numel() > 0:
                    # 计算置信度与准确性之间的均方误差
                    calibration_loss = F.mse_loss(valid_confidence, valid_accuracy)
                    losses['calibration_loss'] = calibration_loss * self.calibration_weight

        # 对比学习损失
        if 'all_logits' in uncertainty_info:
            all_logits = uncertainty_info['all_logits']
            mc_forward_num = all_logits.shape[0]

            # 计算高/低置信度样本之间的对比损失
            if 'adjusted_confidence' in uncertainty_info:
                adjusted_confidence = uncertainty_info['adjusted_confidence']

                # 确定高/低置信度区域
                confidence_threshold = torch.quantile(adjusted_confidence[valid_mask].flatten(), 0.7)
                high_conf_mask = (adjusted_confidence > confidence_threshold) & valid_mask
                low_conf_mask = (adjusted_confidence <= confidence_threshold) & valid_mask

                if high_conf_mask.any() and low_conf_mask.any():
                    # 提取高/低置信度样本的特征
                    high_conf_logits = predictions[:, :, high_conf_mask]
                    low_conf_logits = predictions[:, :, low_conf_mask]

                    # 计算特征之间的相似度
                    high_conf_probs = F.softmax(high_conf_logits / self.temperature, dim=1)
                    low_conf_probs = F.softmax(low_conf_logits / self.temperature, dim=1)

                    # 对比损失 - 使高置信度样本与低置信度样本区分开
                    contrast_loss = -torch.log(
                        torch.exp(self.margin) +
                        torch.sum(high_conf_probs * low_conf_probs, dim=1).mean()
                    )
                    losses['contrast_loss'] = contrast_loss * 0.1

        # 组合不确定性损失
        if 'combined_uncertainty' in uncertainty_info:
            combined_uncertainty = uncertainty_info['combined_uncertainty']

            # 只在有效区域计算
            valid_uncertainty = combined_uncertainty[valid_mask]

            if valid_uncertainty.numel() > 0:
                # 不确定性正则化 - 鼓励不确定性降低
                uncertainty_loss = valid_uncertainty.mean()
                losses['uncertainty_loss'] = uncertainty_loss * self.uncertainty_weight

        # 计算总损失
        total_loss = basic_loss
        for name, loss in losses.items():
            if name != 'basic_loss':
                total_loss += loss

        losses['total_loss'] = total_loss * self.weight
        return losses


def build_segmentation_uncertainty_loss(config):
    """
    构建语义分割不确定性损失函数

    Args:
        config: 损失函数配置

    Returns:
        构建的损失函数
    """
    loss_type = config.get('type', 'SegmentationUncertaintyLoss')

    if loss_type == 'SegmentationUncertaintyLoss':
        return SegmentationUncertaintyLoss(
            weight=config.get('weight', 1.0),
            reduction=config.get('reduction', 'mean'),
            use_epistemic=config.get('use_epistemic', True),
            use_aleatoric=config.get('use_aleatoric', True),
            use_contrast=config.get('use_contrast', True),
            class_weights=config.get('class_weights', None)
        )
    elif loss_type == 'AreaDetectionUncertaintyLoss':
        return AreaDetectionUncertaintyLoss(
            weight=config.get('weight', 1.0),
            reduction=config.get('reduction', 'mean'),
            margin=config.get('margin', -0.6),
            temperature=config.get('temperature', 1.0),
            calibration_weight=config.get('calibration_weight', 0.05),
            class_weights=config.get('class_weights', None),
            road_element_type=config.get('road_element_type', 'zebra')
        )
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")
