"""
Segmentation Uncertainty Heads for MapTR
语义分割的不确定性头部模块，用于边界分类和区域检测
Adapted from GAS+UAUL model for segmentation uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Any
from mmcv.cnn import xavier_init, constant_init
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner.base_module import BaseModule


class MCDropoutConv2D(nn.Module):
    """MC Dropout layer specifically for 2D convolutions"""

    def __init__(self, dropout_rate: float = 0.1):
        super(MCDropoutConv2D, self).__init__()
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        return self.dropout(x)


class SegmentationUncertaintyWrapper(nn.Module):
    """用于边界分类(edge_cls)的不确定性估计包装器，使用MC Dropout方法"""

    def __init__(self, original_head, mc_forward_num=5, dropout_rate=0.3, temperature=1.0,
                 use_epistemic=True, use_aleatoric=True, use_margin=True):
        super(SegmentationUncertaintyWrapper, self).__init__()
        self.original_head = original_head
        self.mc_forward_num = mc_forward_num
        self.dropout = nn.Dropout(dropout_rate)
        self.temperature = temperature
        self.use_epistemic = use_epistemic  # 认知不确定性（模型不确定性）
        self.use_aleatoric = use_aleatoric  # 偶然不确定性（数据不确定性）
        self.use_margin = use_margin        # 是否使用边界估计
        self.training_mode = True

    def forward(self, x, labels=None, return_uncertainty=False):
        """
        执行前向传播，可选返回不确定性估计

        Args:
            x: 输入特征
            labels: 可选的标签，用于训练时计算损失
            return_uncertainty: 是否返回不确定性估计

        Returns:
            如果return_uncertainty=True，返回包含均值预测和不确定性估计的字典
            否则，只返回均值预测
        """
        if not self.training_mode or not return_uncertainty:
            # 在推理模式或不需要不确定性时，直接使用原始头部
            return self.original_head(x)

        # 使用MC Dropout进行多次采样
        all_logits = []
        for _ in range(self.mc_forward_num):
            # 应用dropout获得不同的采样
            features_dropout = self.dropout(x)
            # 使用原始头部处理特征
            logits = self.original_head(features_dropout)
            all_logits.append(logits.unsqueeze(0))

        # 堆叠所有采样结果: [mc_forward_num, batch_size, num_classes, height, width]
        all_logits = torch.cat(all_logits, dim=0)

        # 计算均值预测
        mean_logits = torch.mean(all_logits, dim=0)

        if not return_uncertainty:
            return mean_logits

        # 计算不确定性估计
        uncertainty_dict = {
            'mean_logits': mean_logits,
        }

        # 计算认知不确定性(epistemic)
        if self.use_epistemic:
            # 使用预测方差作为认知不确定性的度量
            probs = F.softmax(all_logits / self.temperature, dim=2)
            epistemic_uncertainty = torch.var(probs, dim=0).mean(dim=1)
            uncertainty_dict['epistemic_uncertainty'] = epistemic_uncertainty

        # 计算偶然不确定性(aleatoric)
        if self.use_aleatoric:
            # 使用预测熵作为偶然不确定性的度量
            probs = F.softmax(mean_logits / self.temperature, dim=1)
            aleatoric_uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            uncertainty_dict['aleatoric_uncertainty'] = aleatoric_uncertainty

        # 计算分类边界不确定性
        if self.use_margin and labels is not None:
            # 计算top-1和top-2之间的差距
            sorted_logits, _ = torch.sort(mean_logits, dim=1, descending=True)
            margin = sorted_logits[:, 0] - sorted_logits[:, 1]
            # 归一化边界值
            margin = torch.sigmoid(margin)
            uncertainty_dict['margin'] = margin

            # 计算与真实标签的一致性
            if labels is not None:
                # 确保标签是张量
                if isinstance(labels, list):
                    labels_tensor = torch.tensor(labels, device=mean_logits.device, dtype=torch.long)
                else:
                    labels_tensor = labels

                # 处理标签形状不匹配的问题
                try:
                    # 确保维度匹配
                    if labels_tensor.dim() < mean_logits.dim():
                        # 可能需要扩展维度
                        expand_dims = mean_logits.dim() - labels_tensor.dim()
                        for _ in range(expand_dims):
                            labels_tensor = labels_tensor.unsqueeze(1)

                    # 确保形状匹配
                    if mean_logits.shape[1:] != labels_tensor.shape[1:]:
                        # 尝试调整形状
                        pred_classes = torch.argmax(mean_logits, dim=1)
                        if labels_tensor.shape != pred_classes.shape:
                            labels_tensor = F.interpolate(
                                labels_tensor.float().unsqueeze(1),
                                size=pred_classes.shape[1:],
                                mode='nearest'
                            ).squeeze(1).long()

                    mask = (labels_tensor >= 0)  # 忽略负标签
                    correct_pred = (torch.argmax(mean_logits, dim=1) == labels_tensor) & mask
                    uncertainty_dict['accuracy'] = correct_pred.float()
                except Exception as e:
                    # 如果处理失败，记录错误但不中断训练
                    print(f"警告：处理标签形状时出错: {e}")

        return uncertainty_dict

    def train(self, mode=True):
        """
        设置模块为训练模式
        """
        super(SegmentationUncertaintyWrapper, self).train(mode)
        self.training_mode = mode
        self.original_head.train(mode)
        return self


class AreaDetectionUncertaintyWrapper(nn.Module):
    """用于区域检测分类(zebra_cls, arrow_cls)的不确定性估计包装器"""

    def __init__(self, original_head, mc_forward_num=5, dropout_rate=0.3, temperature=1.0,
                 road_element_type='zebra', junction_weight=1.2):
        super(AreaDetectionUncertaintyWrapper, self).__init__()
        self.original_head = original_head
        self.mc_forward_num = mc_forward_num
        self.dropout = nn.Dropout(dropout_rate)
        self.temperature = temperature
        self.road_element_type = road_element_type  # 'zebra' 或 'arrow'
        self.junction_weight = junction_weight  # 路口场景权重
        self.training_mode = True

        # 特定于路口元素类型的配置
        if road_element_type == 'zebra':
            self.confidence_threshold = 0.7
            self.uncertainty_ratio = 0.3
        elif road_element_type == 'arrow':
            self.confidence_threshold = 0.6  # 箭头分类需要更低的阈值，因为类别更多
            self.uncertainty_ratio = 0.4     # 更高的不确定性比例
        else:
            self.confidence_threshold = 0.7
            self.uncertainty_ratio = 0.3

    def forward(self, x, labels=None, return_uncertainty=False):
        """
        执行前向传播，可选返回不确定性估计

        Args:
            x: 输入特征
            labels: 可选的标签，用于训练时计算损失
            return_uncertainty: 是否返回不确定性估计

        Returns:
            如果return_uncertainty=True，返回包含均值预测和不确定性估计的字典
            否则，只返回均值预测
        """
        if not self.training_mode or not return_uncertainty:
            # 在推理模式或不需要不确定性时，直接使用原始头部
            return self.original_head(x)

        # 使用MC Dropout进行多次采样
        all_logits = []
        for i in range(self.mc_forward_num):
            # 路口场景采样策略：变化的dropout率
            dropout_rate = self.dropout.p * (1.0 + 0.1 * (i % 3))
            temp_dropout = nn.Dropout(dropout_rate)

            # 应用dropout获得不同的采样
            features_dropout = temp_dropout(x)
            # 使用原始头部处理特征
            logits = self.original_head(features_dropout)
            all_logits.append(logits.unsqueeze(0))

        # 堆叠所有采样结果: [mc_forward_num, batch_size, num_classes, height, width]
        all_logits = torch.cat(all_logits, dim=0)

        # 计算均值预测
        mean_logits = torch.mean(all_logits, dim=0)

        if not return_uncertainty:
            return mean_logits

        # 计算不确定性估计
        uncertainty_dict = {
            'mean_logits': mean_logits,
        }

        # 计算预测概率
        probs = F.softmax(all_logits / self.temperature, dim=2)
        mean_probs = torch.mean(probs, dim=0)
        uncertainty_dict['mean_probs'] = mean_probs

        # 计算认知不确定性(epistemic) - 模型不确定性
        epistemic_uncertainty = torch.var(probs, dim=0).mean(dim=1)
        uncertainty_dict['epistemic_uncertainty'] = epistemic_uncertainty

        # 计算偶然不确定性(aleatoric) - 数据不确定性
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
        uncertainty_dict['aleatoric_uncertainty'] = entropy

        # 组合不确定性
        combined_uncertainty = (
            (1 - self.uncertainty_ratio) * epistemic_uncertainty +
            self.uncertainty_ratio * entropy
        )
        uncertainty_dict['combined_uncertainty'] = combined_uncertainty

        # 使用不确定性调整预测置信度
        confidence = torch.max(mean_probs, dim=1)[0]
        adjusted_confidence = confidence * (1 - combined_uncertainty)
        uncertainty_dict['adjusted_confidence'] = adjusted_confidence

        # 标记不确定预测
        uncertain_predictions = adjusted_confidence < self.confidence_threshold
        uncertainty_dict['uncertain_predictions'] = uncertain_predictions

        # 如果有标签，计算与真实标签的一致性
        if labels is not None:
            # 确保标签是张量
            if isinstance(labels, list):
                labels_tensor = torch.tensor(labels, device=mean_logits.device, dtype=torch.long)
            else:
                labels_tensor = labels

            # 确保标签形状匹配预测（可能需要调整维度）
            if labels_tensor.dim() < mean_logits.dim():
                # 对于分割任务，标签通常缺少通道维度
                # 确定需要扩展的维度
                expand_dims = mean_logits.dim() - labels_tensor.dim()
                for _ in range(expand_dims):
                    labels_tensor = labels_tensor.unsqueeze(1)

            # 确保标签形状与预测匹配
            if labels_tensor.shape != mean_logits.shape:
                # 重新整形或广播标签以匹配预测
                try:
                    # 对于分类任务，可能需要对标签进行广播
                    if mean_logits.dim() > labels_tensor.dim():
                        # 调整标签形状
                        for i in range(mean_logits.dim() - labels_tensor.dim()):
                            labels_tensor = labels_tensor.unsqueeze(-1)
                    # 广播到相同形状
                    labels_tensor = labels_tensor.expand_as(mean_logits.argmax(dim=1))
                except:
                    # 如果调整失败，返回没有标签相关的不确定性信息
                    return uncertainty_dict

            mask = (labels_tensor >= 0)  # 忽略负标签
            pred_classes = torch.argmax(mean_logits, dim=1)
            correct_pred = (pred_classes == labels_tensor) & mask
            uncertainty_dict['accuracy'] = correct_pred.float()

            # 计算标签与预测分布的KL散度
            try:
                target_one_hot = F.one_hot(labels_tensor.clamp(min=0), num_classes=mean_probs.size(1)).float()
                target_one_hot = target_one_hot * mask.unsqueeze(-1)
                kl_div = F.kl_div(
                    torch.log(mean_probs + 1e-8),
                    target_one_hot,
                    reduction='none'
                ).sum(dim=1)
                uncertainty_dict['kl_divergence'] = kl_div * mask
            except:
                # 如果KL散度计算失败，跳过这部分
                pass

        return uncertainty_dict

    def train(self, mode=True):
        """
        设置模块为训练模式
        """
        super(AreaDetectionUncertaintyWrapper, self).train(mode)
        self.training_mode = mode
        self.original_head.train(mode)
        return self


def create_uncertainty_head(original_head, head_type, config=None):
    """
    创建适用于不同分类头部的不确定性包装器

    Args:
        original_head: 原始分类头
        head_type: 头部类型 ('edge_cls', 'zebra_cls', 'arrow_cls')
        config: 可选配置字典

    Returns:
        包装后的不确定性头部
    """
    if config is None:
        config = {}

    mc_forward_num = config.get('mc_forward_num', 5)
    dropout_rate = config.get('dropout_rate', 0.3)
    temperature = config.get('temperature', 1.0)

    if head_type == 'edge_cls':
        return SegmentationUncertaintyWrapper(
            original_head,
            mc_forward_num=mc_forward_num,
            dropout_rate=dropout_rate,
            temperature=temperature,
            use_epistemic=config.get('use_epistemic', True),
            use_aleatoric=config.get('use_aleatoric', True),
            use_margin=config.get('use_margin', True)
        )
    elif head_type in ['zebra_cls', 'arrow_cls']:
        junction_weights = config.get('junction_specific', {})
        junction_weight = junction_weights.get(
            'zebra_weight' if head_type == 'zebra_cls' else 'arrow_weight',
            1.2 if head_type == 'zebra_cls' else 1.5
        )

        return AreaDetectionUncertaintyWrapper(
            original_head,
            mc_forward_num=mc_forward_num,
            dropout_rate=dropout_rate,
            temperature=temperature,
            road_element_type=head_type.split('_')[0],
            junction_weight=junction_weight
        )
    else:
        raise ValueError(f"不支持的头部类型: {head_type}")
