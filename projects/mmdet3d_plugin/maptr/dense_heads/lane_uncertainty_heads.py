"""
Lane Detection Uncertainty Heads for MapTR
车道线检测的不确定性头部模块，包含负样本利用功能
Adapted from GAS+UAUL model for lane detection uncertainty estimation
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


class MCDropoutLayer(nn.Module):
    """Monte Carlo Dropout layer for uncertainty estimation"""

    def __init__(self, dropout_rate: float = 0.1):
        super(MCDropoutLayer, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(x)


class UncertaintyClassificationHead(nn.Module):
    """
    Uncertainty-aware classification head using MC Dropout
    For lane detection classification outputs with negative sample utilization
    """

    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 mc_forward_num: int = 5,
                 dropout_rate: float = 0.1,
                 uncertainty_threshold: float = 0.5):
        super(UncertaintyClassificationHead, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.mc_forward_num = mc_forward_num
        self.uncertainty_threshold = uncertainty_threshold

        # MC Dropout layer
        self.mc_dropout = MCDropoutLayer(dropout_rate)

        # Classification head
        self.classifier = nn.Linear(input_dim, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

    def single_forward(self, features):
        """Single forward pass with MC dropout"""
        # Apply MC dropout
        dropped_features = self.mc_dropout(features)
        # Classification
        logits = self.classifier(dropped_features)
        return logits

    def mc_forward(self, features):
        """Multiple forward passes for uncertainty estimation"""
        all_logits = []

        # Set to training mode to enable dropout during inference
        training_mode = self.training
        self.train()

        for _ in range(self.mc_forward_num):
            logits = self.single_forward(features)
            all_logits.append(logits.unsqueeze(0))

        # Restore original training mode
        self.train(training_mode)

        # Stack all predictions: [mc_forward_num, batch_size, sequence_length, num_classes]
        all_logits = torch.cat(all_logits, dim=0)
        return all_logits

    def compute_uncertainty(self, all_logits):
        """
        Compute uncertainty measures from MC predictions
        Args:
            all_logits: [mc_forward_num, batch_size, sequence_length, num_classes]
        Returns:
            mean_logits: Mean prediction
            uncertainty: Predictive uncertainty (variance)
            entropy: Predictive entropy
        """
        # Convert to probabilities
        all_probs = F.softmax(all_logits, dim=-1)

        # Mean prediction
        mean_probs = torch.mean(all_probs, dim=0)
        mean_logits = torch.log(mean_probs + 1e-8)  # Convert back to logits

        # Predictive uncertainty (variance)
        uncertainty = torch.var(all_probs, dim=0)
        uncertainty = torch.mean(uncertainty, dim=-1)  # Average over classes

        # Predictive entropy
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)

        return mean_logits, uncertainty, entropy

    def get_positive_negative_samples(self, all_logits, labels, uncertainty_scores):
        """
        Extract positive and negative samples based on uncertainty
        Args:
            all_logits: [mc_forward_num, batch_size, sequence_length, num_classes]
            labels: Ground truth labels [batch_size, sequence_length]
            uncertainty_scores: Uncertainty scores [batch_size, sequence_length]
        Returns:
            positive_samples: High-confidence correct predictions
            negative_samples: High-uncertainty predictions
        """
        batch_size, seq_len = labels.shape
        device = labels.device

        # Get mean predictions
        mean_logits, _, _ = self.compute_uncertainty(all_logits)
        mean_probs = F.softmax(mean_logits, dim=-1)
        predicted_classes = torch.argmax(mean_probs, dim=-1)

        # Create masks
        valid_mask = (labels != -100)  # Ignore padding tokens
        correct_mask = (predicted_classes == labels) & valid_mask

        # High confidence correct predictions as positive samples
        confidence_scores = torch.max(mean_probs, dim=-1)[0]
        high_confidence_mask = (confidence_scores > self.uncertainty_threshold)
        positive_mask = correct_mask & high_confidence_mask

        # High uncertainty predictions as negative samples
        uncertainty_threshold = torch.quantile(uncertainty_scores[valid_mask], 0.7)
        high_uncertainty_mask = (uncertainty_scores > uncertainty_threshold) & valid_mask

        # Get top uncertain samples for negative sampling
        negative_mask = high_uncertainty_mask

        return {
            'positive_mask': positive_mask,
            'negative_mask': negative_mask,
            'mean_logits': mean_logits,
            'uncertainty_scores': uncertainty_scores,
            'all_logits': all_logits
        }

    def forward(self, features, labels=None, return_uncertainty=False):
        """
        Forward pass with optional uncertainty estimation
        Args:
            features: Input features [batch_size, sequence_length, input_dim]
            labels: Ground truth labels (optional)
            return_uncertainty: Whether to return uncertainty measures
        """
        if return_uncertainty or (labels is not None):
            # MC forward for uncertainty estimation
            all_logits = self.mc_forward(features)
            mean_logits, uncertainty, entropy = self.compute_uncertainty(all_logits)

            result = {
                'logits': mean_logits,
                'uncertainty': uncertainty,
                'entropy': entropy,
                'all_logits': all_logits
            }

            if labels is not None:
                # Extract positive and negative samples
                sample_info = self.get_positive_negative_samples(
                    all_logits, labels, uncertainty
                )
                result.update(sample_info)

            return result
        else:
            # Standard forward pass
            logits = self.single_forward(features)
            return {'logits': logits}


class UncertaintySampleExtractor(nn.Module):
    """
    Extract positive and negative samples based on uncertainty estimation
    Used for negative sample mining in lane detection
    """

    def __init__(self,
                 mc_forward_num: int = 5,
                 confidence_threshold: float = 0.8,
                 uncertainty_ratio: float = 0.3):
        super(UncertaintySampleExtractor, self).__init__()
        self.mc_forward_num = mc_forward_num
        self.confidence_threshold = confidence_threshold
        self.uncertainty_ratio = uncertainty_ratio

    def extract_samples(self,
                       all_logits: torch.Tensor,
                       labels: torch.Tensor,
                       valid_mask: torch.Tensor = None):
        """
        Extract one positive sample and two negative samples per batch
        Args:
            all_logits: [mc_forward_num, batch_size, num_queries, num_classes]
            labels: [batch_size, num_queries]
            valid_mask: [batch_size, num_queries]
        """
        batch_size = labels.shape[0]
        device = labels.device

        if valid_mask is None:
            valid_mask = (labels != -100)

        # Compute mean predictions and uncertainty
        all_probs = F.softmax(all_logits, dim=-1)
        mean_probs = torch.mean(all_probs, dim=0)
        uncertainty = torch.var(all_probs, dim=0).mean(dim=-1)

        # Get predictions
        predicted_classes = torch.argmax(mean_probs, dim=-1)
        confidence = torch.max(mean_probs, dim=-1)[0]

        positive_samples = []
        negative_samples = []

        for b in range(batch_size):
            valid_indices = torch.where(valid_mask[b])[0]
            if len(valid_indices) == 0:
                continue

            # Positive sample: high confidence correct prediction
            correct_mask = (predicted_classes[b] == labels[b]) & valid_mask[b]
            high_conf_correct = correct_mask & (confidence[b] > self.confidence_threshold)

            if high_conf_correct.any():
                pos_candidates = torch.where(high_conf_correct)[0]
                # Select the most confident correct prediction
                best_conf_idx = pos_candidates[torch.argmax(confidence[b][pos_candidates])]
                positive_samples.append((b, best_conf_idx.item()))

            # Negative samples: high uncertainty predictions
            uncertainty_b = uncertainty[b][valid_mask[b]]
            if len(uncertainty_b) >= 2:
                # Get top 2 most uncertain samples
                _, uncertain_indices = torch.topk(uncertainty_b, min(2, len(uncertainty_b)))
                valid_indices_list = valid_indices[uncertain_indices]
                for idx in valid_indices_list:
                    negative_samples.append((b, idx.item()))

        return {
            'positive_samples': positive_samples,
            'negative_samples': negative_samples,
            'mean_logits': torch.mean(all_logits, dim=0),
            'uncertainty': uncertainty,
            'all_logits': all_logits
        }


def create_lane_uncertainty_head(input_dim, num_classes, config=None):
    """
    创建车道线分类的不确定性头部

    Args:
        input_dim: 输入特征维度
        num_classes: 类别数量
        config: 配置字典

    Returns:
        UncertaintyClassificationHead实例
    """
    if config is None:
        config = {}

    return UncertaintyClassificationHead(
        input_dim=input_dim,
        num_classes=num_classes,
        mc_forward_num=config.get('mc_forward_num', 5),
        dropout_rate=config.get('dropout_rate', 0.1),
        uncertainty_threshold=config.get('uncertainty_threshold', 0.5)
    )


def create_lane_sample_extractor(config=None):
    """
    创建车道线样本提取器

    Args:
        config: 配置字典

    Returns:
        UncertaintySampleExtractor实例
    """
    if config is None:
        config = {}

    return UncertaintySampleExtractor(
        mc_forward_num=config.get('mc_forward_num', 5),
        confidence_threshold=config.get('confidence_threshold', 0.8),
        uncertainty_ratio=config.get('uncertainty_ratio', 0.3)
    )
