import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIScaleConsistencyLoss(nn.Module):
    """
    ROI 尺度一致性约束：
      - 鼓励相邻 decoder 层的尺度变化不要过于剧烈。
      - 可选与误差相关的目标尺度，引导尺度收缩/扩张。
    """

    def __init__(self, weight: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self,
                prev_scales: torch.Tensor,
                next_scales: torch.Tensor,
                target_scales: torch.Tensor = None):
        """
        Args:
            prev_scales: 上一层尺度 [bs, num_q, dim]
            next_scales: 当前层尺度 [bs, num_q, dim]
            target_scales: 可选的期望尺度 [bs, num_q, dim]
        """
        if target_scales is None:
            loss = F.l1_loss(next_scales, prev_scales, reduction=self.reduction)
        else:
            loss = F.l1_loss(next_scales, target_scales, reduction=self.reduction)
        return loss * self.weight


def create_roi_scale_loss(cfg=None):
    if cfg is None:
        cfg = {}
    return ROIScaleConsistencyLoss(
        weight=cfg.get("weight", 1.0),
        reduction=cfg.get("reduction", "mean"),
    )

