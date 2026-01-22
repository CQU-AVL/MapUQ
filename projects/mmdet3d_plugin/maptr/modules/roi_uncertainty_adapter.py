import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIUncertaintyAdapter(nn.Module):
    """
    Progressive ROI 尺度自适应模块（解码器层间传递）。
    思路：
      - 输入当前层的点位预测与 GT，计算偏移误差（L2）。
      - 将误差映射为缩放因子，用于更新下一层 ROI 尺度。
      - 可选使用温度与上下界裁剪，防止尺度过大/过小。
    使用方式（建议在 decoder 层循环中调用）：
      adapter = ROIUncertaintyAdapter(...)
      next_scales = adapter(pred_pts, gt_pts, prev_scales)
      # 将 next_scales 作为下一层解码的 ROI 尺度（与 deformable decoder 配合）
    """

    def __init__(self,
                 alpha: float = 0.25,
                 min_scale: float = 0.3,
                 max_scale: float = 1.5,
                 temperature: float = 1.0,
                 eps: float = 1e-6):
        """
        Args:
            alpha: 缩放强度系数，误差越大，尺度放缩越明显。
            min_scale: 尺度下限，避免收缩为 0。
            max_scale: 尺度上限，避免无限放大。
            temperature: 温度系数，控制误差到尺度的敏感度。
            eps: 数值稳定项。
        """
        super().__init__()
        self.alpha = alpha
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.temperature = temperature
        self.eps = eps

    @torch.no_grad()
    def forward(self,
                pred_pts: torch.Tensor,
                gt_pts: torch.Tensor,
                prev_scales: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_pts: 预测点位 [bs, num_q, num_pts, 2]
            gt_pts:   GT 点位   [bs, num_q, num_pts, 2]，若无 GT 可传 None
            prev_scales: 上一层 ROI 尺度 [bs, num_q, 1] 或 [bs, num_q, 2]
        Returns:
            next_scales: 下一层 ROI 尺度，shape 与 prev_scales 一致
        """
        # 若无 GT，使用预测点自身的离散程度作为不确定性度量
        if gt_pts is None:
            # [bs, num_q, num_pts, 2] -> std over pts
            spread = torch.std(pred_pts, dim=2)  # [bs, num_q, 2]
            spread = torch.norm(spread, dim=-1, keepdim=True)  # [bs, num_q, 1]
            scaled_err = spread / (self.temperature + self.eps)
            scale_factor = 1.0 + self.alpha * scaled_err
            if prev_scales.shape[-1] == 2:
                scale_factor = scale_factor.expand_as(prev_scales)
            # Ensure same shape as prev_scales
            if scale_factor.dim() > prev_scales.dim():
                scale_factor = scale_factor.squeeze(-1)
            next_scales = prev_scales * scale_factor
            return torch.clamp(next_scales, self.min_scale, self.max_scale)

        # 保证形状对齐
        if pred_pts.shape != gt_pts.shape:
            # 尝试广播 GT
            gt_pts = F.interpolate(
                gt_pts.permute(0, 3, 1, 2),  # [bs, 2, num_q, num_pts]
                size=pred_pts.shape[2:4],
                mode='nearest'
            ).permute(0, 2, 3, 1)

        # L2 误差范数
        offset = pred_pts - gt_pts
        err = torch.norm(offset, dim=-1)  # [bs, num_q, num_pts]
        err_mean = err.mean(dim=-1, keepdim=True)  # [bs, num_q, 1]

        # 温度缩放 + 映射为缩放因子
        scaled_err = err_mean / (self.temperature + self.eps)
        scale_factor = 1.0 + self.alpha * scaled_err

        # 如果 prev_scales 是 [bs, num_q, 2]，保持维度
        if prev_scales.shape[-1] == 2:
            scale_factor = scale_factor.expand_as(prev_scales)

        next_scales = prev_scales * scale_factor
        next_scales = torch.clamp(next_scales, self.min_scale, self.max_scale)
        return next_scales

