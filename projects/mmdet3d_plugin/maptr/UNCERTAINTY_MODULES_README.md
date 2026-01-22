# MapTR不确定性模块使用指南

本文档介绍如何在MapTR项目中使用新添加的不确定性模块，包括车道线分类不确定性模块和语义分割不确定性模块。

## 概述

我们基于GAS+UAUL模型的不确定性估计方法，在MapTR项目中集成了以下功能：

1. **车道线分类不确定性模块**：使用MC Dropout进行车道线分类的不确定性估计和负样本利用
2. **语义分割不确定性模块**：用于边界分类和区域检测的不确定性估计
3. **对应的损失函数**：支持不确定性感知的训练

## 文件结构

```
projects/mmdet3d_plugin/maptr/
├── dense_heads/
│   ├── lane_uncertainty_heads.py           # 车道线负样本利用头部模块
│   ├── segmentation_uncertainty_heads.py  # 语义分割不确定性头部模块
│   └── __init__.py                        # 导出不确定性头部
├── losses/
│   ├── lane_uncertainty_losses.py         # 车道线不确定性损失函数
│   ├── segmentation_uncertainty_losses.py # 语义分割不确定性损失函数
│   └── __init__.py                        # 导出不确定性损失
└── UNCERTAINTY_MODULES_README.md          # 本文档
```

## 核心组件

### 1. 车道线负样本利用头部模块 (lane_uncertainty_heads.py)

#### MCDropoutLayer
MC Dropout层，用于在推理时启用dropout以进行不确定性估计。

```python
from projects.mmdet3d_plugin.maptr.dense_heads import MCDropoutLayer

dropout = MCDropoutLayer(dropout_rate=0.1)
features = dropout(input_features)  # 在训练模式下应用dropout
```

#### UncertaintyClassificationHead
车道线分类的不确定性感知头部。

```python
from projects.mmdet3d_plugin.maptr.dense_heads import UncertaintyClassificationHead

# 创建头部
head = UncertaintyClassificationHead(
    input_dim=256,           # 输入特征维度
    num_classes=3,           # 分类类别数
    mc_forward_num=5,        # MC采样次数
    dropout_rate=0.1,        # dropout率
    uncertainty_threshold=0.5  # 不确定性阈值
)

# 前向传播
features = torch.randn(batch_size, seq_len, 256)
labels = torch.randint(0, 3, (batch_size, seq_len))

# 获取不确定性估计
result = head(features, labels, return_uncertainty=True)
# 返回: {'logits': logits, 'uncertainty': uncertainty, 'entropy': entropy, 'all_logits': all_logits}
```

#### SegmentationUncertaintyWrapper
用于边界分类的不确定性包装器。

```python
from projects.mmdet3d_plugin.maptr.dense_heads import SegmentationUncertaintyWrapper

# 包装原始头部
original_head = YourSegmentationHead()
wrapped_head = SegmentationUncertaintyWrapper(
    original_head,
    mc_forward_num=5,      # MC采样次数
    dropout_rate=0.3,      # dropout率
    temperature=1.0,       # 温度参数
    use_epistemic=True,    # 使用认知不确定性
    use_aleatoric=True,    # 使用偶然不确定性
    use_margin=True        # 使用边界估计
)

# 前向传播
x = torch.randn(batch_size, channels, height, width)
result = wrapped_head(x, labels, return_uncertainty=True)
```

#### AreaDetectionUncertaintyWrapper
用于区域检测（斑马线、箭头）的不确定性包装器。

```python
from projects.mmdet3d_plugin.maptr.dense_heads import AreaDetectionUncertaintyWrapper

# 包装原始头部
original_head = YourAreaDetectionHead()
wrapped_head = AreaDetectionUncertaintyWrapper(
    original_head,
    mc_forward_num=5,          # MC采样次数
    dropout_rate=0.3,          # dropout率
    road_element_type='zebra', # 'zebra' 或 'arrow'
    junction_weight=1.2        # 路口场景权重
)
```

#### 便捷创建函数

```python
from projects.mmdet3d_plugin.maptr.dense_heads import (
    create_uncertainty_head,
    create_lane_uncertainty_head,
    create_lane_sample_extractor
)

# 创建车道线不确定性头部
lane_head = create_lane_uncertainty_head(
    input_dim=256,
    num_classes=3,
    config={'mc_forward_num': 5, 'dropout_rate': 0.1}
)

# 创建分割不确定性头部
seg_head = create_uncertainty_head(
    original_head,
    head_type='edge_cls',  # 或 'zebra_cls', 'arrow_cls'
    config={'mc_forward_num': 5, 'dropout_rate': 0.3}
)

# 创建样本提取器
sample_extractor = create_lane_sample_extractor(
    config={'confidence_threshold': 0.8, 'uncertainty_ratio': 0.3}
)
```

### 2. 语义分割不确定性头部模块 (segmentation_uncertainty_heads.py)

#### SegmentationUncertaintyWrapper
用于边界分类的不确定性包装器。

```python
from projects.mmdet3d_plugin.maptr.dense_heads import SegmentationUncertaintyWrapper

# 包装原始头部
original_head = YourSegmentationHead()
wrapped_head = SegmentationUncertaintyWrapper(
    original_head,
    mc_forward_num=5,      # MC采样次数
    dropout_rate=0.3,      # dropout率
    temperature=1.0,       # 温度参数
    use_epistemic=True,    # 使用认知不确定性
    use_aleatoric=True,    # 使用偶然不确定性
    use_margin=True        # 使用边界估计
)
```

#### AreaDetectionUncertaintyWrapper
用于区域检测（斑马线、箭头）的不确定性包装器。

```python
from projects.mmdet3d_plugin.maptr.dense_heads import AreaDetectionUncertaintyWrapper

# 包装原始头部
original_head = YourAreaDetectionHead()
wrapped_head = AreaDetectionUncertaintyWrapper(
    original_head,
    mc_forward_num=5,          # MC采样次数
    dropout_rate=0.3,          # dropout率
    road_element_type='zebra', # 'zebra' 或 'arrow'
    junction_weight=1.2        # 路口场景权重
)
```

### 3. 车道线不确定性损失函数 (lane_uncertainty_losses.py)

#### LaneUncertaintyLoss
车道线不确定性损失，结合分类损失和不确定性正则化。

```python
from projects.mmdet3d_plugin.maptr.losses import LaneUncertaintyLoss

# 创建损失函数
loss_fn = LaneUncertaintyLoss(
    weight=1.0,                    # 损失权重
    reduction='mean',              # 损失规约方式
    use_epistemic=True,            # 使用认知不确定性
    use_aleatoric=True,            # 使用偶然不确定性
    use_contrast=True,             # 使用对比学习
    uncertainty_weight=0.1,        # 不确定性权重
    negative_sample_weight=1.0     # 负样本权重
)

# 计算损失
predictions = head_output  # 包含不确定性信息的字典
targets = ground_truth_labels
loss_dict = loss_fn(predictions, targets, uncertainty_info)
# 返回: {'total_loss': total, 'basic_loss': basic, 'epistemic_loss': epistemic, ...}
```

#### SegmentationUncertaintyLoss
分割不确定性损失，用于边界分类。

```python
from projects.mmdet3d_plugin.maptr.losses import SegmentationUncertaintyLoss

loss_fn = SegmentationUncertaintyLoss(
    weight=1.0,
    reduction='mean',
    use_epistemic=True,
    use_aleatoric=True,
    use_contrast=True
)

# 计算损失
predictions = torch.randn(batch_size, num_classes, height, width)
targets = torch.randint(0, num_classes, (batch_size, height, width))
uncertainty_info = segmentation_head_uncertainty_output
loss_dict = loss_fn(predictions, targets, uncertainty_info)
```

#### AreaDetectionUncertaintyLoss
区域检测不确定性损失，用于斑马线和箭头分类。

```python
from projects.mmdet3d_plugin.maptr.losses import AreaDetectionUncertaintyLoss

loss_fn = AreaDetectionUncertaintyLoss(
    weight=1.0,
    reduction='mean',
    margin=-0.6,              # 边界参数
    temperature=1.0,          # 温度参数
    calibration_weight=0.05,  # 校准权重
    road_element_type='zebra' # 'zebra' 或 'arrow'
)

# 计算损失
predictions = torch.randn(batch_size, num_classes, height, width)
targets = torch.randint(0, num_classes, (batch_size, height, width))
uncertainty_info = area_detection_head_uncertainty_output
loss_dict = loss_fn(predictions, targets, uncertainty_info)
```

#### MultiTaskUncertaintyLoss
多任务不确定性损失，用于同时处理多个任务。

```python
from projects.mmdet3d_plugin.maptr.losses import MultiTaskUncertaintyLoss

loss_fn = MultiTaskUncertaintyLoss(
    task_weights={
        'cls_scores': 1.0,
        'pts_preds': 2.0,
        'seg': 1.5
    },
    uncertainty_weight=0.1,
    calibration_weight=0.05
)

# 计算多任务损失
predictions_dict = {'task1': pred1, 'task2': pred2}
targets_dict = {'task1': target1, 'task2': target2}
uncertainty_info = multi_task_uncertainty_output
loss_dict = loss_fn(predictions_dict, targets_dict, uncertainty_info)
```

#### 便捷构建函数

```python
from projects.mmdet3d_plugin.maptr.losses import (
    create_lane_uncertainty_loss
)
from projects.mmdet3d_plugin.maptr.losses import (
    build_segmentation_uncertainty_loss
)

# 快速创建车道线损失
lane_loss = create_lane_uncertainty_loss({
    'weight': 1.0,
    'uncertainty_weight': 0.1
})

# 构建语义分割损失
seg_config = {
    'type': 'SegmentationUncertaintyLoss',
    'weight': 1.0,
    'use_epistemic': True
}
seg_loss = build_segmentation_uncertainty_loss(seg_config)
```

## 在MapTR中集成使用

### 1. 修改配置文件

在MapTR的配置文件中添加不确定性模块：

```python
# 在model配置中
model = dict(
    # ... 其他配置 ...
    pts_bbox_head=dict(
        # ... 现有配置 ...
        # 添加不确定性配置
        uncertainty_config=dict(
            use_uncertainty=True,
            mc_forward_num=5,
            dropout_rate=0.1,
            uncertainty_weight=0.1
        )
    )
)
```

### 2. 修改头部实现

在MapTR头部中集成不确定性模块：

```python
from projects.mmdet3d_plugin.maptr.dense_heads import UncertaintyClassificationHead

class MapTRHeadWithUncertainty(MapTRHead):
    def __init__(self, *args, uncertainty_config=None, **kwargs):
        super().__init__(*args, **kwargs)

        if uncertainty_config and uncertainty_config.get('use_uncertainty', False):
            # 为相关分类头添加不确定性包装
            self.lane_cls_uncertainty = UncertaintyClassificationHead(
                input_dim=self.embed_dims,
                num_classes=num_lane_classes,
                **uncertainty_config
            )

    def forward(self, *args, **kwargs):
        # 正常前向传播
        outputs = super().forward(*args, **kwargs)

        # 如果启用不确定性，添加不确定性估计
        if hasattr(self, 'lane_cls_uncertainty'):
            uncertainty_outputs = self.lane_cls_uncertainty(
                lane_features, lane_labels, return_uncertainty=True
            )
            outputs.update(uncertainty_outputs)

        return outputs
```

### 3. 修改损失计算

在训练过程中使用不确定性损失：

```python
from projects.mmdet3d_plugin.maptr.losses import LaneUncertaintyLoss

class MapTRLossWithUncertainty(MapTRLoss):
    def __init__(self, *args, uncertainty_config=None, **kwargs):
        super().__init__(*args, **kwargs)

        if uncertainty_config:
            self.lane_uncertainty_loss = LaneUncertaintyLoss(**uncertainty_config)

    def forward(self, outputs, targets):
        # 正常损失计算
        losses = super().forward(outputs, targets)

        # 添加不确定性损失
        if hasattr(self, 'lane_uncertainty_loss') and 'uncertainty' in outputs:
            uncertainty_loss_dict = self.lane_uncertainty_loss(
                outputs['lane_logits'], targets['lane_labels'], outputs
            )
            losses.update(uncertainty_loss_dict)

        return losses
```

## 训练和推理注意事项

### 训练阶段
1. 在训练时启用`return_uncertainty=True`来获取不确定性估计
2. 不确定性损失会自动与基础分类损失结合
3. MC Dropout在训练时正常工作，在推理时通过多次前向传播获得不确定性

### 推理阶段
1. 对于不确定性估计，需要运行多次前向传播（`mc_forward_num`次）
2. 可以根据不确定性阈值过滤低置信度预测
3. 负样本利用可以提高模型对困难样本的鲁棒性

### 超参数调优
- `mc_forward_num`: 建议5-10，权衡计算成本和不确定性估计质量
- `dropout_rate`: 建议0.1-0.3，影响不确定性估计的方差
- `uncertainty_weight`: 建议0.05-0.2，控制不确定性正则化的强度
- `uncertainty_threshold`: 建议0.5-0.8，根据任务调整

## 示例代码

完整的使用示例请参考项目中的测试文件 `test_uncertainty_modules.py` 和 `test_uncertainty_direct.py`。

## 扩展和定制

这些模块设计为模块化的，可以根据具体需求进行扩展：

1. **自定义不确定性度量**：修改`compute_uncertainty`方法
2. **自定义样本选择策略**：修改`get_positive_negative_samples`方法
3. **添加新的损失项**：在损失类中添加新的损失计算逻辑
4. **集成其他不确定性方法**：如集成深度学习不确定性方法

## 参考文献

- GAS+UAUL: Geometry-aware Semi-supervised Learning for Lane Detection
- MC Dropout: Dropout as a Bayesian Approximation
- Uncertainty Estimation in Deep Learning

## 文件拆分说明

为了更好地组织代码和提高模块化程度，我们将原来的统一不确定性模块文件拆分成了两个独立的模块：

### 1. 车道线负样本利用模块
- **头部**: `lane_uncertainty_heads.py` - 包含车道线分类的不确定性头部和样本提取器
- **损失**: `lane_uncertainty_losses.py` - 包含车道线不确定性损失函数

### 2. 语义分割不确定性模块
- **头部**: `segmentation_uncertainty_heads.py` - 包含分割和区域检测的不确定性头部
- **损失**: `segmentation_uncertainty_losses.py` - 包含分割和区域检测的不确定性损失函数

### 拆分优势
1. **模块化**: 不同类型的任务使用独立的模块，避免耦合
2. **维护性**: 更容易维护和扩展特定类型的功能
3. **导入清晰**: 导入语句更加明确，便于理解使用意图
4. **性能优化**: 可以根据需要只导入使用的模块

### 兼容性
所有API保持向后兼容，原有的导入方式仍然有效，但建议使用新的模块化导入方式。

---

如有问题或需要进一步定制，请参考源代码或联系开发团队。
