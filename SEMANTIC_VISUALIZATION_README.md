# MapTRv2 语义分割可视化使用指南

## 📋 脚本概述

`vis_pred_semantic.py` 是基于MapTRv2的可视化脚本，增强了语义分割叠加功能，可以同时显示：
- 向量地图预测结果（车道线、道路边界、人行横道）
- 语义分割结果叠加（可选）

## 🔧 环境准备

### 1. 运行设置脚本
```bash
cd /home/ubuntunew/model/MapTR-maptrvnew124
python setup_semantic_visualization.py
```

### 2. 验证文件结构
```bash
python test_semantic_visualization.py
```

## 🚀 使用方法

### 基本使用（仅向量地图可视化）
```bash
cd /home/ubuntunew/model/MapTR-maptrvnew124

python tools/maptr/vis_pred_semantic.py \
  projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
  /path/to/your/checkpoint.pth
```

### 启用语义分割可视化
```bash
python tools/maptr/vis_pred_semantic.py \
  projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
  /path/to/your/checkpoint.pth \
  --enable-semantic \
  --semantic-alpha 0.3 \
  --save-separate-masks
```

### 指定输出目录
```bash
python tools/maptr/vis_pred_semantic.py \
  projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
  /path/to/your/checkpoint.pth \
  --enable-semantic \
  --show-dir ./my_visualization_results
```

## 📁 输入文件路径

### 必需文件

1. **配置文件**: `projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py`
   - 已包含数据集路径：`/home/ubuntunew/model/nuscences-mini/data/nuscenes/`

2. **Checkpoint文件**: 训练好的MapTRv2模型权重
   - 例如：`work_dirs/maptrv2_nusc_r50_24ep/latest.pth`
   - 或其他.pth文件路径

3. **nuScenes数据集**:
   - 路径：`/home/ubuntunew/model/nuscences-mini/data/nuscenes/`
   - 版本：v1.0-mini
   - 需要包含：samples/, sweeps/, maps/, v1.0-mini/ 等目录

### 可选文件（语义分割）

1. **HRNet TensorRT引擎** (可选):
   - 路径：`tools/semantic_mapping/hrnet/assets/seg_weights/hrnet-avl-map.engine`
   - 如果不存在，将使用简易语义分割进行演示

2. **语义分割配置**:
   - 路径：`tools/semantic_mapping/config/config_65.json`
   - 自动从SemVecNet复制

## 🎨 输出结果

脚本会在指定目录生成以下文件：

### 基础输出
- `CAM_FRONT.jpg` - 原始前置相机图像
- `CAM_FRONT_LEFT.jpg` - 原始左前相机图像
- `CAM_FRONT_RIGHT.jpg` - 原始右前相机图像
- `CAM_BACK.jpg` - 原始后置相机图像
- `CAM_BACK_LEFT.jpg` - 原始左后相机图像
- `CAM_BACK_RIGHT.jpg` - 原始右后相机图像
- `surround_view.jpg` - 6相机全景拼接图
- `GT_fixednum_pts_MAP.png` - 真值向量地图
- `PRED_MAP_plot.png` - 预测向量地图

### 语义分割增强输出（启用--enable-semantic时）
- `CAM_FRONT_semantic_overlay.jpg` - 语义分割叠加结果
- `CAM_FRONT_semantic_mask.jpg` - 纯语义分割mask（启用--save-separate-masks时）
- `surround_semantic_view.jpg` - 带语义分割的全景图

## 🎯 语义分割颜色编码

| 类别 | 颜色 (RGB) | 说明 |
|------|------------|------|
| 道路 | (128, 64, 128) | 紫色 |
| 人行横道 | (140, 140, 200) | 蓝色 |
| 车道线 | (255, 255, 255) | 白色 |
| 建筑物 | (70, 130, 180) | 蓝色 |
| 植被 | (107, 142, 35) | 绿色 |

## ⚙️ 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `config` | `projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py` | 模型配置文件 |
| `checkpoint` | `maptrv2_nusc_r50_24e.pth` | 模型权重文件 |
| `--enable-semantic` | False | 启用语义分割可视化 |
| `--semantic-alpha` | 0.3 | 叠加透明度 (0-1) |
| `--save-separate-masks` | False | 保存独立的语义分割mask |
| `--show-dir` | 自动生成 | 输出目录 |
| `--score-thresh` | 0.4 | 预测置信度阈值 |

## 🔍 故障排除

### 问题1: 找不到checkpoint文件
```
错误: [Errno 2] No such file or directory: 'maptrv2_nusc_r50_24e.pth'
```
**解决**:
```bash
# 使用完整路径
python tools/maptr/vis_pred_semantic.py \
  projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
  /home/ubuntunew/model/MapTR-maptrvnew/maptrv2_nusc_r50_24e.pth
```

### 问题2: 语义分割模块不可用
```
警告: 无法导入语义分割模块
```
**解决**: 脚本会自动使用简易语义分割进行演示，不影响基本功能。

### 问题3: 内存不足
**解决**: 减少批次大小或调整图像分辨率。

## 📊 性能对比

| 功能 | 原版vis_pred.py | 语义分割增强版 |
|------|---------------|----------------|
| 向量地图可视化 | ✅ | ✅ |
| 相机图像可视化 | ✅ | ✅ |
| 语义分割叠加 | ❌ | ✅ |
| 全景图生成 | ✅ | ✅ (增强版) |
| 处理速度 | 快 | 中等 (+语义分割时间) |

## 🎬 示例命令

### 快速测试（不使用语义分割）
```bash
python tools/maptr/vis_pred_semantic.py
```

### 完整可视化（包含语义分割）
```bash
python tools/maptr/vis_pred_semantic.py \
  --enable-semantic \
  --semantic-alpha 0.4 \
  --save-separate-masks \
  --show-dir ./semantic_vis_results
```

### 自定义路径
```bash
python tools/maptr/vis_pred_semantic.py \
  projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py \
  /path/to/checkpoint.pth \
  --enable-semantic \
  --show-dir /path/to/output
```

---

**注意**: 确保nuScenes数据集路径在配置文件中正确设置，并且checkpoint文件存在。


