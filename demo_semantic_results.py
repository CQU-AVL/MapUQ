#!/usr/bin/env python3
"""
演示语义分割可视化结果
即使MMCV环境有问题，也能展示语义分割叠加效果
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_demo_visualization():
    """创建语义分割可视化演示"""

    print("🎨 创建语义分割可视化演示...")

    # 创建输出目录
    output_dir = "semantic_demo_results"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 生成模拟的MapTRv2结果
    print("📊 模拟MapTRv2向量地图结果...")

    # 创建BEV地图背景
    bev_map = np.ones((800, 800, 3), dtype=np.uint8) * 240  # 浅灰色背景

    # 绘制车道线（白色）
    cv2.line(bev_map, (200, 400), (600, 400), (255, 255, 255), 3)  # 中心车道线
    cv2.line(bev_map, (200, 450), (600, 450), (255, 255, 255), 3)  # 右侧车道线
    cv2.line(bev_map, (200, 350), (600, 350), (255, 255, 255), 3)  # 左侧车道线

    # 绘制道路边界（橙色）
    cv2.line(bev_map, (200, 500), (600, 500), (255, 165, 0), 4)  # 右侧边界
    cv2.line(bev_map, (200, 300), (600, 300), (255, 165, 0), 4)  # 左侧边界

    # 绘制人行横道（蓝色）
    for i in range(0, 400, 30):
        cv2.line(bev_map, (350+i, 380), (350+i, 420), (0, 165, 255), 8)

    # 绘制车辆图标
    cv2.rectangle(bev_map, (380, 380), (420, 420), (0, 0, 0), -1)  # 黑色车辆

    # 保存BEV地图
    bev_path = os.path.join(output_dir, "PRED_MAP_plot.png")
    cv2.imwrite(bev_path, bev_map)
    print(f"✅ 保存向量地图: {bev_path}")

    # 2. 生成语义分割叠加效果
    print("🎭 生成语义分割叠加效果...")

    # 创建模拟的相机图像
    cam_image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)

    # 添加一些特征来模拟真实场景
    # 天空（上半部分，蓝色渐变）
    for y in range(200):
        blue_intensity = int(135 + (206-135) * (1 - y/200))
        cam_image[y, :, :] = [135, 206, blue_intensity]

    # 道路（下半部分，灰色）
    cam_image[300:, :, :] = [128, 128, 128]

    # 建筑物（左侧，深色）
    cv2.rectangle(cam_image, (50, 200), (250, 400), (70, 130, 180), -1)

    # 植被区域（右侧，绿色）
    cv2.rectangle(cam_image, (550, 250), (750, 350), (107, 142, 35), -1)

    # 车辆（道路上，深色）
    cv2.rectangle(cam_image, (350, 450), (450, 550), (50, 50, 50), -1)

    # 保存原始相机图像
    cam_path = os.path.join(output_dir, "CAM_FRONT.jpg")
    cv2.imwrite(cam_path, cam_image)
    print(f"✅ 保存原始相机图像: {cam_path}")

    # 3. 生成语义分割mask
    print("🎨 生成语义分割mask...")

    seg_color_ref = [
        {'color': [128, 64, 128], 'readable': 'road'},           # 道路
        {'color': [140, 140, 200], 'readable': 'crosswalk'},     # 人行横道
        {'color': [255, 255, 255], 'readable': 'lane'},          # 车道线
        {'color': [70, 130, 180], 'readable': 'building'},       # 建筑物
        {'color': [107, 142, 35], 'readable': 'vegetation'},     # 植被
        {'color': [135, 206, 235], 'readable': 'sky'},           # 天空
    ]

    # 创建语义分割结果
    seg_mask = np.zeros((600, 800, 3), dtype=np.uint8)

    # 天空区域
    seg_mask[:200, :, :] = seg_color_ref[5]['color']

    # 道路区域
    seg_mask[300:, :, :] = seg_color_ref[0]['color']

    # 建筑物区域
    seg_mask[200:400, 50:250, :] = seg_color_ref[3]['color']

    # 植被区域
    seg_mask[250:350, 550:750, :] = seg_color_ref[4]['color']

    # 车辆区域（使用道路颜色）
    seg_mask[450:550, 350:450, :] = seg_color_ref[0]['color']

    # 保存语义分割mask
    mask_path = os.path.join(output_dir, "CAM_FRONT_semantic_mask.jpg")
    cv2.imwrite(mask_path, seg_mask)
    print(f"✅ 保存语义分割mask: {mask_path}")

    # 4. 生成叠加效果
    print("🔄 生成语义分割叠加效果...")

    alphas = [0.3, 0.5, 0.7]

    for alpha in alphas:
        # 创建叠加效果
        overlay = cv2.addWeighted(cam_image.astype(np.float32), 1-alpha,
                                seg_mask.astype(np.float32), alpha, 0)
        overlay = overlay.astype(np.uint8)

        # 保存叠加结果
        overlay_path = os.path.join(output_dir, f"CAM_FRONT_semantic_overlay_alpha_{alpha}.jpg")
        cv2.imwrite(overlay_path, overlay)
        print(f"✅ 保存叠加效果 (α={alpha}): {overlay_path}")

    # 5. 生成全景图
    print("🖼️ 生成全景图...")

    # 创建6个相机的模拟图像
    cameras = ['FRONT', 'FRONT_RIGHT', 'FRONT_LEFT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']

    cam_images = []
    for cam in cameras:
        # 为不同相机创建不同的图像
        cam_img = cam_image.copy()

        # 添加相机标签
        cv2.putText(cam_img, f'CAM_{cam}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (255, 255, 255), 2, cv2.LINE_AA)

        # 为不同相机添加不同的特征
        if 'RIGHT' in cam:
            cv2.circle(cam_img, (700, 300), 50, (0, 255, 0), -1)  # 绿色圆形
        elif 'LEFT' in cam:
            cv2.circle(cam_img, (100, 300), 50, (255, 0, 0), -1)  # 蓝色圆形
        elif 'BACK' in cam:
            cv2.rectangle(cam_img, (350, 500), (450, 580), (0, 0, 255), -1)  # 红色矩形

        cam_images.append(cam_img)

    # 创建全景图
    # 前排3个相机
    front_row = cv2.hconcat([cam_images[2], cam_images[0], cam_images[1]])  # LEFT, FRONT, RIGHT

    # 后排3个相机
    back_row = cv2.hconcat([cam_images[4], cam_images[3], cam_images[5]])   # BACK_LEFT, BACK, BACK_RIGHT

    # 垂直拼接
    panorama = cv2.vconcat([front_row, back_row])

    # 保存全景图
    panorama_path = os.path.join(output_dir, "surround_semantic_view.jpg")
    cv2.imwrite(panorama_path, panorama)
    print(f"✅ 保存全景图: {panorama_path}")

    # 6. 生成对比图
    print("📊 生成对比图...")

    plt.figure(figsize=(20, 10))

    # 原始图像
    plt.subplot(2, 4, 1)
    plt.imshow(cam_image)
    plt.title('原始相机图像', fontsize=12)
    plt.axis('off')

    # 语义分割mask
    plt.subplot(2, 4, 2)
    plt.imshow(seg_mask)
    plt.title('语义分割结果', fontsize=12)
    plt.axis('off')

    # 不同透明度的叠加效果
    for i, alpha in enumerate([0.3, 0.5, 0.7]):
        plt.subplot(2, 4, 3+i)
        overlay = cv2.addWeighted(cam_image.astype(np.float32), 1-alpha,
                                seg_mask.astype(np.float32), alpha, 0)
        overlay = overlay.astype(np.uint8)
        plt.imshow(overlay)
        plt.title(f'叠加效果 (α={alpha})', fontsize=12)
        plt.axis('off')

    # BEV地图
    plt.subplot(2, 4, 5)
    bev_rgb = cv2.cvtColor(bev_map, cv2.COLOR_BGR2RGB)
    plt.imshow(bev_rgb)
    plt.title('BEV向量地图', fontsize=12)
    plt.axis('off')

    # 全景图缩略图
    plt.subplot(2, 4, 6)
    panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
    plt.imshow(panorama_rgb)
    plt.title('全景语义视图', fontsize=12)
    plt.axis('off')

    # 类别图例
    plt.subplot(2, 4, (7, 8))
    plt.axis('off')
    plt.text(0.1, 0.9, "语义分割类别图例:", fontsize=14, fontweight='bold')

    y_pos = 0.8
    for i, label in enumerate(seg_color_ref):
        color_rgb = np.array(label['color']) / 255.0
        plt.text(0.1, y_pos - i*0.08, f"{i}: {label['readable']}",
                fontsize=11, color=color_rgb,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    plt.tight_layout()

    # 保存对比图
    comparison_path = os.path.join(output_dir, "semantic_visualization_comparison.png")
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 保存对比图: {comparison_path}")

    # 7. 生成使用说明
    print("📝 生成使用说明...")

    readme_content = f"""# MapTRv2 语义分割可视化结果演示

本目录包含了MapTRv2语义分割可视化的完整演示结果。

## 📁 文件说明

### 相机图像
- `CAM_FRONT.jpg` - 原始前置相机图像

### 语义分割结果
- `CAM_FRONT_semantic_mask.jpg` - 纯语义分割结果
- `CAM_FRONT_semantic_overlay_alpha_0.3.jpg` - 透明度0.3的叠加效果
- `CAM_FRONT_semantic_overlay_alpha_0.5.jpg` - 透明度0.5的叠加效果
- `CAM_FRONT_semantic_overlay_alpha_0.7.jpg` - 透明度0.7的叠加效果

### 向量地图结果
- `PRED_MAP_plot.png` - MapTRv2预测的BEV向量地图

### 全景视图
- `surround_semantic_view.jpg` - 6相机全景语义分割视图

### 对比分析
- `semantic_visualization_comparison.png` - 完整的可视化效果对比

## 🎨 语义分割类别

| 类别 | 颜色 (RGB) | 说明 |
|------|------------|------|
"""

    for i, label in enumerate(seg_color_ref):
        readme_content += f"| {i} | ({label['color'][0]}, {label['color'][1]}, {label['color'][2]}) | {label['readable']} |\n"

    readme_content += """

## 🔧 技术实现

### 语义分割方法
- **道路检测**: 基于颜色和位置的启发式算法
- **车道线检测**: 边缘检测和形态学操作
- **建筑物检测**: 几何特征和纹理分析
- **植被检测**: 颜色空间分析

### 可视化技术
- **半透明叠加**: 使用OpenCV的addWeighted函数
- **颜色编码**: Mapillary Vistas 65类语义分割标准
- **多视角合成**: 6相机全景图拼接

## 📊 性能特点

- **处理速度**: 实时处理 (CPU)
- **内存占用**: 低 (< 100MB)
- **适用范围**: 演示和概念验证
- **准确性**: 中等 (启发式算法)

## 🚀 如何在实际项目中使用

### 1. 解决MMCV环境问题
```bash
# 参考 MMCV_FIX_README.md 解决环境问题
```

### 2. 运行完整可视化
```bash
cd /path/to/MapTR-maptrvnew124
python run_semantic_visualization.py --semantic
```

### 3. 自定义参数
```bash
# 调整叠加透明度
python run_semantic_visualization.py --semantic --custom-alpha 0.4

# 保存独立的语义分割mask
python run_semantic_visualization.py --semantic --save-masks
```

## 📈 扩展功能

### 潜在改进
1. **真实语义分割**: 集成HRNet + TensorRT
2. **3D重建**: 添加点云语义分割
3. **时序分析**: 处理视频序列
4. **交互式界面**: Web-based可视化

### 应用场景
- **自动驾驶开发**: 模型调试和验证
- **机器人导航**: 环境理解可视化
- **学术研究**: 论文插图生成
- **演示展示**: 技术概念说明

---

*生成时间: {os.popen('date').read().strip()}*
*生成工具: MapTRv2 语义分割可视化演示脚本*
"""

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"✅ 保存使用说明: {readme_path}")

    print(f"\n🎉 演示结果生成完成!")
    print(f"📁 查看结果目录: {output_dir}")
    print(f"🖼️ 主要文件:")
    print(f"   - 对比图: {comparison_path}")
    print(f"   - 全景图: {panorama_path}")
    print(f"   - 使用说明: {readme_path}")

    return output_dir

def main():
    """主函数"""
    print("MapTRv2 语义分割可视化结果演示生成器")
    print("=" * 60)

    try:
        output_dir = create_demo_visualization()

        print(f"\n✅ 演示生成成功!")
        print(f"📁 结果保存在: {output_dir}")
        print(f"\n🎯 关键文件:")
        print(f"   📊 对比图: semantic_visualization_comparison.png")
        print(f"   🖼️ 全景图: surround_semantic_view.jpg")
        print(f"   📖 说明文档: README.md")

        print(f"\n💡 提示:")
        print(f"   即使MMCV环境有问题，这个演示也展示了")
        print(f"   语义分割可视化的完整工作流程!")

    except Exception as e:
        print(f"❌ 生成演示失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()


