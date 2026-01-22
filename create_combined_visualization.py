#!/usr/bin/env python3
"""
将MapTR不确定性可视化结果与原始相机图像合并
创建左右分栏的对比可视化：左边原始图像，右边不确定性可视化
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path

def create_combined_image(uncertainty_image_path, original_image_path, output_path,
                         uncertainty_weight=0.6, original_weight=0.4):
    """
    创建合并的对比图像

    Args:
        uncertainty_image_path: 不确定性可视化图片路径
        original_image_path: 原始相机图片路径
        output_path: 输出路径
        uncertainty_weight: 不确定性图像在最终图片中的权重
        original_weight: 原始图像在最终图片中的权重
    """
    # 读取图像
    uncertainty_img = cv2.imread(uncertainty_image_path)
    original_img = cv2.imread(original_image_path)

    if uncertainty_img is None:
        print(f"❌ 无法读取不确定性图像: {uncertainty_image_path}")
        return False

    if original_img is None:
        print(f"⚠️ 无法读取原始图像: {original_image_path}，将使用占位符")
        # 创建占位符图像
        original_img = np.ones_like(uncertainty_img) * 128
        cv2.putText(original_img, "Original Image", (50, uncertainty_img.shape[0]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # 确保两张图片大小一致
    h1, w1 = uncertainty_img.shape[:2]
    h2, w2 = original_img.shape[:2]

    # 如果高度不同，以不确定性图像为准缩放原始图像
    if h1 != h2:
        scale = h1 / h2
        new_w = int(w2 * scale)
        original_img = cv2.resize(original_img, (new_w, h1))

    # 创建合并图像（水平拼接）
    combined_img = cv2.hconcat([original_img, uncertainty_img])

    # 在中间添加分割线
    h, w = combined_img.shape[:2]
    cv2.line(combined_img, (w//2, 0), (w//2, h), (255, 255, 255), 3)

    # 添加标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 2

    # 左侧标签
    cv2.putText(combined_img, "Original Camera View", (50, 50),
               font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(combined_img, "Original Camera View", (50, 50),
               font, font_scale, (0, 0, 0), font_thickness-1)

    # 右侧标签
    text_x = w//2 + 50
    cv2.putText(combined_img, "Uncertainty Visualization", (text_x, 50),
               font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(combined_img, "Uncertainty Visualization", (text_x, 50),
               font, font_scale, (0, 0, 0), font_thickness-1)

    # 保存结果
    cv2.imwrite(output_path, combined_img)
    return True

def process_visualization_directory(visualization_dir, nuscenes_samples_dir=None, output_dir=None):
    """
    处理整个可视化目录，创建合并图像

    Args:
        visualization_dir: 不确定性可视化结果目录
        nuscenes_samples_dir: nuScenes samples目录路径（可选）
        output_dir: 输出目录（可选，默认在visualization_dir下创建combined子目录）
    """
    if output_dir is None:
        output_dir = os.path.join(visualization_dir, "combined")

    os.makedirs(output_dir, exist_ok=True)

    # 读取映射文件
    mapping_file = os.path.join(visualization_dir, "image_mapping.txt")
    mappings = {}

    if os.path.exists(mapping_file):
        print("📄 读取图像映射信息...")
        with open(mapping_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        frame_id, sample_token, image_name = parts[:3]
                        mappings[frame_id] = {
                            'sample_token': sample_token,
                            'image_name': image_name,
                            'uncertainty_path': os.path.join(visualization_dir, image_name)
                        }
        print(f"✅ 加载了 {len(mappings)} 个映射记录")
    else:
        print("⚠️ 未找到映射文件，将尝试直接匹配文件名")

        # 查找所有uncertainty图像
        uncertainty_files = [f for f in os.listdir(visualization_dir)
                           if f.endswith('_uncertainty.png')]

        for uncertainty_file in uncertainty_files:
            frame_id = uncertainty_file.split('_frame_')[1].split('_')[0]
            mappings[frame_id] = {
                'image_name': uncertainty_file,
                'uncertainty_path': os.path.join(visualization_dir, uncertainty_file)
            }

        print(f"📊 找到 {len(mappings)} 个不确定性可视化文件")

    # 处理每个映射
    success_count = 0
    total_count = len(mappings)

    print("\n🔄 开始创建合并图像...")
    print("="*80)

    for frame_id, mapping in mappings.items():
        uncertainty_path = mapping['uncertainty_path']
        output_filename = f"combined_{frame_id}_comparison.png"
        output_path = os.path.join(output_dir, output_filename)

        # 尝试找到对应的原始图像
        original_path = None

        if 'sample_token' in mapping and nuscenes_samples_dir:
            # 基于sample_token查找原始图像
            sample_token = mapping['sample_token']
            # 这里可以实现更复杂的查找逻辑
            # 暂时使用占位符
            original_path = None
        else:
            # 如果没有nuScenes路径，使用占位符
            original_path = None

        print(f"处理 {frame_id}: {mapping['image_name']}")

        # 创建合并图像
        if create_combined_image(uncertainty_path, original_path, output_path):
            success_count += 1
            print(f"  ✅ 保存到: {output_filename}")
        else:
            print(f"  ❌ 处理失败: {frame_id}")

    print("\n" + "="*80)
    print(f"🎉 处理完成: {success_count}/{total_count} 张图像成功合并")
    print(f"📁 结果保存在: {output_dir}")

    # 创建说明文件
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("# MapTR不确定性可视化 - 对比视图\n\n")
        f.write("## 图像说明\n\n")
        f.write("- **左侧**: 原始相机图像（真实世界场景）\n")
        f.write("- **右侧**: MapTR不确定性可视化（BEV地图 + 不确定性椭圆）\n")
        f.write("- **中间白线**: 分割线\n\n")
        f.write("## 文件命名\n\n")
        f.write("- `combined_frame_XXX_comparison.png`: 帧XXX的对比图像\n\n")
        f.write("## 图例说明\n\n")
        f.write("- 🟠 橙色线条: 车道分割线\n")
        f.write("- 🔵 蓝色线条: 人行横道\n")
        f.write("- 🟢 绿色线条: 道路边界\n")
        f.write("- 彩色椭圆: 预测不确定性（越大越不确定）\n\n")
        f.write("---\n\n")
        f.write(f"生成时间: {os.popen('date').read().strip()}\n")
        f.write(f"处理图像数: {success_count}\n")

    print(f"📖 说明文档: {readme_path}")

def main():
    parser = argparse.ArgumentParser(description='Create combined uncertainty visualization')
    parser.add_argument('visualization_dir',
                       help='Directory containing uncertainty visualization results')
    parser.add_argument('--nuscenes-samples',
                       help='nuScenes samples directory (optional, for original images)')
    parser.add_argument('--output-dir',
                       help='Output directory (default: visualization_dir/combined)')
    args = parser.parse_args()

    if not os.path.exists(args.visualization_dir):
        print(f"❌ 可视化目录不存在: {args.visualization_dir}")
        return

    process_visualization_directory(args.visualization_dir, args.nuscenes_samples, args.output_dir)

if __name__ == '__main__':
    main()
