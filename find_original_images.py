#!/usr/bin/env python3
"""
根据可视化结果查找对应的原始nuScenes图像
"""

import os
import argparse
from pathlib import Path

def find_original_images(visualization_dir, nuscenes_root=None):
    """
    根据可视化结果目录查找对应的原始图像

    Args:
        visualization_dir: 可视化结果目录
        nuscenes_root: nuScenes数据集根目录（可选）
    """
    mapping_file = os.path.join(visualization_dir, "image_mapping.txt")

    if not os.path.exists(mapping_file):
        print(f"❌ 映射文件不存在: {mapping_file}")
        print("请确保在生成可视化时使用了包含映射功能的脚本版本")
        return

    print("🔍 读取图像映射信息..."    print(f"📁 可视化目录: {visualization_dir}")
    print(f"📄 映射文件: {mapping_file}")
    print("="*80)

    with open(mapping_file, 'r') as f:
        lines = f.readlines()

    # 跳过注释行
    data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('#')]

    print(f"📊 找到 {len(data_lines)} 个映射记录")
    print()

    # 显示前几个映射示例
    print("📋 映射记录示例:")
    for i, line in enumerate(data_lines[:5]):
        if '\t' in line:
            parts = line.split('\t')
            if len(parts) >= 3:
                frame_id, sample_token, image_name = parts[:3]
                print(f"  {frame_id} -> {sample_token} -> {image_name}")

    if len(data_lines) > 5:
        print(f"  ... 还有 {len(data_lines) - 5} 条记录")
    print()

    # 如果提供了nuScenes根目录，尝试查找实际的图像文件
    if nuscenes_root and os.path.exists(nuscenes_root):
        print("🏠 尝试查找原始nuScenes图像文件...")
        print(f"📂 数据集根目录: {nuscenes_root}")

        samples_dir = os.path.join(nuscenes_root, "samples")
        if os.path.exists(samples_dir):
            print("✅ 找到samples目录")

            # 统计每种相机类型的图像
            cam_types = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

            total_images = 0
            for cam_type in cam_types:
                cam_dir = os.path.join(samples_dir, cam_type)
                if os.path.exists(cam_dir):
                    image_count = len([f for f in os.listdir(cam_dir) if f.endswith('.jpg')])
                    print(f"  📷 {cam_type}: {image_count} 张图像")
                    total_images += image_count

            print(f"📊 总共找到 {total_images} 张原始相机图像")
        else:
            print("❌ 未找到samples目录")
    else:
        print("ℹ️ 未提供nuScenes根目录，跳过原始图像查找")
        if not nuscenes_root:
            print("💡 提示: 如需查找原始图像，请提供 --nuscenes-root 参数")

    print()
    print("🔗 如何查看原始图像:")
    print("1. 使用 nuScenes-devkit 加载数据:")
    print("   from nuscenes.nuscenes import NuScenes")
    print("   nusc = NuScenes(version='v1.0-mini', dataroot='/path/to/nuscenes')")
    print()
    print("2. 根据sample_token获取样本:")
    print("   sample = nusc.get('sample', 'your_sample_token_here')")
    print()
    print("3. 获取相机数据:")
    print("   cam_data = sample['data']['CAM_FRONT']")
    print("   cam_record = nusc.get('sample_data', cam_data)")
    print("   image_path = os.path.join(nusc.dataroot, cam_record['filename'])")

def main():
    parser = argparse.ArgumentParser(description='Find original nuScenes images for visualization results')
    parser.add_argument('visualization_dir', help='Visualization results directory')
    parser.add_argument('--nuscenes-root', help='nuScenes dataset root directory')
    args = parser.parse_args()

    if not os.path.exists(args.visualization_dir):
        print(f"❌ 可视化目录不存在: {args.visualization_dir}")
        return

    find_original_images(args.visualization_dir, args.nuscenes_root)

if __name__ == '__main__':
    main()


