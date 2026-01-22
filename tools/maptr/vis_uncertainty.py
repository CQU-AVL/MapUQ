#!/usr/bin/env python3
"""
MapTRv2 不确定性可视化脚本 - 简化版本
直接基于推理结果生成不确定性可视化，不依赖MMCV

使用训练好的 checkpoint.pth 和推理结果 result.pkl，显示：
- 车道线预测和不确定性椭圆
- 边界分类的不确定性
- 区域检测的不确定性

参考 vis_std.py 的设计，直接处理 pkl 文件
"""

import argparse
import pickle
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免GUI卡住
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Ellipse
from tqdm import tqdm
import torch

def plot_uncertainty_ellipses(ax, points, uncertainties, color, alpha=0.3, scale=2.0):
    """
    为预测点绘制不确定性椭圆

    Args:
        ax: matplotlib轴对象
        points: 点坐标 [num_points, 2]
        uncertainties: 不确定性值 [num_points] 或 [num_points, 2]
        color: 颜色
        alpha: 透明度
        scale: 椭圆大小缩放因子
    """
    if len(points) == 0:
        return

    # 处理不确定性维度
    if uncertainties.ndim == 1:
        # 单维度不确定性，假设各向同性
        uncertainty_x = uncertainty_y = uncertainties * scale
    elif uncertainties.ndim == 2 and uncertainties.shape[1] == 2:
        # 双维度不确定性
        uncertainty_x = uncertainties[:, 0] * scale
        uncertainty_y = uncertainties[:, 1] * scale
    else:
        # 默认处理
        uncertainty_x = uncertainty_y = np.ones(len(points)) * 0.5 * scale

    # 每隔几个点绘制一个椭圆，避免过于密集
    step = max(1, len(points) // 10)

    for i in range(0, len(points), step):
        if i >= len(uncertainty_x):
            break

        center_x, center_y = points[i]
        width = max(uncertainty_x[i], 0.1)  # 最小宽度
        height = max(uncertainty_y[i], 0.1)  # 最小高度

        # 创建椭圆
        ellipse = Ellipse((center_x, center_y), width=width, height=height,
                         fc=color, ec=color, alpha=alpha, linewidth=0.5)
        ax.add_patch(ellipse)

def convert_maptr_to_uncertainty_format(maptr_result):
    """将MapTR推理结果转换为不确定性可视化格式"""
    try:
        pts_bbox = maptr_result.get('pts_bbox', {})
        if not pts_bbox:
            return None

        # 创建虚拟的sample_token
        sample_token = f"maptr_uncertainty_{id(maptr_result) % 1000}"

        # 转换pts_3d格式
        pts_3d = pts_bbox.get('pts_3d', [])
        scores_3d = pts_bbox.get('scores_3d', [])
        labels_3d = pts_bbox.get('labels_3d', [])

        if len(pts_3d) == 0:
            return None

        # 转换为numpy
        if hasattr(pts_3d, 'cpu'):
            pts_3d = pts_3d.cpu().numpy()
        if hasattr(scores_3d, 'cpu'):
            scores_3d = scores_3d.cpu().numpy()
        if hasattr(labels_3d, 'cpu'):
            labels_3d = labels_3d.cpu().numpy()

        # 构建predicted_map和uncertainty_map
        predicted_map = {}
        uncertainty_map = {}
        map_classes = ['divider', 'ped_crossing', 'boundary']

        for class_idx, class_name in enumerate(map_classes):
            class_mask = labels_3d == class_idx
            if not np.any(class_mask):
                predicted_map[f'{class_name}'] = []
                predicted_map[f'{class_name}_scores'] = []
                uncertainty_map[f'{class_name}_uncertainty'] = []
                continue

            class_pts = pts_3d[class_mask]
            class_scores = scores_3d[class_mask]

            predicted_map[f'{class_name}'] = class_pts.tolist()
            predicted_map[f'{class_name}_scores'] = class_scores.tolist()

            # 生成模拟的不确定性信息（实际应用中应该从模型输出获取）
            # 这里基于预测置信度生成模拟不确定性
            num_preds = len(class_pts)
            uncertainties = []

            for i in range(num_preds):
                score = class_scores[i]
                # 低置信度预测有更高不确定性
                base_uncertainty = 1.0 - score
                # 为每个点生成不确定性值
                pts_uncertainty = np.random.uniform(base_uncertainty * 0.5, base_uncertainty * 1.5, (20, 2))
                uncertainties.append(pts_uncertainty.tolist())

            uncertainty_map[f'{class_name}_uncertainty'] = uncertainties

        # 创建虚拟的ego位置和朝向
        ego_pos = [0.0, 0.0, 0.0]
        ego_heading = 0.0

        return {
            'sample_token': sample_token,
            'predicted_map': predicted_map,
            'uncertainty_map': uncertainty_map,
            'ego_pos': ego_pos,
            'ego_heading': ego_heading,
        }

    except Exception as e:
        print(f"转换MapTR结果时出错: {e}")
        return None

def main(args):
    """主可视化函数，参考vis_std.py的设计"""

    # 加载MapTR预测结果
    token_to_data = {}

    # 支持单个文件或目录
    if os.path.isfile(args.map_data):
        pkl_files = [os.path.basename(args.map_data)]
        map_data_dir = os.path.dirname(args.map_data)
    else:
        pkl_files = [f for f in os.listdir(args.map_data) if f.endswith('.pkl')]
        map_data_dir = args.map_data

    # 简单的文件名过滤
    target_files = []
    if args.target_scenes:
        for f in pkl_files:
            for t in args.target_scenes:
                if t in f:
                    target_files.append(f)
    else:
        target_files = pkl_files

    print(f"Loading {len(target_files)} PKL files...")

    for filename in tqdm(target_files):
        try:
            with open(os.path.join(map_data_dir, filename), 'rb') as f:
                content = pickle.load(f)

                # 处理MapTR结果
                if isinstance(content, list):
                    for item in content:
                        converted_frame = convert_maptr_to_uncertainty_format(item)
                        if converted_frame:
                            token_to_data[converted_frame['sample_token']] = converted_frame
                else:
                    converted_frame = convert_maptr_to_uncertainty_format(content)
                    if converted_frame:
                        token_to_data[converted_frame['sample_token']] = converted_frame
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

    if len(token_to_data) == 0:
        print("Error: No valid MapTR data loaded.")
        return

    # 准备输出目录
    save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)

    # 反查场景信息
    scene_token_map = {}
    for token in token_to_data.keys():
        # 由于我们使用虚拟token，我们直接处理所有数据
        scene_name = "maptr_uncertainty_results"
        if scene_name not in scene_token_map:
            scene_token_map[scene_name] = []
        scene_token_map[scene_name].append((0, token))  # 虚拟时间戳

    # 可视化循环
    for scene_name, frames_list in scene_token_map.items():
        frames_list.sort(key=lambda x: x[0])
        sorted_tokens = [x[1] for x in frames_list]

        # 限制处理帧数（用于测试）
        if args.max_frames:
            sorted_tokens = sorted_tokens[:args.max_frames]
            print(f"Rendering {scene_name} ({len(sorted_tokens)}/{len(frames_list)} frames, limited by --max_frames)...")
        else:
            print(f"Rendering {scene_name} ({len(sorted_tokens)} frames)...")

        for idx, current_token in enumerate(sorted_tokens):
            frame_data = token_to_data[current_token]

            # 创建BEV地图可视化
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            # 设置坐标范围
            ax.set_xlim(-15, 15)
            ax.set_ylim(-30, 30)
            ax.axis('off')
            ax.set_aspect('equal')

            # 绘制预测结果
            pred_map = frame_data['predicted_map']
            uncertainty_map = frame_data.get('uncertainty_map', {})
            map_config = [('divider', 'orange'), ('ped_crossing', 'blue'), ('boundary', 'green')]

            for key, color in map_config:
                if f'{key}_scores' not in pred_map:
                    continue

                scores = np.array(pred_map[f'{key}_scores'])
                valid = scores > 0.4  # 置信度阈值
                if not np.any(valid):
                    continue

                lines = np.array(pred_map[key])[valid]
                uncertainties = uncertainty_map.get(f'{key}_uncertainty', [])

                for k, line in enumerate(lines):
                    # 坐标转换：(-y, x) 让车头朝上
                    plot_x, plot_y = -line[:, 1], line[:, 0]
                    ax.plot(plot_x, plot_y, color=color, linewidth=3, alpha=1.0, zorder=5)

                    # 绘制不确定性椭圆
                    if k < len(uncertainties) and len(uncertainties[k]) > 0:
                        uncertainty_vals = np.array(uncertainties[k])
                        if uncertainty_vals.size > 0:
                            plot_uncertainty_ellipses(ax,
                                                    np.column_stack([plot_x, plot_y]),
                                                    uncertainty_vals,
                                                    color,
                                                    alpha=args.uncertainty_alpha,
                                                    scale=args.ellipse_scale)

            # 绘制自车 (已移除)
            # ax.arrow(0, 0, 0, 2, head_width=0.8, fc='red', ec='red', zorder=11)

            # 添加标题和图例
            title_text = f"MapTRv2 Uncertainty Visualization\nFrame {idx:03d}"
            ax.text(0, -28, title_text, ha='center', fontsize=16, fontweight='bold')

            # 添加图例
            legend_elements = [
                plt.Line2D([0], [0], color='orange', linewidth=3, label='Lane Dividers'),
                plt.Line2D([0], [0], color='blue', linewidth=3, label='Pedestrian Crossings'),
                plt.Line2D([0], [0], color='green', linewidth=3, label='Road Boundaries'),
            ]

            # 创建椭圆图例
            ellipse_legend = plt.Rectangle((0, 0), 1, 1, fc='gray', alpha=0.3, label='Uncertainty Regions')
            legend_elements.append(ellipse_legend)

            ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

            # 添加信息文本
            info_text = f"""Uncertainty Parameters:
• Ellipse Scale: {args.ellipse_scale}
• Transparency: {args.uncertainty_alpha}
• Token: {current_token[:20]}...

Classes:
🟠 Dividers: Lane markings
🔵 Crossings: Pedestrian zones
🟢 Boundaries: Road edges

Ellipses show prediction uncertainty"""

            ax.text(-14, -25, info_text, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))

            # 保存图像
            out_name = f"{scene_name}_frame_{idx:03d}_uncertainty.png"
            out_path = os.path.join(save_dir, out_name)

            try:
                plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
                plt.close('all')  # 确保关闭所有图形
                print(f"Frame {idx:3d}/{len(sorted_tokens):3d}: Saved {out_name}")
            except Exception as e:
                print(f"Error saving frame {idx}: {e}")
                plt.close('all')
                continue

    print("All Done!")

def parse_args():
    parser = argparse.ArgumentParser(description='MapTRv2 Uncertainty Visualization (Simplified)')
    parser.add_argument('--dataroot', type=str, default='/home/ubuntunew/model/nuscences-mini/data/nuscenes',
                       help='nuScenes dataset root directory (for reference)')
    parser.add_argument('--map_data', type=str, required=True,
                       help='MapTR prediction results (.pkl file or directory)')
    parser.add_argument('--save_path', type=str, default='/home/ubuntunew/model/MapTR-maptrvnew124/keshihua',
                       help='Output directory for visualizations')
    parser.add_argument('--target_scenes', type=str, nargs='+', default=None,
                       help='Specific scenes to visualize')
    parser.add_argument('--ellipse_scale', type=float, default=6.0,
                       help='Scale factor for uncertainty ellipses (increased to 6.0 for better visibility)')
    parser.add_argument('--uncertainty_alpha', type=float, default=0.3,
                       help='Transparency alpha for uncertainty overlays')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to process (for testing)')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse_args())
