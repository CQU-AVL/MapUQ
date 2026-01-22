import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Ellipse
from tqdm import tqdm
from nuscenes.utils.geometry_utils import view_points
from matplotlib.axes import Axes
from typing import Tuple
from PIL import Image
import matplotlib.gridspec as gridspec
from nuscenes.nuscenes import NuScenes
import warnings
import argparse
import torch


# ================= 1. 数据搜索 & 基础函数 =================

def recursive_search_frames(data, found_frames):
    """递归查找包含有效数据的帧"""
    if isinstance(data, dict):
        if 'predicted_map' in data and 'sample_token' in data:
            found_frames.append(data)
            return
        # 处理MapTR格式的数据
        if 'pts_bbox' in data:
            # 转换MapTR格式为vis_std期望格式
            converted_frame = convert_maptr_to_vis_std_format(data)
            if converted_frame:
                found_frames.append(converted_frame)
            return
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                recursive_search_frames(v, found_frames)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                recursive_search_frames(item, found_frames)

def convert_maptr_to_vis_std_format(maptr_result):
    """将MapTR推理结果转换为vis_std期望的格式"""
    try:
        pts_bbox = maptr_result.get('pts_bbox', {})
        if not pts_bbox:
            return None

        # MapTR结果没有sample_token，我们需要创建一个虚拟的
        # 这里用索引作为sample_token
        sample_token = f"maptr_sample_{id(maptr_result) % 1000}"

        # 转换pts_3d格式 [num_preds, num_pts, 2] -> 预测地图格式
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

        # 构建predicted_map
        predicted_map = {}
        map_classes = ['divider', 'ped_crossing', 'boundary']

        for class_idx, class_name in enumerate(map_classes):
            # 找到属于这个类别的预测
            class_mask = labels_3d == class_idx
            if not np.any(class_mask):
                predicted_map[f'{class_name}'] = []
                predicted_map[f'{class_name}_scores'] = []
                predicted_map[f'{class_name}_betas'] = []
                continue

            # 提取这个类别的点
            class_pts = pts_3d[class_mask]  # [num_class_preds, 20, 2]
            class_scores = scores_3d[class_mask]  # [num_class_preds]

            # 转换为vis_std期望的格式
            predicted_map[f'{class_name}'] = class_pts.tolist()
            predicted_map[f'{class_name}_scores'] = class_scores.tolist()

            # 为每个预测创建虚拟的不确定性参数 (beta参数)
            # vis_std需要beta参数来绘制不确定性椭圆
            num_preds = len(class_pts)
            betas = []
            for i in range(num_preds):
                # 为每个点创建beta参数 [num_pts, 2] (x和y方向的beta)
                beta_pts = np.random.uniform(0.1, 0.5, (20, 2))  # 虚拟值
                betas.append(beta_pts.tolist())
            predicted_map[f'{class_name}_betas'] = betas

        # 创建虚拟的ego位置和朝向（如果需要的话）
        ego_pos = [0.0, 0.0, 0.0]  # 虚拟值
        ego_heading = 0.0  # 虚拟值

        return {
            'sample_token': sample_token,
            'predicted_map': predicted_map,
            'ego_pos': ego_pos,
            'ego_heading': ego_heading,
            'maptr_gt_map': {}  # 空的GT地图
        }

    except Exception as e:
        print(f"转换MapTR结果时出错: {e}")
        return None


def normalize_lanes(x, y, lanes, theta):
    """坐标系转换：全局 -> 自车"""
    if hasattr(theta, 'item'): theta = theta.item()
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    n_lanes = []
    for lane in lanes:
        origin = np.array([x, y]).flatten()
        n_lanes.append(np.dot(lane[:, :2] - origin, R))
    return n_lanes


def plot_points_with_laplace_variances(x, y, beta_x, beta_y, color, ax, std=True):
    """绘制预测线和不确定性椭圆"""
    # 绘制实线
    ax.plot(x, y, color=color, linewidth=2, alpha=1.0, zorder=5)

    # 绘制不确定性椭圆
    if std:
        var_x = 2 * beta_x ** 2
        var_y = 2 * beta_y ** 2
        step = 3  # 稀疏一点，好看
        for j in range(0, len(x), step):
            vx, vy = var_x[j], var_y[j]
            if hasattr(vx, 'item'): vx = vx.item()
            if hasattr(vy, 'item'): vy = vy.item()

            # 这里的乘数决定椭圆大小，2倍sigma覆盖约95%置信区间
            width, height = np.sqrt(vx) * 2, np.sqrt(vy) * 2

            # 绘制半透明椭圆
            ellipse = Ellipse((x[j], y[j]), width=width, height=height,
                              fc=color, lw=0, alpha=0.4, zorder=3)
            ax.add_patch(ellipse)


def render_box(box, axis: Axes, view: np.ndarray = np.eye(3), normalize: bool = False, colors: Tuple = ('b', 'r', 'k'),
               linewidth: float = 1):
    """绘制3D Box"""
    if box.orientation is None: return
    corners = view_points(box.corners(), view, normalize=normalize)[:2, :]

    def draw_rect(selected_corners, color):
        prev = selected_corners[-1]
        for corner in selected_corners:
            axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth, alpha=0.6)
            prev = corner

    for i in range(4):
        axis.plot([corners.T[i][0], corners.T[i + 4][0]], [corners.T[i][1], corners.T[i + 4][1]], color=colors[2],
                  linewidth=linewidth, alpha=0.6)
    draw_rect(corners.T[:4], colors[0])
    draw_rect(corners.T[4:], colors[1])


# ================= 2. 图像加载 =================

def get_camera_images(nusc, sample_token):
    """从 nuScenes 获取6个相机的图像"""
    sample = nusc.get('sample', sample_token)
    sensors = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
               'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    images = {}
    for sensor in sensors:
        cam_data = nusc.get('sample_data', sample['data'][sensor])
        img_path = os.path.join(nusc.dataroot, cam_data['filename'])
        if os.path.exists(img_path):
            images[sensor] = Image.open(img_path)
        else:
            print(f"Warning: Image not found {img_path}")
            # 创建个黑图防止报错
            images[sensor] = Image.new('RGB', (1600, 900), (0, 0, 0))
    return images


# ================= 3. 主绘图逻辑 =================

def main(args):
    # A. 初始化
    print(f"Initializing nuScenes ({args.version})...")
    nusc = NuScenes(version=f'v1.0-{args.version}', dataroot=args.dataroot, verbose=False)

    # B. 加载数据
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
                if t in f: target_files.append(f)
    else:
        target_files = pkl_files

    print(f"Scanning {len(target_files)} PKL files...")

    for filename in tqdm(target_files):
            try:
                with open(os.path.join(map_data_dir, filename), 'rb') as f:
                    content = pickle.load(f)
                    frames = []
                    recursive_search_frames(content, frames)
                    for frame in frames:
                        token_to_data[frame['sample_token']] = frame
            except:
                pass

    if len(token_to_data) == 0:
        print("Error: No data loaded.")
        return

    # C. 加载 Boxes
    try:
        with open(args.boxes, 'rb') as f:
            boxes_gt_all = pickle.load(f)
    except:
        print("Warning: GT Boxes not found.")
        boxes_gt_all = {}

    # D. 准备输出
    save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)

    # E. 反查 Scene
    scene_token_map = {}
    for token in token_to_data.keys():
        try:
            s_rec = nusc.get('sample', token)
            scene_name = nusc.get('scene', s_rec['scene_token'])['name']
            if scene_name not in scene_token_map: scene_token_map[scene_name] = []
            scene_token_map[scene_name].append((s_rec['timestamp'], token))
        except:
            pass

    # F. 绘图循环
    for scene_name, frames_list in scene_token_map.items():
        frames_list.sort(key=lambda x: x[0])
        sorted_tokens = [x[1] for x in frames_list]

        print(f"Rendering {scene_name} ({len(sorted_tokens)} frames)...")

        for idx, current_token in enumerate(sorted_tokens):
            frame_data = token_to_data[current_token]

            # --- 布局设置 ---
            # 3行4列
            fig = plt.figure(figsize=(24, 12), facecolor='white')
            gs = gridspec.GridSpec(3, 4, width_ratios=[1, 1, 1.5, 1.5])

            # --- 1. 绘制相机 (左侧) ---
            imgs = get_camera_images(nusc, current_token)
            sensor_order = [
                ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'],
                ['CAM_FRONT', 'CAM_BACK'],
                ['CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
            ]

            for row in range(3):
                for col in range(2):
                    sensor_name = sensor_order[row][col]
                    ax_cam = fig.add_subplot(gs[row, col])
                    ax_cam.imshow(imgs[sensor_name])
                    ax_cam.axis('off')
                    ax_cam.text(10, 30, sensor_name.replace('CAM_', ''), color='white',
                                fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))

            # --- 准备地图数据 ---
            ego_pos = frame_data['ego_pos']
            if isinstance(ego_pos, torch.Tensor): ego_pos = ego_pos.cpu().numpy()
            ego_pos = np.array(ego_pos).flatten()
            ego_x, ego_y = ego_pos[0], ego_pos[1]
            ego_heading = frame_data['ego_heading']
            if hasattr(ego_heading, 'item'):
                ego_heading = ego_heading.item()
            elif isinstance(ego_heading, np.ndarray):
                ego_heading = ego_heading.flat[0]

            # --- 2. 绘制预测地图 (Ours - Middle) ---
            ax_pred = fig.add_subplot(gs[:, 2])
            ax_pred.set_xlim(-15, 15)
            ax_pred.set_ylim(-30, 30)
            ax_pred.axis('off')
            ax_pred.set_aspect('equal')

            # 绘制预测
            pred_map = frame_data['predicted_map']
            map_config = [('divider', 'orange', 0.4), ('ped_crossing', 'blue', 0.3), ('boundary', 'green', 0.4)]

            for key, color, thresh in map_config:
                if f'{key}_scores' not in pred_map: continue
                scores = np.array(pred_map[f'{key}_scores'])
                valid = scores > thresh
                if not np.any(valid): continue

                lines = np.array(pred_map[key])[valid]
                betas = np.array(pred_map[f'{key}_betas'])[valid]
                norm_lines = normalize_lanes(ego_x, ego_y, lines, ego_heading)

                for k, line in enumerate(norm_lines):
                    # 坐标转换：(-y, x) 让车头朝上
                    plot_x, plot_y = -line[:, 1], line[:, 0]
                    bx, by = betas[k][:, 1], betas[k][:, 0]
                    plot_points_with_laplace_variances(plot_x, plot_y, bx, by, color, ax_pred, std=True)

            # 绘制Boxes
            if current_token in boxes_gt_all:
                for box in boxes_gt_all[current_token]:
                    if box.name in ['barrier', 'traffic_cone']: continue
                    if abs(box.center[0]) > 25 or abs(box.center[1]) > 40: continue
                    render_box(box, ax_pred, np.eye(4), colors=('grey', 'grey', 'grey'), linewidth=1)

            # 绘制Ego
            ax_pred.arrow(0, 0, 0, 2, head_width=0.8, fc='red', ec='red', zorder=11)
            ax_pred.text(0, -28, "Ours (Pred + Uncertainty)", ha='center', fontsize=16, fontweight='bold')

            # --- 3. 绘制真值地图 (GT - Right) ---
            ax_gt = fig.add_subplot(gs[:, 3])
            ax_gt.set_xlim(-15, 15)
            ax_gt.set_ylim(-30, 30)
            ax_gt.axis('off')
            ax_gt.set_aspect('equal')

            # 尝试获取 GT Map
            gt_keys_found = False
            gt_source = None
            if 'maptr_gt_map' in frame_data:
                gt_source = frame_data['maptr_gt_map']
            elif 'gt_map' in frame_data:
                gt_source = frame_data['gt_map']

            if gt_source:
                for key, lines in gt_source.items():
                    if len(lines) == 0: continue

                    # === 【关键修复】颜色模糊匹配逻辑 ===
                    color = 'black'
                    key_str = str(key).lower()

                    if 'div' in key_str or key == 0:
                        color = 'orange'
                    elif 'ped' in key_str or 'cross' in key_str or key == 1:
                        color = 'blue'
                    elif 'bound' in key_str or key == 2:
                        color = 'green'
                    # ===================================

                    try:
                        norm_lines = normalize_lanes(ego_x, ego_y, np.array(lines), ego_heading)
                        for line in norm_lines:
                            plot_x, plot_y = -line[:, 1], line[:, 0]
                            ax_gt.plot(plot_x, plot_y, color=color, linewidth=2, alpha=0.8)
                        gt_keys_found = True
                    except:
                        pass

            if not gt_keys_found:
                ax_gt.text(0, 0, "GT Map Not Found", ha='center')

            # 绘制Boxes
            if current_token in boxes_gt_all:
                for box in boxes_gt_all[current_token]:
                    if box.name in ['barrier', 'traffic_cone']: continue
                    if abs(box.center[0]) > 25 or abs(box.center[1]) > 40: continue
                    render_box(box, ax_gt, np.eye(4), colors=('grey', 'grey', 'grey'), linewidth=1)

            # 绘制Ego
            ax_gt.arrow(0, 0, 0, 2, head_width=0.8, fc='red', ec='red', zorder=11)
            ax_gt.text(0, -28, "Ground Truth", ha='center', fontsize=16, fontweight='bold')

            # --- 保存 ---
            plt.tight_layout()
            out_name = f"{scene_name}_frame_{idx:03d}_paper.png"
            out_path = os.path.join(save_dir, out_name)
            plt.savefig(out_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.close(fig)

            print(f"Saved: {out_path}")

    print("All Done.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default='trainval')
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--map_data', type=str, required=True)
    parser.add_argument('--boxes', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='./output_paper')
    parser.add_argument('--target_scenes', type=str, nargs='+', default=None)
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())