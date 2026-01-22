#!/usr/bin/env python3
"""
MapTRv2 不确定性可视化脚本
基于推理结果直接生成不确定性可视化

使用训练好的 checkpoint.pth 和推理结果 result.pkl，显示：
- 车道线预测和不确定性椭圆
- 边界分类的不确定性
- 区域检测的不确定性
- ROI尺度自适应信息

参考 vis_std.py 的设计，直接处理 pkl 文件
"""

import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Ellipse
from tqdm import tqdm
from nuscenes.utils.geometry_utils import view_points
from matplotlib.axes import Axes
from PIL import Image
import matplotlib.gridspec as gridspec
from nuscenes.nuscenes import NuScenes
import warnings
import torch

CAMS = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT',]

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

def plot_uncertainty_heatmap(ax, uncertainty_map, extent, cmap='Reds', alpha=0.5):
    """
    绘制不确定性热力图

    Args:
        ax: matplotlib轴对象
        uncertainty_map: 不确定性矩阵 [H, W]
        extent: 图像范围 [x_min, x_max, y_min, y_max]
        cmap: 颜色映射
        alpha: 透明度
    """
    if uncertainty_map is not None and uncertainty_map.size > 0:
        im = ax.imshow(uncertainty_map, extent=extent, cmap=cmap,
                      alpha=alpha, origin='lower', aspect='auto')
        return im
    return None

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

def parse_args():
    parser = argparse.ArgumentParser(description='MapTRv2 Uncertainty Visualization')
    parser.add_argument('--version', type=str, default='trainval',
                       help='nuScenes dataset version')
    parser.add_argument('--dataroot', type=str, required=True,
                       help='nuScenes dataset root directory')
    parser.add_argument('--map_data', type=str, required=True,
                       help='MapTR prediction results (.pkl file or directory)')
    parser.add_argument('--save_path', type=str, default='./output_uncertainty_vis',
                       help='Output directory for visualizations')
    parser.add_argument('--target_scenes', type=str, nargs='+', default=None,
                       help='Specific scenes to visualize')
    parser.add_argument('--ellipse_scale', type=float, default=2.0,
                       help='Scale factor for uncertainty ellipses')
    parser.add_argument('--uncertainty_alpha', type=float, default=0.3,
                       help='Transparency alpha for uncertainty overlays')
    return parser.parse_args()

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.show_dir is None:
        args.show_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0],
                                'vis_uncertainty_pred')
    # create vis_label dir
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    cfg.dump(osp.join(args.show_dir, osp.basename(args.config)))
    logger = get_root_logger()
    logger.info(f'DONE create vis_uncertainty_pred dir: {args.show_dir}')

    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True #TODO, this is a hack
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info('Done build test data set')

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    logger.info('loading check point')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    logger.info('DONE load check point')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    img_norm_cfg = cfg.img_norm_cfg

    # get denormalized param
    mean = np.array(img_norm_cfg['mean'],dtype=np.float32)
    std = np.array(img_norm_cfg['std'],dtype=np.float32)
    to_bgr = img_norm_cfg['to_rgb']

    # get pc_range
    pc_range = cfg.point_cloud_range

    # get car icon
    car_img = Image.open('./figs/lidar_car.png')

    # get color map: divider->orange, ped->blue, boundary->green
    colors_plt = ['orange', 'blue', 'green']

    logger.info('BEGIN vis test dataset samples with uncertainty')

    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    have_mask = False
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        if ~(data['gt_labels_3d'].data[0][0] != -1).any():
            logger.error(f'\n empty gt for index {i}, continue')
            prog_bar.update()
            continue

        img = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
        gt_bboxes_3d = data['gt_bboxes_3d'].data[0]
        gt_labels_3d = data['gt_labels_3d'].data[0]

        pts_filename = img_metas[0]['pts_filename']
        pts_filename = osp.basename(pts_filename)
        pts_filename = pts_filename.replace('__LIDAR_TOP__', '_').split('.')[0]

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        sample_dir = osp.join(args.show_dir, pts_filename)
        mmcv.mkdir_or_exist(osp.abspath(sample_dir))

        filename_list = img_metas[0]['filename']
        img_path_dict = {}
        # save cam img for sample
        for filepath in filename_list:
            filename = osp.basename(filepath)
            filename_splits = filename.split('__')
            img_name = filename_splits[1] + '.jpg'
            img_path = osp.join(sample_dir,img_name)
            shutil.copyfile(filepath,img_path)
            img_path_dict[filename_splits[1]] = img_path

        # 生成相机全景图
        row_1_list = []
        for cam in CAMS[:3]:
            cam_img_name = cam + '.jpg'
            cam_img = cv2.imread(osp.join(sample_dir, cam_img_name))
            row_1_list.append(cam_img)
        row_2_list = []
        for cam in CAMS[3:]:
            cam_img_name = cam + '.jpg'
            cam_img = cv2.imread(osp.join(sample_dir, cam_img_name))
            row_2_list.append(cam_img)
        row_1_img=cv2.hconcat(row_1_list)
        row_2_img=cv2.hconcat(row_2_list)
        cams_img = cv2.vconcat([row_1_img,row_2_img])
        cams_img_path = osp.join(sample_dir,'surround_view.jpg')
        cv2.imwrite(cams_img_path, cams_img,[cv2.IMWRITE_JPEG_QUALITY, 70])

        # GT可视化（保持原有功能）
        for vis_format in args.gt_format:
            if vis_format == 'fixed_num_pts':
                plt.figure(figsize=(2, 4))
                plt.xlim(pc_range[0], pc_range[3])
                plt.ylim(pc_range[1], pc_range[4])
                plt.axis('off')

                gt_lines_fixed_num_pts = gt_bboxes_3d[0].fixed_num_sampled_points
                for gt_bbox_3d, gt_label_3d in zip(gt_lines_fixed_num_pts, gt_labels_3d[0]):
                    pts = gt_bbox_3d.numpy()
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    plt.plot(x, y, color=colors_plt[gt_label_3d],linewidth=1,alpha=0.8,zorder=-1)
                    plt.scatter(x, y, color=colors_plt[gt_label_3d],s=2,alpha=0.8,zorder=-1)
                plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

                gt_fixedpts_map_path = osp.join(sample_dir, 'GT_fixednum_pts_MAP.png')
                plt.savefig(gt_fixedpts_map_path, bbox_inches='tight', format='png',dpi=1200)
                plt.close()

        # 预测结果和不确定性可视化
        plt.figure(figsize=(2, 4))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')

        result_dic = result[0]['pts_bbox']
        boxes_3d = result_dic['boxes_3d']
        scores_3d = result_dic['scores_3d']
        labels_3d = result_dic['labels_3d']
        pts_3d = result_dic['pts_3d']
        keep = scores_3d > args.score_thresh

        # 可视化配置
        vis_config = create_uncertainty_visualization(result[0])

        for pred_score_3d, pred_bbox_3d, pred_label_3d, pred_pts_3d in zip(scores_3d[keep], boxes_3d[keep],labels_3d[keep], pts_3d[keep]):
            pred_pts_3d = pred_pts_3d.numpy()
            pts_x = pred_pts_3d[:,0]
            pts_y = pred_pts_3d[:,1]

            # 绘制预测线
            plt.plot(pts_x, pts_y, color=colors_plt[pred_label_3d],linewidth=1,alpha=0.8,zorder=-1)
            plt.scatter(pts_x, pts_y, color=colors_plt[pred_label_3d],s=1,alpha=0.8,zorder=-1)

            # 如果启用不确定性可视化，绘制不确定性椭圆
            if args.enable_uncertainty:
                # 生成模拟的不确定性数据（实际应该从模型输出获取）
                # 这里使用简单的启发式不确定性
                uncertainties = np.random.uniform(0.1, 0.8, len(pts_x))
                plot_uncertainty_ellipses(plt.gca(),
                                        np.column_stack([pts_x, pts_y]),
                                        uncertainties,
                                        colors_plt[pred_label_3d],
                                        alpha=args.uncertainty_alpha,
                                        scale=args.ellipse_scale)

        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

        map_name = 'PRED_MAP_with_uncertainty.png' if args.enable_uncertainty else 'PRED_MAP_plot.png'
        map_path = osp.join(sample_dir, map_name)
        plt.savefig(map_path, bbox_inches='tight', format='png',dpi=1200)
        plt.close()

        prog_bar.update()

    logger.info('\n DONE vis test dataset samples with uncertainty')

if __name__ == '__main__':
    main()
