#!/usr/bin/env python3
"""
MapTRv2 语义分割可视化脚本
在相机图像上叠加语义分割mask，提供丰富的可视化结果

基于 SemVecNet 的语义分割模块，为 MapTRv2 的相机图像添加语义分割可视化
"""

import argparse
import mmcv
import os
import shutil
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet3d.utils import collect_env, get_root_logger
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib.patches import Rectangle
import cv2
import sys

# 添加语义分割模块路径
semantic_src_path = osp.abspath(osp.join(osp.dirname(__file__), '../semantic_mapping'))
sys.path.insert(0, semantic_src_path)

# 尝试导入真实的语义分割模块
HRNetSemanticSegmentationTensorRT = None
get_custom_hrnet_args = None
apply_color_map_real = None
get_labels_real = None

try:
    from hrnet.hrnet_semantic_segmentation_tensorrt import HRNetSemanticSegmentationTensorRT, get_custom_hrnet_args
    from utils.mapillary_visualization import apply_color_map as apply_color_map_real, get_labels as get_labels_real
    REAL_SEMANTIC_AVAILABLE = True
    print("✅ 真实语义分割模块加载成功")
except ImportError as e:
    REAL_SEMANTIC_AVAILABLE = False
    print(f"⚠️ 真实语义分割模块不可用: {e}")
    print("将使用简易语义分割进行演示")

# 简易语义分割始终可用
SEMANTIC_AVAILABLE = True

CAMS = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT',]

def get_semantic_segmentation_overlay(image, segmentation_model, seg_color_ref, alpha=0.3, force_simple=False):
    """
    为相机图像添加语义分割可视化叠加

    Args:
        image (np.ndarray): 原始相机图像 (RGB格式)
        segmentation_model: 语义分割模型 (如果为None则使用简易分割)
        seg_color_ref: 语义类别颜色参考
        alpha (float): 叠加透明度 (0-1)

    Returns:
        tuple: (叠加图像, 语义分割mask)
    """
    try:
        # 转换为RGB格式（如果需要）
        if image.shape[2] == 3 and image.dtype == np.uint8:
            # 已经是RGB格式，直接使用
            pass
        else:
            # 转换格式
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if segmentation_model is not None and REAL_SEMANTIC_AVAILABLE:
            # 使用真实的语义分割模型
            try:
                # 1. 预处理图像
                processed_img = segmentation_model.preprocess(image)

                # 2. 进行语义分割
                seg_result = segmentation_model.segmentation(processed_img)

                # 3. 转换为彩色mask
                # seg_result 需要reshape为正确格式
                if len(seg_result.shape) == 3:
                    seg_result = seg_result.squeeze()

                colored_mask = apply_color_map_real(seg_result.astype(np.uint8), seg_color_ref)
            except Exception as e:
                print(f"真实语义分割模型失败，使用简易分割: {e}")
                colored_mask = get_simple_semantic_segmentation(image, seg_color_ref)
        else:
            # 使用简易语义分割（基于颜色和边缘检测）
            print("使用简易语义分割（演示用）")
            colored_mask = get_simple_semantic_segmentation(image, seg_color_ref)

        # 确保colored_mask是正确的尺寸
        if colored_mask.shape[:2] != image.shape[:2]:
            colored_mask = cv2.resize(colored_mask, (image.shape[1], image.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

        # 4. 创建半透明叠加
        overlay = cv2.addWeighted(image.astype(np.float32), 1-alpha,
                                colored_mask.astype(np.float32), alpha, 0)
        overlay = overlay.astype(np.uint8)

        return overlay, colored_mask

    except Exception as e:
        print(f"语义分割处理失败: {e}")
        return image, None

def get_simple_semantic_segmentation(image, seg_color_ref):
    """
    简易语义分割实现（用于演示）

    基于简单的图像处理技术创建伪语义分割结果：
    - 道路：基于亮度和纹理
    - 车道线：基于边缘检测
    - 建筑物：基于垂直边缘
    - 植被：基于绿色通道

    Args:
        image: 输入图像
        seg_color_ref: 颜色参考

    Returns:
        彩色语义分割mask
    """
    height, width = image.shape[:2]

    # 创建语义分割结果
    seg_mask = np.zeros((height, width), dtype=np.uint8)

    # 转换为HSV用于颜色分析
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 1. 检测道路区域（通常在图像下半部分，亮度较高）
    road_mask = np.zeros((height, width), dtype=np.uint8)
    road_mask[height//2:, :] = 1  # 下半部分标记为道路

    # 2. 检测车道线（基于边缘检测）
    edges = cv2.Canny(gray, 50, 150)
    # 车道线通常是水平的或接近水平的线条
    kernel = np.ones((1, 5), np.uint8)
    lane_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    lane_mask = (lane_mask > 0).astype(np.uint8) * 2  # 类别2：车道线

    # 3. 检测建筑物（基于垂直边缘和几何形状）
    kernel_vert = np.ones((10, 1), np.uint8)
    vert_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_vert)
    building_mask = (vert_edges > 0).astype(np.uint8) * 3  # 类别3：建筑物

    # 4. 检测植被（基于绿色通道）
    green_channel = image[:, :, 1]
    green_mask = (green_channel > np.mean(green_channel) + 20).astype(np.uint8) * 4  # 类别4：植被

    # 合并所有mask（优先级：植被 > 建筑物 > 车道线 > 道路）
    seg_mask = road_mask
    seg_mask = np.where(lane_mask > 0, lane_mask, seg_mask)
    seg_mask = np.where(building_mask > 0, building_mask, seg_mask)
    seg_mask = np.where(green_mask > 0, green_mask, seg_mask)

    # 应用颜色映射
    colored_mask = apply_color_map(seg_mask, seg_color_ref)

    return colored_mask

def parse_args():
    parser = argparse.ArgumentParser(description='MapTRv2 with Semantic Segmentation Visualization')

    # 基础参数 - 设置默认路径
    parser.add_argument('config',
                       nargs='?',
                       default='projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py',
                       help='test config file path')
    parser.add_argument('checkpoint',
                       nargs='?',
                       default='/home/ubuntunew/model/MapTR-maptrvnew/maptrv2_nusc_r50_24e.pth',
                       help='checkpoint file')
    parser.add_argument('--score-thresh', default=0.4, type=float, help='samples to visualize')

    # 输出目录参数
    parser.add_argument('--show-dir', help='directory where visualizations will be saved')
    parser.add_argument('--show-cam', action='store_true', help='show camera pic')

    # 语义分割参数
    parser.add_argument('--enable-semantic', action='store_true',
                       help='enable semantic segmentation visualization')
    parser.add_argument('--semantic-engine', type=str,
                       default='./tools/semantic_mapping/hrnet/assets/seg_weights/hrnet-avl-map.engine',
                       help='path to HRNet TensorRT engine')
    parser.add_argument('--semantic-config', type=str,
                       default='./tools/semantic_mapping/config/config_65.json',
                       help='path to semantic segmentation config')
    parser.add_argument('--semantic-alpha', type=float, default=0.3,
                       help='transparency alpha for semantic overlay (0-1)')
    parser.add_argument('--save-separate-masks', action='store_true',
                       help='save separate semantic segmentation masks')
    parser.add_argument('--force-simple-semantic', action='store_true',
                       help='force using simple semantic segmentation even if real model is available')

    # GT可视化参数
    parser.add_argument(
        '--gt-format',
        type=str,
        nargs='+',
        default=['fixed_num_pts'],
        help='vis format, default should be "points",'
        'support ["se_pts","bbox","fixed_num_pts","polyline_pts"]')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.enable_semantic and not SEMANTIC_AVAILABLE:
        print("警告: 真实语义分割模块不可用，将使用简易语义分割进行演示")
        print("如需真实语义分割，请确保SemVecNet的语义分割依赖正确安装")

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
                                'vis_semantic_pred')
    # create vis_label dir
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    cfg.dump(osp.join(args.show_dir, osp.basename(args.config)))

    # 复制语义分割配置文件（如果启用语义分割）
    if args.enable_semantic:
        semantic_config_dir = osp.join(osp.dirname(__file__), 'semantic_mapping', 'config')
        semantic_config_src = osp.join(osp.dirname(__file__), '../../../SemVecNet/semantic_mapping/config/config_65.json')

        mmcv.mkdir_or_exist(semantic_config_dir)

        if osp.exists(semantic_config_src):
            shutil.copy2(semantic_config_src, osp.join(semantic_config_dir, 'config_65.json'))
            print(f"Copied semantic config to: {semantic_config_dir}")
        else:
            print(f"Warning: Semantic config not found at {semantic_config_src}")
            print("Using default color reference for demonstration")
    logger = get_root_logger()
    logger.info(f'DONE create vis_semantic_pred dir: {args.show_dir}')

    # 初始化语义分割模型
    segmentation_model = None
    seg_color_ref = None
    if args.enable_semantic:
        logger.info('Initializing semantic segmentation model...')
        try:
            # 首先尝试使用真实的语义分割
            if REAL_SEMANTIC_AVAILABLE and get_labels_real is not None:
                try:
                    seg_color_ref = get_labels_real(args.semantic_config)
                    logger.info('✅ 成功加载真实语义分割颜色参考')

                    # 尝试创建HRNet语义分割模型
                    if os.path.exists(args.semantic_engine):
                        seg_args = type('Args', (), {})()
                        seg_args.engine_file_path = args.semantic_engine
                        seg_args.dummy_image_path = './figs/lidar_car.png'

                        segmentation_model = HRNetSemanticSegmentationTensorRT(seg_args)
                        logger.info('🎯 成功加载HRNet TensorRT模型 - 使用真实语义分割！')
                    else:
                        logger.warning(f'⚠️ HRNet引擎文件不存在: {args.semantic_engine}')
                        logger.warning('将使用简易语义分割进行演示')
                        seg_color_ref = create_default_color_ref()
                except Exception as e:
                    logger.error(f'❌ 真实语义分割初始化失败: {e}')
                    logger.warning('降级使用简易语义分割')
                    seg_color_ref = create_default_color_ref()
            else:
                logger.info('ℹ️ 真实语义分割模块不可用，使用简易语义分割')
                seg_color_ref = create_default_color_ref()

        except Exception as e:
            logger.error(f'❌ 语义分割初始化完全失败: {e}')
            logger.warning('使用默认颜色参考')
            seg_color_ref = create_default_color_ref()

def create_default_color_ref():
    """创建默认的颜色参考"""
    return [
        {'color': [128, 64, 128], 'readable': 'road'},      # 道路
        {'color': [140, 140, 200], 'readable': 'crosswalk'}, # 人行横道
        {'color': [255, 255, 255], 'readable': 'lane'},     # 车道线
        {'color': [70, 130, 180], 'readable': 'building'},  # 建筑物
        {'color': [107, 142, 35], 'readable': 'vegetation'} # 植被
    ]

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

    # get color map: divider->r, ped->b, boundary->g
    colors_plt = ['orange', 'b', 'r', 'g']

    logger.info('BEGIN vis test dataset samples with semantic segmentation')

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

        # 生成语义分割可视化
        semantic_overlays = {}
        if args.enable_semantic and segmentation_model is not None:
            logger.info(f'Processing semantic segmentation for sample {pts_filename}')
            for cam_name, img_path in img_path_dict.items():
                try:
                    # 读取原始图像
                    cam_img = cv2.imread(img_path)
                    if cam_img is None:
                        logger.warning(f'Could not read image: {img_path}')
                        continue

                    # 添加语义分割叠加
                    overlay_img, seg_mask = get_semantic_segmentation_overlay(
                        cam_img, segmentation_model, seg_color_ref, args.semantic_alpha, args.force_simple_semantic)

                    # 保存叠加结果
                    overlay_path = img_path.replace('.jpg', '_semantic_overlay.jpg')
                    cv2.imwrite(overlay_path, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

                    # 保存单独的语义分割mask（如果启用）
                    if args.save_separate_masks and seg_mask is not None:
                        mask_path = img_path.replace('.jpg', '_semantic_mask.jpg')
                        cv2.imwrite(mask_path, cv2.cvtColor(seg_mask, cv2.COLOR_RGB2BGR))

                    semantic_overlays[cam_name] = overlay_path
                    logger.info(f'Saved semantic overlay: {overlay_path}')

                except Exception as e:
                    logger.error(f'Failed to process semantic segmentation for {cam_name}: {e}')
                    continue

        # 生成全景图（包含语义分割叠加）
        row_1_list = []
        for cam in CAMS[:3]:
            cam_img_name = cam + '_semantic_overlay.jpg' if args.enable_semantic else cam + '.jpg'
            cam_img_path = osp.join(sample_dir, cam_img_name)

            if osp.exists(cam_img_path):
                cam_img = cv2.imread(cam_img_path)
            else:
                # fallback to original image
                cam_img = cv2.imread(osp.join(sample_dir, cam + '.jpg'))

            if cam_img is not None:
                row_1_list.append(cam_img)

        row_2_list = []
        for cam in CAMS[3:]:
            cam_img_name = cam + '_semantic_overlay.jpg' if args.enable_semantic else cam + '.jpg'
            cam_img_path = osp.join(sample_dir, cam_img_name)

            if osp.exists(cam_img_path):
                cam_img = cv2.imread(cam_img_path)
            else:
                # fallback to original image
                cam_img = cv2.imread(osp.join(sample_dir, cam + '.jpg'))

            if cam_img is not None:
                row_2_list.append(cam_img)

        if row_1_list and row_2_list:
            row_1_img = cv2.hconcat(row_1_list)
            row_2_img = cv2.hconcat(row_2_list)
            cams_img = cv2.vconcat([row_1_img, row_2_img])
            cams_img_path = osp.join(sample_dir, 'surround_semantic_view.jpg')
            cv2.imwrite(cams_img_path, cams_img, [cv2.IMWRITE_JPEG_QUALITY, 70])

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

        # 预测结果可视化（保持原有功能）
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

        for pred_score_3d, pred_bbox_3d, pred_label_3d, pred_pts_3d in zip(scores_3d[keep], boxes_3d[keep],labels_3d[keep], pts_3d[keep]):
            pred_pts_3d = pred_pts_3d.numpy()
            pts_x = pred_pts_3d[:,0]
            pts_y = pred_pts_3d[:,1]
            plt.plot(pts_x, pts_y, color=colors_plt[pred_label_3d],linewidth=1,alpha=0.8,zorder=-1)
            plt.scatter(pts_x, pts_y, color=colors_plt[pred_label_3d],s=1,alpha=0.8,zorder=-1)

        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

        map_path = osp.join(sample_dir, 'PRED_MAP_plot.png')
        plt.savefig(map_path, bbox_inches='tight', format='png',dpi=1200)
        plt.close()

        prog_bar.update()

    logger.info('\n DONE vis test dataset samples with semantic segmentation')

if __name__ == '__main__':
    main()
