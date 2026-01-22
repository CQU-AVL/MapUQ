# MMCV环境问题解决方案

## 问题描述

运行MapTRv2可视化脚本时遇到以下错误：
```
ModuleNotFoundError: No module named 'mmcv._ext'
```

## 根本原因

MMCV的C++扩展模块没有正确编译或安装。

## 解决方案

### 方案1: 重新安装MMCV (推荐)

```bash
# 1. 卸载当前的MMCV
pip uninstall mmcv mmcv-full

# 2. 根据CUDA版本安装对应的MMCV
# CUDA 11.1
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

# CUDA 11.3
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.8.0/index.html

# CUDA 11.6
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.8.0/index.html

# 其他版本请参考: https://mmcv.readthedocs.io/en/latest/get_started/installation.html
```

### 方案2: 使用源码编译安装

```bash
# 从源码安装MMCV
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
pip install -r requirements.txt
pip install -v -e .
```

### 方案3: 使用预编译的wheel文件

```bash
# 下载对应版本的wheel文件
# https://download.openmmlab.com/mmcv/dist/index.html
pip install mmcv_full-1.4.0-cp38-cp38-linux_x86_64.whl
```

## 验证安装

```bash
python -c "import mmcv; print('MMCV version:', mmcv.__version__)"
python -c "import mmcv.ops; print('MMCV ops available')"
```

## 临时解决方案

如果MMCV问题无法解决，可以使用我们提供的**简化版本**：

```bash
# 使用简化的语义分割演示
python test_semantic_overlay_demo.py

# 这将生成语义分割叠加效果的演示图片
# 查看: semantic_overlay_demo_result.png
```

## 环境要求

- **Python**: 3.8+
- **PyTorch**: 1.8.0+
- **CUDA**: 与PyTorch版本匹配
- **GCC**: 7.3+

## 常见问题

### Q: 安装后仍然报错
A: 尝试清除pip缓存：
```bash
pip cache purge
pip install --no-cache-dir mmcv-full
```

### Q: CUDA版本不匹配
A: 检查CUDA版本：
```bash
nvidia-smi
nvcc --version
```

### Q: 权限问题
A: 使用用户权限安装：
```bash
pip install --user mmcv-full
```

## 备用方案

如果MMCV问题持续存在，可以考虑：

1. **使用Google Colab**: 预装了MMCV
2. **使用Docker**: 使用官方的MMDetection3D Docker镜像
3. **使用简化版本**: 我们的`test_semantic_overlay_demo.py`脚本

---

**注意**: 解决MMCV问题后，MapTRv2的可视化脚本就能正常运行了！


