# 垃圾分类项目 - 最终结构

## 核心文件

### 主要代码
- `train.py` - 模型训练脚本
- `main.py` - 主程序入口
- `model_utils.py` - 模型工具函数
- `realtime_classifier.py` - 实时分类器
- `web_realtime.py` - Web界面实时分类
- `generate_confusion_matrix.py` - 生成混淆矩阵和评估指标

### 配置文件
- `requirements.txt` - Python依赖包列表
- `README_COMPLETE.md` - 完整项目文档

### 结果文件
- `confusion_matrix.png` - 混淆矩阵图
- `detailed_metrics.png` - 详细评估指标图
- `model_results.npy` - 模型预测结果数据

## 目录结构

### models/
训练好的模型文件：
- `trash_classifier_final.pth` - 最终训练模型
- `trash_classifier_epoch*.pth` - 各训练阶段模型

### templates/
Web界面模板文件

### dataset_cache/
数据集缓存目录

### trash_classifier_env/
Python虚拟环境

## 使用方法

1. **训练模型**: `python train.py`
2. **生成评估图表**: `python generate_confusion_matrix.py`
3. **实时分类**: `python realtime_classifier.py`
4. **Web界面**: `python web_realtime.py`

详细说明请参考 `README_COMPLETE.md`
