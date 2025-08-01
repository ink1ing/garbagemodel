# 🤖 垃圾分类AI识别系统

基于深度学习的智能垃圾分类系统，使用PyTorch和ResNet50架构，支持实时摄像头识别和Web界面操作。

## 📋 项目概述

本项目实现了一个完整的垃圾分类AI系统，能够识别6种不同类型的垃圾：
- 📦 纸板 (cardboard)
- 🍶 玻璃 (glass) 
- 🔧 金属 (metal)
- 📄 纸张 (paper)
- 🥤 塑料 (plastic)
- 🗑️ 其他垃圾 (trash)

## 🚀 核心功能

- ✅ **深度学习模型训练**: 基于ResNet50的迁移学习
- ✅ **实时摄像头识别**: 1080p 60fps高质量识别
- ✅ **Web界面操作**: 友好的浏览器界面
- ✅ **性能评估**: 混淆矩阵和详细指标分析
- ✅ **训练可视化**: 完整的训练过程图表

## 🛠️ 环境要求

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.0+
- Apple Silicon Mac (推荐，支持MPS加速)

## 📦 安装依赖

```bash
# 激活虚拟环境
source trash_classifier_env/bin/activate

# 安装依赖包
pip install -r requirements.txt
```

## 🎯 核心命令

### 1. 开始模型训练

```bash
# 激活环境并开始训练（15轮）
source trash_classifier_env/bin/activate && python train.py --epochs 15
```

**训练参数说明:**
- 默认训练15轮
- 批次大小: 32 (Apple Silicon优化)
- 学习率: 0.001 (余弦退火调度)
- 模型: ResNet50 + 迁移学习
- 数据增强: 随机翻转、旋转、颜色变换

### 2. 训练结束绘图

```bash
# 生成混淆矩阵和评估指标
source trash_classifier_env/bin/activate && python generate_confusion_matrix.py

# 生成训练过程分析图表
source trash_classifier_env/bin/activate && python generate_training_charts.py
```

**生成的图表:**
- `results/charts/confusion_matrix.png` - 混淆矩阵
- `results/charts/detailed_metrics.png` - 精确度、召回率、F1分数
- `results/charts/training_progress_analysis.png` - 训练过程分析

### 3. 启动网页服务

```bash
# 启动Flask Web服务器
source trash_classifier_env/bin/activate && python web_realtime.py
```

**访问地址:**
- 主界面: http://localhost:5001
- 简化测试: http://localhost:5001/test

## 📊 项目结构

```
canva1/
├── README.md                           # 项目说明文档
├── requirements.txt                    # Python依赖包
├── train.py                           # 模型训练脚本 ⭐
├── web_realtime.py                    # Web服务启动 ⭐
├── realtime_classifier.py             # 实时分类器
├── generate_confusion_matrix.py       # 评估图表生成 ⭐
├── generate_training_charts.py        # 训练图表生成 ⭐
├── app.py                             # 模型和类别定义
├── model_utils.py                     # 模型工具函数
├── main.py                            # 命令行入口
├── models/                            # 训练好的模型文件
│   └── trash_classifier_final.pth    # 最终模型
├── results/charts/                    # 结果图表目录
│   ├── confusion_matrix.png          # 混淆矩阵
│   ├── detailed_metrics.png          # 详细指标
│   └── training_progress_analysis.png # 训练过程
├── templates/                         # Web界面模板
├── dataset_cache/                     # 数据集缓存
└── trash_classifier_env/              # Python虚拟环境
```

## 🎮 使用方法

### 方法1: Web界面使用

1. 启动服务: `python web_realtime.py`
2. 打开浏览器访问: http://localhost:5001
3. 点击"开始识别"按钮
4. 将垃圾物品放在摄像头前即可识别

### 方法2: 命令行使用

```bash
# 实时摄像头识别
python realtime_classifier.py

# 训练新模型
python train.py --epochs 20 --batch-size 32

# 评估模型性能
python generate_confusion_matrix.py
```

## 📈 性能指标

- **总体准确率**: 96%+
- **处理速度**: 1080p @ 60fps
- **模型大小**: ~100MB
- **推理延迟**: <50ms (Apple Silicon)

## 🔧 技术特性

- **Apple Silicon优化**: 支持MPS GPU加速
- **高质量摄像头**: 1080p 60fps实时处理
- **稳定预测**: 基于历史记录的预测平滑
- **Web界面**: 响应式设计，支持移动设备
- **可视化分析**: 丰富的训练和评估图表

## 🐛 故障排除

### 摄像头问题
- 确保给予浏览器摄像头权限
- 检查是否有其他应用占用摄像头
- 尝试重新启动服务

### 性能问题
- Apple Silicon用户建议使用MPS加速
- 降低批次大小如果内存不足
- 确保已安装正确版本的PyTorch

### 模型加载错误
- 检查models目录下是否有.pth文件
- 确保模型文件完整未损坏
- 重新训练模型如果必要

## 📄 许可证

MIT License - 详见LICENSE文件

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

---

**⭐ 如果这个项目对你有帮助，请给个Star支持一下！**
