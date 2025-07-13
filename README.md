# 智能垃圾分类系统

基于深度学习的实时垃圾分类识别系统，支持摄像头实时识别和分类。

## 功能特点

- 实时视频流垃圾识别
- 支持多种常见垃圾类型分类
- 提供Web界面展示
- 高准确率的分类模型

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- Flask

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行实时分类系统：
```bash
python realtime_classifier.py
```

3. 运行Web界面：
```bash
python web_realtime.py
```

## 模型训练

如需重新训练模型：

```bash
python train.py
```

## 注意事项

- 模型文件需要单独下载
- 确保摄像头权限已开启
- 建议使用GPU进行实时识别

## 许可证

MIT License