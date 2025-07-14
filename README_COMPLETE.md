# 🤖 智能垃圾分类识别系统

> 基于深度学习的实时垃圾分类识别系统，支持Web界面、实时摄像头识别和高精度模型预测

## 📋 项目概述

本项目是一个完整的垃圾分类识别解决方案，使用ResNet50深度学习模型，能够准确识别6种常见垃圾类型：
- 📦 **纸板** (cardboard) - 瓦楞纸板、纸箱、包装盒等
- 🍶 **玻璃** (glass) - 玻璃瓶、玻璃杯、窗玻璃等  
- 🥫 **金属** (metal) - 易拉罐、金属餐具、废铁等
- 📄 **纸张** (paper) - 报纸、杂志、办公用纸等
- 🍼 **塑料** (plastic) - 塑料瓶、塑料袋、塑料容器等
- 🗑️ **其他垃圾** (trash) - 不可回收的混合垃圾等

## 🎯 核心特性

### 🧠 高精度模型
- **模型架构**: ResNet50 + 自定义分类层
- **训练轮次**: 41 epochs
- **最终准确率**: **99.40%**
- **设备支持**: Apple Silicon MPS / CUDA / CPU

### 🌐 多种使用方式
1. **Streamlit Web界面** - 上传图片进行分类
2. **Flask实时识别** - 摄像头实时流处理
3. **OpenCV桌面应用** - 本地摄像头窗口
4. **命令行工具** - 批量图片处理

### 📊 可视化分析
- 混淆矩阵分析
- 详细性能指标
- 训练过程可视化
- 实时置信度显示

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- 其他依赖（见requirements.txt）

### 安装步骤

1. **克隆项目**
```bash
git clone <项目地址>
cd canva1
```

2. **创建虚拟环境**
```bash
python -m venv trash_classifier_env
source trash_classifier_env/bin/activate  # macOS/Linux
# trash_classifier_env\Scripts\activate   # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **运行系统**
```bash
# 使用统一启动脚本
bash launch_realtime.sh
```

## 💻 使用指南

### 方式一：Streamlit Web界面
```bash
streamlit run app.py
```
- 访问: http://localhost:8501
- 功能: 上传图片，获得详细分类结果
- 特色: 中英文对照、置信度分析、处理建议

### 方式二：Flask实时识别
```bash
python optimized_web_realtime.py
```
- 访问: http://localhost:5001
- 功能: 实时摄像头识别
- 特色: 实时预测、平滑算法、美观界面

### 方式三：OpenCV桌面应用
```bash
python realtime_classifier.py
```
- 功能: 本地摄像头窗口
- 操作: 按'q'退出，按's'截图
- 特色: 低延迟、直接显示

### 方式四：图片测试工具
```bash
python image_test_classifier.py -i 图片路径
python image_test_classifier.py -c  # 摄像头测试
```

## 📊 模型性能

### 整体指标
- **准确率**: 99.40%
- **宏平均F1**: 0.9945
- **加权平均F1**: 0.9940

### 各类别详细指标
| 类别 | 精确率 | 召回率 | F1分数 |
|------|--------|--------|--------|
| 纸板 | 100.0% | 100.0% | 100.0% |
| 玻璃 | 100.0% | 97.0% | 98.5% |
| 金属 | 96.6% | 100.0% | 98.3% |
| 纸张 | 100.0% | 100.0% | 100.0% |
| 塑料 | 100.0% | 100.0% | 100.0% |
| 其他垃圾 | 100.0% | 100.0% | 100.0% |

## 🛠️ 技术架构

### 模型结构
```
ResNet50 主干网络
    ↓
Dropout (0.5)
    ↓
Linear (2048 → 6)
    ↓
Softmax 输出
```

### 数据预处理
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### 核心组件
- **model_utils.py** - 模型加载、保存、预测工具
- **app.py** - Streamlit主界面
- **optimized_web_realtime.py** - Flask实时识别服务
- **realtime_classifier.py** - OpenCV实时分类器
- **generate_confusion_matrix.py** - 性能评估工具

## 📁 项目结构

```
canva1/
├── 📱 Web界面
│   ├── app.py                      # Streamlit主界面
│   ├── optimized_web_realtime.py   # Flask实时识别
│   └── templates/                  # HTML模板
├── 🎥 实时识别
│   ├── realtime_classifier.py      # OpenCV桌面版
│   ├── web_realtime.py            # Flask基础版
│   └── simple_camera_test.py      # 摄像头测试
├── 🧠 模型相关
│   ├── train.py                   # 训练脚本
│   ├── model_utils.py             # 模型工具
│   └── models/                    # 训练好的模型
├── 📊 评估工具
│   ├── generate_confusion_matrix.py # 混淆矩阵生成
│   ├── confusion_matrix.png       # 混淆矩阵图
│   └── detailed_metrics.png       # 详细指标图
├── 🚀 启动脚本
│   ├── launch_realtime.sh         # 统一启动脚本
│   └── run_realtime.sh           # 实时识别启动
└── 📋 配置文件
    ├── requirements.txt           # 依赖列表
    └── font_config.py            # 字体配置
```

## 🔧 高级功能

### 训练自定义模型
```bash
python train.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

### 生成评估报告
```bash
python generate_confusion_matrix.py
```

### 摄像头问题诊断
```bash
python simple_camera_test.py
```

## 🌟 特色功能

### 🎨 美观界面
- 现代化设计风格
- 响应式布局
- 实时状态指示
- 动画效果

### 🚀 高性能
- Apple Silicon MPS加速
- 实时预测平滑算法
- 多线程处理
- 内存优化

### 🔍 智能分析
- 置信度可视化
- 预测结果解释
- 处理建议系统
- 详细性能报告

### 🛡️ 稳定可靠
- 异常处理机制
- 摄像头错误恢复
- 模型自动加载
- 日志记录系统

## 🐛 故障排除

### 常见问题

**Q: 摄像头显示黑屏？**
A: 使用简化摄像头测试工具诊断：
```bash
python simple_camera_test.py
```

**Q: 模型加载失败？**
A: 检查模型文件是否存在：
```bash
ls models/
```

**Q: 依赖安装失败？**
A: 使用虚拟环境重新安装：
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Q: macOS摄像头权限问题？**
A: 在系统偏好设置 → 安全性与隐私 → 摄像头中授权Python/Terminal

## 📈 性能优化建议

1. **硬件加速**
   - 使用Apple Silicon MPS
   - 启用CUDA（如有NVIDIA GPU）

2. **摄像头优化**
   - 降低分辨率提升帧率
   - 调整缓冲区大小
   - 使用合适的编码质量

3. **预测优化**
   - 控制预测频率
   - 启用结果平滑
   - 异步处理

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见LICENSE文件

## 👥 致谢

- **数据集**: garythung/trashnet
- **框架**: PyTorch, Streamlit, Flask, OpenCV
- **模型**: ResNet50预训练模型

---

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 📧 Email: [联系邮箱]
- 💬 Issues: [项目Issues页面]

---

**🎉 感谢使用智能垃圾分类识别系统！让我们一起为环保贡献力量！**
