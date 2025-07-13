# 核心文件: 是 - 模型工具函数
# 必须: 是 - 用于模型的加载、保存和预测

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np

# 获取设备
def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("使用 Apple Silicon MPS 加速训练")
        return torch.device('mps')  # 使用M系列芯片的Metal加速
    elif torch.cuda.is_available():
        print("使用 NVIDIA GPU CUDA 加速训练")
        return torch.device('cuda')
    else:
        print("使用 CPU 进行训练")
        return torch.device('cpu')

# 训练进度条
class ProgressBar:
    def __init__(self, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
        """
        初始化进度条
        :param total: 总迭代次数
        :param prefix: 前缀字符串
        :param suffix: 后缀字符串
        :param decimals: 进度百分比小数位数
        :param length: 进度条长度
        :param fill: 进度条填充字符
        """
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.fill = fill
        self.iteration = 0

    def update(self):
        """
        更新进度条
        """
        self.iteration += 1
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (self.iteration / float(self.total)))
        filled_length = int(self.length * self.iteration // self.total)
        bar = self.fill * filled_length + '-' * (self.length - filled_length)
        print(f'\r{self.prefix} |{bar}| {percent}% {self.suffix}', end='\r')
        # 打印新行当迭代完成
        if self.iteration == self.total: 
            print()

# 保存模型
def save_model(model, class_names, path='trash_classifier.pth'):
    model_info = {
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }
    torch.save(model_info, path)
    print(f"模型已保存到 {path}")

# 加载模型
def load_model(model, path='trash_classifier.pth'):
    try:
        device = get_device()
        # 安全加载模型，确保设备兼容性
        model_info = torch.load(path, map_location=device)
        model.load_state_dict(model_info['model_state_dict'])
        class_names = model_info['class_names']
        
        # 确保模型在正确的设备上
        model = model.to(device)
        model.eval()  # 设置为评估模式
        
        print(f"✅ 模型已加载到设备: {device}")
        return model, class_names
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise e

# 图像预处理转换
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 预测单个图像
def predict_image(model, image, class_names):
    try:
        device = get_device()
        transform = get_transform()

        # 如果是文件路径，打开图像
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        # 如果是字节数据，转换为图像
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert('RGB')
        # 如果已经是PIL图像，确保为RGB模式
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            raise ValueError("不支持的图像格式，请提供PIL图像、文件路径或图像字节数据")

        # 应用转换
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # 确保模型在正确的设备上
        model = model.to(device)
        model.eval()

        # 预测
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_idx = torch.argmax(probabilities).item()

        # 获取分类和置信度
        predicted_class = class_names[predicted_idx]
        confidence = probabilities[predicted_idx].item()

        # 获取所有类别的置信度
        all_confidences = {class_names[i]: probabilities[i].item() for i in range(len(class_names))}

        return {
            'class': predicted_class,
            'confidence': confidence,
            'all_confidences': all_confidences
        }
    except Exception as e:
        print(f"❌ 预测过程中出错: {e}")
        raise e

# 可视化预测结果
def visualize_prediction(image, prediction_result, title=None):
    import matplotlib.pyplot as plt

    # 如果是文件路径，打开图像
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    # 如果是字节数据，转换为图像
    elif isinstance(image, bytes):
        image = Image.open(io.BytesIO(image)).convert('RGB')

    # 排序置信度
    confidences = prediction_result['all_confidences']
    sorted_confidences = dict(sorted(confidences.items(), key=lambda item: item[1], reverse=True))

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 显示图像
    ax1.imshow(image)
    ax1.set_title(title or f"预测: {prediction_result['class']} ({prediction_result['confidence']:.2%})")
    ax1.axis('off')

    # 显示置信度条形图
    classes = list(sorted_confidences.keys())
    values = list(sorted_confidences.values())

    bars = ax2.barh(classes, values)
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel('置信度')
    ax2.set_title('各类别置信度')

    # 在条形上添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2%}', 
                 va='center')

    plt.tight_layout()
    return fig
