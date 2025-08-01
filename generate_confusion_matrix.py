#!/usr/bin/env python3
"""
生成混淆矩阵和详细评估指标
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from datasets import load_dataset
from torch.utils.data import DataLoader
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Hiragino Sans GB', 'STSong', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_test_data():
    """加载测试数据"""
    from train import TrashDataset, get_transforms
    
    # 加载数据集
    dataset = load_dataset("garythung/trashnet")
    
    # 获取验证变换
    _, val_transform = get_transforms()
    
    # 创建测试数据集
    test_dataset = TrashDataset(dataset['train'], transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return test_loader, test_dataset.dataset.features['label'].names

def evaluate_model():
    """评估模型并生成图表"""
    from model_utils import load_model, get_device
    from app import create_model
    
    # 加载模型
    device = get_device()
    model = create_model(6)
    model, class_names = load_model(model, 'models/trash_classifier_final.pth')
    model.eval()
    
    # 加载测试数据
    test_loader, label_names = load_test_data()
    
    # 进行预测
    all_predictions = []
    all_labels = []
    
    print("🔍 正在评估模型...")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f"已处理 {batch_idx * len(data)} 张图片...")
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    # 创建results/charts目录
    os.makedirs('results/charts', exist_ok=True)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('垃圾分类模型混淆矩阵', fontsize=16, fontweight='bold')
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.tight_layout()
    plt.savefig('results/charts/confusion_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 混淆矩阵已保存到: results/charts/confusion_matrix.png")
    plt.close()
    
    # 生成分类报告
    report = classification_report(all_labels, all_predictions, 
                                 target_names=class_names, output_dict=True)
    
    # 绘制详细指标图
    metrics = ['precision', 'recall', 'f1-score']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        values = [report[class_name][metric] for class_name in class_names]
        bars = axes[i].bar(class_names, values, alpha=0.8, color=plt.cm.viridis(np.linspace(0, 1, len(class_names))))
        axes[i].set_title(f'{metric.capitalize()}', fontsize=14, fontweight='bold')
        axes[i].set_ylabel(metric.capitalize())
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('分类详细指标', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/charts/detailed_metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 详细指标图已保存到: results/charts/detailed_metrics.png")
    plt.close()
    
    # 打印总体准确率
    accuracy = np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
    print(f"\n📊 模型总体准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

if __name__ == "__main__":
    evaluate_model()
