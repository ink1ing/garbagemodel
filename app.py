#!/usr/bin/env python3
"""
垃圾分类模型应用模块
包含模型创建和类别映射等核心功能
"""

import torch
import torch.nn as nn
from torchvision import models

# 垃圾分类类别映射 (英文类别名 -> 中文信息)
CLASS_MAPPING = {
    'cardboard': {
        'chinese': '纸板',
        'icon': '📦',
        'category': '可回收垃圾',
        'color': '#4CAF50'
    },
    'glass': {
        'chinese': '玻璃',
        'icon': '🍶',
        'category': '可回收垃圾',
        'color': '#2196F3'
    },
    'metal': {
        'chinese': '金属',
        'icon': '🔧',
        'category': '可回收垃圾',
        'color': '#9E9E9E'
    },
    'paper': {
        'chinese': '纸张',
        'icon': '📄',
        'category': '可回收垃圾',
        'color': '#FF9800'
    },
    'plastic': {
        'chinese': '塑料',
        'icon': '🥤',
        'category': '可回收垃圾',
        'color': '#E91E63'
    },
    'trash': {
        'chinese': '其他垃圾',
        'icon': '🗑️',
        'category': '其他垃圾',
        'color': '#607D8B'
    }
}

def create_model(num_classes):
    """
    创建ResNet50垃圾分类模型
    
    Args:
        num_classes: 分类类别数量
        
    Returns:
        torch.nn.Module: 预训练的ResNet50模型
    """
    # 使用预训练的ResNet50模型
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # 冻结部分层以加速训练
    trainable_layer_count = 30
    for param in list(model.parameters())[:-trainable_layer_count]:
        param.requires_grad = False

    # 修改最后的全连接层以适应我们的分类任务
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    return model

def get_class_info(class_name):
    """
    获取类别的详细信息
    
    Args:
        class_name: 英文类别名
        
    Returns:
        dict: 包含中文名、图标、类别、颜色的字典
    """
    return CLASS_MAPPING.get(class_name, {
        'chinese': class_name,
        'icon': '❓',
        'category': '未知',
        'color': '#000000'
    })

if __name__ == "__main__":
    # 测试模型创建
    model = create_model(6)
    print(f"模型创建成功，类别数: 6")
    print(f"类别映射: {list(CLASS_MAPPING.keys())}")
