# 核心文件: 是 - 模型训练主脚本
# 必须: 是 - 用于训练垃圾分类模型

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from font_config import setup_chinese_font

# 设置中文字体
setup_chinese_font()
import numpy as np
from model_utils import save_model, get_device, ProgressBar  # 导入ProgressBar类
import argparse

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# 英文图表生成函数
def setup_english_plotting():
    """Setup English plotting environment"""
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.unicode_minus'] = False

def plot_english_training_results(train_losses, train_accs, val_losses, val_accs, save_path="training_results_english.png"):
    """Plot training results with English labels"""
    setup_english_plotting()
    
    epochs = range(1, len(train_losses) + 1)
    
    # Create comprehensive training results plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Trash Classification Model Training Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Training and Validation Loss
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o', markersize=4)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=4)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training and Validation Accuracy
    ax2.plot(epochs, train_accs, 'g-', label='Training Accuracy', linewidth=2, marker='o', markersize=4)
    ax2.plot(epochs, val_accs, 'orange', label='Validation Accuracy', linewidth=2, marker='s', markersize=4)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Highlight best performance
    if val_accs:
        best_val_epoch = np.argmax(val_accs) + 1
        best_val_acc = max(val_accs)
        ax2.plot(best_val_epoch, best_val_acc, 'r*', markersize=15)
        ax2.annotate(f'Best: {best_val_acc:.4f}', 
                     xy=(best_val_epoch, best_val_acc),
                     xytext=(best_val_epoch + 1, best_val_acc - 0.02),
                     arrowprops=dict(facecolor='red', shrink=0.05),
                     fontsize=12, fontweight='bold')
    
    # Plot 3: Loss Comparison Bar Chart (last 10 epochs)
    recent_epochs = epochs[-min(10, len(epochs)):]
    recent_train_losses = train_losses[-min(10, len(train_losses)):]
    recent_val_losses = val_losses[-min(10, len(val_losses)):]
    
    x = np.arange(len(recent_epochs))
    width = 0.35
    ax3.bar(x - width/2, recent_train_losses, width, label='Training Loss', alpha=0.8)
    ax3.bar(x + width/2, recent_val_losses, width, label='Validation Loss', alpha=0.8)
    ax3.set_title('Recent Loss Comparison (Last 10 Epochs)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_xticks(x)
    ax3.set_xticklabels(recent_epochs)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Accuracy Improvement with Fill
    ax4.plot(epochs, train_accs, 'g-', label='Training Accuracy', linewidth=3, alpha=0.7)
    ax4.plot(epochs, val_accs, 'orange', label='Validation Accuracy', linewidth=3, alpha=0.7)
    ax4.fill_between(epochs, train_accs, alpha=0.3, color='green')
    ax4.fill_between(epochs, val_accs, alpha=0.3, color='orange')
    ax4.set_title('Accuracy Improvement Over Time', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ English training results chart saved: {save_path}")
    plt.close()

def plot_english_final_summary(train_losses, train_accs, val_losses, val_accs, final_epoch, save_path="final_summary_english.png"):
    """Plot final training summary with English labels"""
    setup_english_plotting()
    
    # Create summary plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Trash Classification Model - Final Training Summary', fontsize=16, fontweight='bold')
    
    # Final performance metrics
    final_metrics = [train_accs[-1], val_accs[-1], 1 - train_accs[-1], 1 - val_accs[-1]]
    labels = ['Train Accuracy', 'Val Accuracy', 'Train Error', 'Val Error']
    colors = ['#2E8B57', '#FF6347', '#98FB98', '#FFA07A']
    
    ax1.pie(final_metrics, labels=labels, colors=colors, autopct='%1.2f%%', startangle=90)
    ax1.set_title(f'Final Performance (Epoch {final_epoch})', fontsize=14)
    
    # Training progression
    epochs = range(1, len(train_accs) + 1)
    ax2.plot(epochs, val_accs, 'o-', linewidth=3, markersize=6, color='#4169E1', label='Validation Accuracy')
    ax2.plot(epochs, train_accs, 's-', linewidth=2, markersize=4, color='#32CD32', alpha=0.7, label='Training Accuracy')
    ax2.set_title('Accuracy Progression', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Loss progression
    ax3.plot(epochs, val_losses, 'o-', linewidth=3, markersize=6, color='#DC143C', label='Validation Loss')
    ax3.plot(epochs, train_losses, 's-', linewidth=2, markersize=4, color='#FF69B4', alpha=0.7, label='Training Loss')
    ax3.set_title('Loss Progression', fontsize=14)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Model performance statistics
    stats = {
        'Final Train Acc': train_accs[-1],
        'Final Val Acc': val_accs[-1],
        'Best Val Acc': max(val_accs),
        'Final Train Loss': train_losses[-1],
        'Final Val Loss': val_losses[-1],
        'Min Val Loss': min(val_losses)
    }
    
    y_pos = np.arange(len(stats))
    values = [round(v, 4) for v in stats.values()]
    bars = ax4.barh(y_pos, values, color=['#87CEEB', '#98FB98', '#FFB6C1', '#DDA0DD', '#F0E68C', '#FFA07A'])
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(stats.keys())
    ax4.set_title('Training Statistics Summary', fontsize=14)
    ax4.set_xlabel('Value')
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{values[i]:.4f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ English final summary chart saved: {save_path}")
    plt.close()

# 数据预处理和增强
def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform

# 数据集加载和处理类
class TrashDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']

        if self.transform:
            image = self.transform(image)

        return image, label

# 加载数据集并创建数据加载器
def prepare_data(batch_size=32, val_split=0.2):
    print("正在加载数据集...")

    # 创建数据集缓存目录
    cache_dir = os.path.join(os.getcwd(), 'dataset_cache')
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"已创建数据集缓存目录: {cache_dir}")
    else:
        print(f"使用本地缓存数据集: {cache_dir}")

    # 使用缓存目录加载数据集
    dataset = load_dataset("garythung/trashnet", cache_dir=cache_dir)
    train_dataset = dataset['train']
    print(f"✅ 数据集加载完成，共 {len(train_dataset)} 个样本")

    # 获取分类标签
    labels = train_dataset.features['label'].names
    num_classes = len(labels)
    print(f"分类类别: {labels}")

    # 获取数据转换
    train_transform, val_transform = get_transforms()

    # 计算分割大小
    dataset_size = len(train_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    # 分割数据集
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # 创建数据集实例
    train_ds = TrashDataset(train_subset, transform=train_transform)
    val_ds = TrashDataset(val_subset, transform=val_transform)

    # 创建数据加载器
    # 针对M3 Pro优化工作线程数(12核心CPU)
    num_workers = min(8, os.cpu_count() or 4)  # 使用可用CPU核心数，但最多8个
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)

    print(f"训练集大小: {len(train_subset)}")
    print(f"验证集大小: {len(val_subset)}")

    return train_loader, val_loader, num_classes, labels

# 创建模型
def create_model(num_classes):
    # 使用预训练的ResNet50模型
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # 冻结部分层以加速训练
    # 针对M3 Pro，只冻结前面的层以充分利用芯片性能
    trainable_layer_count = 30
    print(f"冻结ResNet50前面层，保留后{trainable_layer_count}层用于训练")
    for param in list(model.parameters())[:-trainable_layer_count]:
        param.requires_grad = False

    # 修改最后的全连接层以适应我们的分类任务
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )

    # 模型总结信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型信息: ResNet50")
    print(f"  总参数数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")

    return model

# 训练和验证函数
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 创建进度条
    progress_bar = ProgressBar(len(train_loader), prefix='训练进度:', suffix='完成', length=30)

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条
        progress_bar.update()

        # 实时显示批次训练信息
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            current_acc = correct / total
            current_loss = running_loss / total
            print(f"  批次: {batch_idx+1}/{len(train_loader)} | 损失: {current_loss:.4f} | 准确率: {current_acc:.4f}", end='\r')

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print()  # 新行

    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # 创建进度条
    progress_bar = ProgressBar(len(val_loader), prefix='验证进度:', suffix='完成', length=30)

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条
            progress_bar.update()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    # 按类别计算准确率
    class_correct = {}
    class_total = {}

    if hasattr(val_loader.dataset, 'dataset') and hasattr(val_loader.dataset.dataset, 'features'):
        class_names = val_loader.dataset.dataset.features['label'].names

        # 初始化计数器
        for i in range(len(class_names)):
            class_correct[class_names[i]] = 0
            class_total[class_names[i]] = 0

        # 重新计算每个类别的准确率
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                # 统计每个类别的准确率
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_name = class_names[label]
                    class_total[class_name] += 1
                    if predicted[i] == labels[i]:
                        class_correct[class_name] += 1

        # 输出每个类别的准确率
        print("\n各类别验证准确率:")
        for class_name in class_names:
            if class_total[class_name] > 0:
                accuracy = class_correct[class_name] / class_total[class_name]
                print(f"  {class_name}: {accuracy:.4f} ({class_correct[class_name]}/{class_total[class_name]})")

    return epoch_loss, epoch_acc

# 绘制训练过程
def plot_training_results(train_losses, val_losses, train_accs, val_accs, save_path='training_results.png'):
    plt.figure(figsize=(15, 10))

    # 设置中文字体 (如果系统支持)
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    except:
        print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")

    # 损失曲线
    plt.subplot(2, 1, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'bo-', label='训练损失')
    plt.plot(epochs, val_losses, 'ro-', label='验证损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    plt.title('训练和验证损失曲线')

    # 准确率曲线
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_accs, 'g*-', label='训练准确率')
    plt.plot(epochs, val_accs, 'm^-', label='验证准确率')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right')
    plt.title('训练和验证准确率曲线')

    # 标记最佳验证准确率点
    best_epoch = np.argmax(val_accs)
    best_acc = val_accs[best_epoch]
    plt.plot([best_epoch + 1], [best_acc], 'r*', markersize=15)
    plt.annotate(f'最佳: {best_acc:.4f}', 
                 xy=(best_epoch + 1, best_acc),
                 xytext=(best_epoch + 1 + 2, best_acc - 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ 训练结果图表已保存到 {save_path}")

    # 额外保存每个类别的准确率图表
    try:
        plot_class_accuracy(save_path.replace('.png', '_classes.png'))
    except Exception as e:
        print(f"无法生成类别准确率图表: {e}")

# 主训练函数
def train_model(args):
    # 准备数据
    train_loader, val_loader, num_classes, labels = prepare_data(args.batch_size, args.val_split)

    # 设置设备
    device = get_device()
    print(f"使用设备: {device}")

    # 针对M3芯片输出额外信息
    if device.type == 'mps':
        print("使用Apple Silicon M3 Pro的MPS加速训练")
        print(f"内存: 18GB, 建议批次大小: {args.batch_size}")

    # 创建模型
    model = create_model(num_classes)
    model = model.to(device)

    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)

    # 使用余弦退火学习率调度器，对长期训练更友好
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 训练循环
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    final_model_path = args.model_path.replace('.pth', '_final.pth')

    print("\n" + "="*50)
    print(f"🚀 开始训练 - 总轮次: {args.epochs}, 批次大小: {args.batch_size}")
    print("="*50)

    # 记录开始时间
    import time
    start_time = time.time()

    for epoch in range(args.epochs):
        # 显示当前轮次和时间
        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"\n📊 轮次 {epoch+1}/{args.epochs} [开始时间: {current_time}]")

        # 训练一个周期
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        # 保存统计数据
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # 计算剩余时间
        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_epochs = args.epochs - (epoch + 1)
        estimated_time = avg_time_per_epoch * remaining_epochs
        hours, remainder = divmod(estimated_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        # 格式化显示时间
        remaining_time = f"{int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒"

        # 打印本轮次总结
        print(f"\n📝 轮次总结:") 
        print(f"  训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"  验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  预计剩余时间: {remaining_time}")

        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            checkpoint_path = args.model_path.replace('.pth', f'_epoch{epoch+1}.pth')
            save_model(model, labels, checkpoint_path)
            print(f"💾 已保存模型检查点到 {checkpoint_path} (验证准确率: {val_acc:.4f})")

        # 实时绘制并保存训练曲线
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            tmp_plot_path = f"training_progress_epoch{epoch+1}.png"
            plot_training_results(train_losses, val_losses, train_accs, val_accs, save_path=tmp_plot_path)
            print(f"📈 已更新训练进度图表: {tmp_plot_path}")

    # 计算训练总时间
    train_time = time.time() - start_time
    hours, remainder = divmod(train_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    time_str = f"{int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒"

    print("\n" + "="*50)
    print(f"🎉 训练完成! 总用时: {time_str}")

    # 保存最终模型
    save_model(model, labels, final_model_path)
    print(f"💾 最终模型已保存到 {final_model_path}")

    # 绘制训练结果
    results_path = 'training_results.png'
    plot_training_results(train_losses, val_losses, train_accs, val_accs, save_path=results_path)

    # 找出最佳验证准确率
    best_epoch = np.argmax(val_accs)
    best_val_acc = val_accs[best_epoch]
    print(f"📊 最佳验证准确率: {best_val_acc:.4f} (第 {best_epoch+1} 轮)")

    # 计算并绘制混淆矩阵
    print("\n开始计算混淆矩阵和类别准确率...")
    try:
        class_accuracies = plot_confusion_matrix(model, val_loader, labels, device)

        # 输出每个类别的准确率
        print("\n📊 各类别准确率:")
        for class_name, accuracy in class_accuracies.items():
            print(f"  {class_name}: {accuracy:.4f}")
    except Exception as e:
        print(f"❌ 计算混淆矩阵时出错: {e}")

    print("\n🚀 模型训练与评估已全部完成!")
    
    # 生成英文版训练结果图表
    print("\n📊 生成英文版训练图表...")
    try:
        plot_english_training_results(train_losses, train_accs, val_losses, val_accs)
        plot_english_final_summary(train_losses, train_accs, val_losses, val_accs, args.epochs)
        print("✅ 英文图表生成完成!")
    except Exception as e:
        print(f"❌ 生成英文图表时出错: {e}")
    
    print("="*50)

    return model, labels

# 绘制类别准确率
def plot_class_accuracy(save_path='class_accuracy.png'):
    # 从model_results.npy加载结果（如果存在）
    results_file = 'model_results.npy'
    if not os.path.exists(results_file):
        print(f"未找到{results_file}，无法绘制类别准确率图表")
        return

    # 加载结果
    results = np.load(results_file, allow_pickle=True).item()
    class_names = results.get('class_names', [])
    class_accuracies = results.get('class_accuracies', {})

    if not class_accuracies:
        print("未找到类别准确率数据")
        return

    # 创建图表
    plt.figure(figsize=(12, 6))

    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    except:
        print("警告: 无法设置中文字体")

    # 排序并绘制准确率条形图
    classes = list(class_accuracies.keys())
    accuracies = [class_accuracies[cls] for cls in classes]

    # 颜色映射
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))

    # 绘制条形图
    bars = plt.bar(classes, accuracies, color=colors, alpha=0.8)

    # 在条形上方显示具体数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', rotation=0)

    plt.xlabel('类别')
    plt.ylabel('准确率')
    plt.title('各类别验证准确率')
    plt.ylim(0, 1.1)  # 确保y轴范围合适
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 保存图表
    plt.savefig(save_path, dpi=300)
    print(f"✅ 类别准确率图表已保存到 {save_path}")

# 绘制混淆矩阵
def plot_confusion_matrix(model, data_loader, class_names, device, save_path='confusion_matrix.png'):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    print("\n计算混淆矩阵...")

    # 收集所有预测和真实标签
    y_true = []
    y_pred = []

    # 创建进度条 (ProgressBar已在顶部导入)
    progress_bar = ProgressBar(len(data_loader), prefix='预测进度:', suffix='完成', length=30)

    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            # 更新进度条
            progress_bar.update()

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 创建图表
    plt.figure(figsize=(10, 8))

    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    except:
        print("警告: 无法设置中文字体")

    # 绘制热图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()

    # 保存图表
    plt.savefig(save_path, dpi=300)
    print(f"✅ 混淆矩阵图表已保存到 {save_path}")

    # 保存每个类别的准确率
    class_accuracies = {}
    for i in range(len(class_names)):
        class_accuracies[class_names[i]] = cm[i, i] / np.sum(cm[i])

    # 保存结果
    np.save('model_results.npy', {
        'confusion_matrix': cm,
        'class_names': class_names,
        'class_accuracies': class_accuracies
    })

    return class_accuracies

# 主函数
def main():
    parser = argparse.ArgumentParser(description='垃圾分类模型训练')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=40, help='训练周期数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集比例')
    parser.add_argument('--model_path', type=str, default='trash_classifier.pth', help='模型保存路径')

    args = parser.parse_args()

    # 检查模型保存目录是否存在
    model_dir = os.path.dirname(args.model_path)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 开始训练
    train_model(args)

if __name__ == '__main__':
    main()
