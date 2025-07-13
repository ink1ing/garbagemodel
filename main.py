# 核心文件: 是 - 项目主入口
# 必须: 是 - 提供模型训练和应用启动功能

import argparse
import os
import subprocess
import sys

def print_header(message):
    print("\n" + "-" * 50)
    print(f" {message} ")
    print("-" * 50)

def check_requirements():
    print_header("检查依赖项")
    try:
        import torch
        import streamlit
        import datasets
        print("✅ 基本依赖已安装")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请先运行: pip install -r requirements.txt")
        return False

def train_model(args):
    print_header("开始训练模型")
    try:
        cmd = [
            sys.executable, "train.py", 
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--learning_rate", str(args.learning_rate),
            "--model_path", args.model_path
        ]
        subprocess.run(cmd, check=True)
        print("✅ 模型训练完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 模型训练失败: {e}")
        return False

def start_app():
    print_header("启动Streamlit应用")
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 应用启动失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="垃圾分类识别系统 - 主控制脚本")
    subparsers = parser.add_subparsers(dest="command", help="命令")

    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--epochs", type=int, default=20, help="训练周期数")
    train_parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    train_parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    train_parser.add_argument("--model_path", type=str, default="trash_classifier.pth", help="模型保存路径")

    # 运行应用命令
    app_parser = subparsers.add_parser("app", help="启动Streamlit应用")

    # 一体化命令
    all_parser = subparsers.add_parser("all", help="训练模型并启动应用")
    all_parser.add_argument("--epochs", type=int, default=20, help="训练周期数")
    all_parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    all_parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    all_parser.add_argument("--model_path", type=str, default="trash_classifier.pth", help="模型保存路径")

    args = parser.parse_args()

    # 检查依赖项
    if not check_requirements():
        return

    # 根据命令执行相应功能
    if args.command == "train":
        train_model(args)
    elif args.command == "app":
        if not os.path.exists("trash_classifier.pth"):
            print("⚠️ 警告: 未找到模型文件，请先训练模型或确保模型文件存在")
            choice = input("是否继续启动应用? (y/n): ")
            if choice.lower() != 'y':
                return
        start_app()
    elif args.command == "all":
        if train_model(args):
            start_app()
    else:
        print("请指定命令: train, app 或 all")
        parser.print_help()

if __name__ == "__main__":
    main()
