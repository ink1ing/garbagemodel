#!/bin/bash
# GitHub仓库连接脚本
# 请先在GitHub上创建仓库，然后将下面的 YOUR_USERNAME 和 REPO_NAME 替换为实际值

echo "🔗 连接到GitHub仓库..."

# 替换为您的GitHub用户名和仓库名
GITHUB_USERNAME="YOUR_USERNAME"
REPO_NAME="trash-classification-ml"

# 添加远程仓库
git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git

# 设置主分支
git branch -M main

# 推送到GitHub
git push -u origin main

echo "✅ 已成功推送到GitHub仓库!"
echo "🌐 仓库地址: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
