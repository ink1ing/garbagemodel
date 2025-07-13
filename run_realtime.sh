#!/bin/bash

echo "🚀 实时垃圾分类识别系统启动脚本"
echo "======================================"

# 检查虚拟环境
if [ ! -d "trash_classifier_env" ]; then
    echo "❌ 虚拟环境不存在，请先运行 bash run_app.sh 或创建虚拟环境"
    exit 1
fi

echo "🔍 检查依赖..."
source trash_classifier_env/bin/activate

# 检查OpenCV
python -c "import cv2; print(f'✅ OpenCV版本: {cv2.__version__}')" 2>/dev/null || {
    echo "⚠️ 正在安装OpenCV..."
    pip install opencv-python
}

# 检查Flask
python -c "import flask; print(f'✅ Flask版本: {flask.__version__}')" 2>/dev/null || {
    echo "⚠️ 正在安装Flask..."
    pip install flask
}

echo ""
echo "🎯 选择启动模式:"
echo "1. OpenCV桌面版 (推荐用于测试)"
echo "2. Flask Web版 (推荐用于演示)"
echo ""

read -p "请选择模式 [1/2]: " mode

case $mode in
    1)
        echo "🖥️ 启动OpenCV桌面版..."
        echo "   操作说明:"
        echo "   - 按 'q' 键退出"
        echo "   - 按 's' 键截图"
        echo ""
        python realtime_classifier.py
        ;;
    2)
        echo "🌐 启动Flask Web版..."
        echo "   访问地址: http://localhost:5000"
        echo "   按 Ctrl+C 停止服务"
        echo ""
        python web_realtime.py
        ;;
    *)
        echo "❌ 无效选择，退出"
        exit 1
        ;;
esac
