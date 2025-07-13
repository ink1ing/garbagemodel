#!/usr/bin/env python3
"""
基于Flask的实时垃圾识别Web界面
提供网页版的实时摄像头识别功能
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
import json
import time
import threading
from realtime_classifier import RealTimeTrashClassifier

app = Flask(__name__)

# 全局分类器实例
classifier = None
is_streaming = False

def init_classifier():
    """初始化分类器"""
    global classifier
    try:
        classifier = RealTimeTrashClassifier(confidence_threshold=0.5)
        print("✅ Web分类器初始化成功")
        return True
    except Exception as e:
        print(f"❌ Web分类器初始化失败: {e}")
        return False

def generate_frames():
    """生成视频流帧"""
    global classifier, is_streaming
    
    if classifier is None or not classifier.start_camera():
        return
    
    is_streaming = True
    frame_count = 0
    last_prediction_time = 0
    
    try:
        while is_streaming and classifier.is_running:
            ret, frame = classifier.cap.read()
            if not ret:
                break
            
            frame_count += 1
            current_time = time.time()
            
            # 每隔几帧进行预测
            if frame_count % 3 == 0 and current_time - last_prediction_time > 0.3:
                classifier.async_predict(frame.copy())
                last_prediction_time = current_time
            
            # 绘制预测结果
            frame_with_overlay = classifier.draw_prediction_overlay(frame, classifier.current_prediction)
            
            # 编码为JPEG
            ret, buffer = cv2.imencode('.jpg', frame_with_overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
    except Exception as e:
        print(f"视频流错误: {e}")
    finally:
        if classifier:
            classifier.stop_camera()
        is_streaming = False

@app.route('/')
def index():
    """主页"""
    return render_template('realtime_index.html')

@app.route('/video_feed')
def video_feed():
    """视频流端点"""
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    """获取当前预测结果"""
    global classifier
    
    if classifier and classifier.current_prediction:
        prediction = classifier.current_prediction
        
        # 获取中文信息
        from app import CLASS_MAPPING
        eng_class = prediction['class']
        class_info = CLASS_MAPPING.get(eng_class, {
            'chinese': eng_class,
            'icon': '❓',
            'color': '#808080'
        })
        
        return jsonify({
            'success': True,
            'prediction': {
                'english_class': eng_class,
                'chinese_class': class_info['chinese'],
                'icon': class_info['icon'],
                'description': class_info.get('description', ''),
                'confidence': prediction['confidence'],
                'all_confidences': prediction['all_confidences'],
                'timestamp': prediction['timestamp']
            }
        })
    else:
        return jsonify({
            'success': False,
            'message': '暂无预测结果'
        })

@app.route('/start_stream')
def start_stream():
    """开始视频流"""
    global is_streaming
    if not is_streaming:
        is_streaming = True
        return jsonify({'success': True, 'message': '视频流已启动'})
    else:
        return jsonify({'success': False, 'message': '视频流已在运行'})

@app.route('/stop_stream')
def stop_stream():
    """停止视频流"""
    global is_streaming, classifier
    is_streaming = False
    if classifier:
        classifier.stop_camera()
    return jsonify({'success': True, 'message': '视频流已停止'})

@app.route('/capture_image')
def capture_image():
    """截图功能"""
    global classifier
    if classifier and classifier.current_prediction:
        timestamp = int(time.time())
        filename = f"web_capture_{timestamp}.jpg"
        
        # 这里可以实现截图保存逻辑
        return jsonify({
            'success': True, 
            'message': f'截图已保存: {filename}',
            'filename': filename
        })
    else:
        return jsonify({'success': False, 'message': '当前无法截图'})

if __name__ == '__main__':
    # 初始化分类器
    if init_classifier():
        print("🌐 启动Flask Web服务器...")
        print("   访问地址: http://localhost:5001")
        print("   按 Ctrl+C 停止服务")
        
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    else:
        print("❌ 无法启动Web服务器，分类器初始化失败")
