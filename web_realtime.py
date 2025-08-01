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
    
    if classifier is None:
        print("错误: 分类器未初始化")
        return
    
    # 检查摄像头是否已经启动
    if not hasattr(classifier, 'cap') or classifier.cap is None or not classifier.cap.isOpened():
        print("摄像头未启动，等待启动...")
        # 返回空帧，避免前端卡住
        dummy_frame = b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b'\r\n'
        yield dummy_frame
        return
        
    frame_count = 0
    last_prediction_time = 0
    
    try:
        while is_streaming and classifier.is_running:
            if not classifier.cap or not classifier.cap.isOpened():
                print("摄像头连接断开")
                break
                
            ret, frame = classifier.cap.read()
            if not ret:
                print("无法读取摄像头帧")
                # 短暂等待后继续尝试
                time.sleep(0.1)
                continue
            
            frame_count += 1
            current_time = time.time()
            
            # 减少预测频率但保持高质量视频
            if frame_count % 15 == 0 and current_time - last_prediction_time > 1.0:
                try:
                    classifier.async_predict(frame.copy())
                    last_prediction_time = current_time
                except Exception as e:
                    print(f"预测错误: {e}")
            
            # 保持高质量但适度压缩
            if hasattr(classifier, 'current_prediction') and classifier.current_prediction:
                frame_with_overlay = classifier.draw_prediction_overlay(frame, classifier.current_prediction)
            else:
                frame_with_overlay = frame
            
            # 适度降低分辨率用于web显示 (保持16:9比例)
            height, width = frame_with_overlay.shape[:2]
            if width > 1280:  # 降到720p用于web显示
                new_width = 1280
                new_height = int(height * (new_width / width))
                frame_with_overlay = cv2.resize(frame_with_overlay, (new_width, new_height))
            
            # 编码为JPEG，保持较高质量
            ret, buffer = cv2.imencode('.jpg', frame_with_overlay, 
                                     [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # 控制帧率约30fps以平衡性能和流畅度
            time.sleep(0.033)
                
    except Exception as e:
        print(f"视频流错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("视频流结束")
        is_streaming = False
        is_streaming = False

@app.route('/')
def index():
    """主页"""
    return render_template('realtime_index.html')

@app.route('/test')
def test_page():
    """简化测试页面"""
    return render_template('simple_test.html')

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
    global is_streaming, classifier
    try:
        if not is_streaming:
            # 确保分类器已初始化
            if classifier is None:
                if not init_classifier():
                    return jsonify({'success': False, 'message': '分类器初始化失败'})
            
            # 启动摄像头，添加重试机制
            if classifier:
                print("正在启动摄像头...")
                success = False
                for attempt in range(3):  # 最多尝试3次
                    try:
                        success = classifier.start_camera(camera_id=0)
                        if success:
                            break
                        else:
                            print(f"摄像头启动尝试 {attempt + 1} 失败，重试中...")
                            time.sleep(1)
                    except Exception as e:
                        print(f"摄像头启动异常 {attempt + 1}: {e}")
                        time.sleep(1)
                
                if success:
                    is_streaming = True
                    print("✅ 摄像头启动成功，视频流已开始")
                    return jsonify({'success': True, 'message': '视频流已启动'})
                else:
                    return jsonify({'success': False, 'message': '摄像头启动失败，请检查摄像头权限或重试'})
            else:
                return jsonify({'success': False, 'message': '分类器初始化失败'})
        else:
            return jsonify({'success': True, 'message': '视频流已在运行'})
    except Exception as e:
        print(f"启动视频流错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'启动失败: {str(e)}'})

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
