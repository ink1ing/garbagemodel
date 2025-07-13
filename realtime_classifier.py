#!/usr/bin/env python3
"""
实时垃圾识别模块
使用OpenCV进行摄像头捕获和实时预测
"""

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
from collections import deque
import threading
import queue

# 导入模型相关模块
from model_utils import get_device, load_model
from app import create_model, CLASS_MAPPING

class RealTimeTrashClassifier:
    def __init__(self, model_path=None, confidence_threshold=0.6):
        """
        初始化实时垃圾分类器
        
        Args:
            model_path: 模型文件路径
            confidence_threshold: 置信度阈值
        """
        self.device = get_device()
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.class_names = None
        
        # 预测历史记录（用于稳定预测结果）
        self.prediction_history = deque(maxlen=5)
        
        # 摄像头设置
        self.cap = None
        self.is_running = False
        
        # 预测队列（用于异步预测）
        self.prediction_queue = queue.Queue(maxsize=1)
        self.current_prediction = None
        
        # 加载模型
        self.load_model(model_path)
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ 实时垃圾分类器初始化完成")
        print(f"   设备: {self.device}")
        print(f"   置信度阈值: {confidence_threshold}")
    
    def load_model(self, model_path=None):
        """加载模型"""
        try:
            # 如果没有指定路径，自动寻找最新模型
            if model_path is None:
                import os
                if os.path.exists('models'):
                    model_files = [f for f in os.listdir('models') if f.endswith('.pth')]
                    if model_files:
                        # 优先选择final模型
                        if 'trash_classifier_final.pth' in model_files:
                            model_path = os.path.join('models', 'trash_classifier_final.pth')
                        else:
                            # 选择最新的模型
                            model_files.sort(key=lambda x: os.path.getmtime(os.path.join('models', x)), reverse=True)
                            model_path = os.path.join('models', model_files[0])
                    else:
                        raise FileNotFoundError("未找到模型文件")
                else:
                    raise FileNotFoundError("models目录不存在")
            
            # 创建并加载模型（使用正确的ResNet50架构）
            import torchvision.models as models
            import torch.nn as nn
            
            self.model = models.resnet50(weights=None)
            self.model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.model.fc.in_features, 6)  # 6个类别
            )
            
            self.model, self.class_names = load_model(self.model, model_path)
            self.model.eval()
            
            print(f"✅ 模型加载成功: {model_path}")
            print(f"   支持类别: {self.class_names}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise e
    
    def preprocess_frame(self, frame):
        """预处理摄像头帧"""
        # 转换BGR到RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像
        pil_image = Image.fromarray(rgb_frame)
        
        # 应用变换
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        return tensor, pil_image
    
    def predict_frame(self, frame):
        """预测单帧图像"""
        try:
            if self.model is None:
                return None
            
            # 预处理
            tensor, pil_image = self.preprocess_frame(frame)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted_idx = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_idx].item()
            
            # 获取预测结果
            predicted_class = self.class_names[predicted_idx]
            
            # 获取所有类别的置信度
            all_confidences = {
                self.class_names[i]: probabilities[i].item() 
                for i in range(len(self.class_names))
            }
            
            return {
                'class': predicted_class,
                'confidence': confidence,
                'all_confidences': all_confidences,
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"预测错误: {e}")
            return None
    
    def get_stable_prediction(self, new_prediction):
        """获取稳定的预测结果（基于历史记录）"""
        if new_prediction is None:
            return self.current_prediction
        
        # 添加到历史记录
        self.prediction_history.append(new_prediction)
        
        # 如果置信度足够高，直接返回
        if new_prediction['confidence'] > 0.8:
            return new_prediction
        
        # 否则基于历史记录进行平滑
        if len(self.prediction_history) >= 3:
            # 统计最常见的类别
            class_counts = {}
            total_confidence = 0
            
            for pred in self.prediction_history:
                cls = pred['class']
                conf = pred['confidence']
                
                if cls not in class_counts:
                    class_counts[cls] = {'count': 0, 'total_conf': 0}
                
                class_counts[cls]['count'] += 1
                class_counts[cls]['total_conf'] += conf
                total_confidence += conf
            
            # 找到最常见且置信度合理的类别
            best_class = max(class_counts.items(), 
                           key=lambda x: x[1]['count'] * x[1]['total_conf'])
            
            avg_confidence = best_class[1]['total_conf'] / best_class[1]['count']
            
            if avg_confidence > self.confidence_threshold:
                return {
                    'class': best_class[0],
                    'confidence': avg_confidence,
                    'all_confidences': new_prediction['all_confidences'],
                    'timestamp': new_prediction['timestamp'],
                    'stable': True
                }
        
        return new_prediction
    
    def async_predict(self, frame):
        """异步预测（在单独线程中）"""
        if not self.prediction_queue.full():
            try:
                self.prediction_queue.put_nowait(frame)
            except queue.Full:
                pass
    
    def prediction_worker(self):
        """预测工作线程"""
        while self.is_running:
            try:
                frame = self.prediction_queue.get(timeout=0.1)
                prediction = self.predict_frame(frame)
                self.current_prediction = self.get_stable_prediction(prediction)
                self.prediction_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"预测线程错误: {e}")
    
    def start_camera(self, camera_id=0):
        """启动摄像头"""
        try:
            self.cap = cv2.VideoCapture(camera_id)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"无法打开摄像头 {camera_id}")
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            
            # 启动预测线程
            self.prediction_thread = threading.Thread(target=self.prediction_worker, daemon=True)
            self.prediction_thread.start()
            
            print("✅ 摄像头启动成功")
            return True
            
        except Exception as e:
            print(f"❌ 摄像头启动失败: {e}")
            return False
    
    def stop_camera(self):
        """停止摄像头"""
        self.is_running = False
        
        if self.cap is not None:
            self.cap.release()
            
        cv2.destroyAllWindows()
        print("✅ 摄像头已停止")
    
    def draw_prediction_overlay(self, frame, prediction):
        """在帧上绘制预测结果覆盖层"""
        if prediction is None:
            # 显示"正在识别..."
            cv2.putText(frame, "Detecting...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return frame
        
        # 获取类别信息
        eng_class = prediction['class']
        confidence = prediction['confidence']
        
        class_info = CLASS_MAPPING.get(eng_class, {
            'chinese': eng_class,
            'icon': '❓',
            'color': '#808080'
        })
        
        # 确定颜色（基于置信度）
        if confidence > 0.8:
            color = (0, 255, 0)  # 绿色 - 高置信度
        elif confidence > 0.6:
            color = (0, 255, 255)  # 黄色 - 中等置信度
        else:
            color = (0, 0, 255)  # 红色 - 低置信度
        
        # 绘制主要结果
        chinese_name = class_info['chinese']
        main_text = f"{class_info['icon']} {chinese_name}"
        confidence_text = f"Confidence: {confidence:.1%}"
        
        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 绘制文本
        cv2.putText(frame, main_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, confidence_text, (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"English: {eng_class}", (20, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 绘制置信度条
        bar_width = int(300 * confidence)
        cv2.rectangle(frame, (20, 105), (320, 115), (100, 100, 100), 1)
        cv2.rectangle(frame, (20, 105), (20 + bar_width, 115), color, -1)
        
        # 绘制其他类别的置信度（前3个）
        y_offset = 140
        sorted_confidences = sorted(prediction['all_confidences'].items(), 
                                  key=lambda x: x[1], reverse=True)
        
        for i, (cls, conf) in enumerate(sorted_confidences[:3]):
            if cls != eng_class:  # 跳过主要预测类别
                cls_info_other = CLASS_MAPPING.get(cls, {'chinese': cls, 'icon': '❓'})
                other_text = f"{cls_info_other['icon']} {cls_info_other['chinese']}: {conf:.1%}"
                cv2.putText(frame, other_text, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                y_offset += 20
        
        return frame
    
    def run_realtime(self):
        """运行实时识别"""
        if not self.start_camera():
            return
        
        print("🎥 实时垃圾识别已启动")
        print("   按 'q' 键退出")
        print("   按 's' 键截图保存")
        
        frame_count = 0
        last_prediction_time = 0
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ 无法读取摄像头画面")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # 每隔几帧进行一次预测（减少计算负担）
                if frame_count % 5 == 0 and current_time - last_prediction_time > 0.5:
                    self.async_predict(frame.copy())
                    last_prediction_time = current_time
                
                # 绘制预测结果
                frame_with_overlay = self.draw_prediction_overlay(frame, self.current_prediction)
                
                # 添加FPS信息
                fps_text = f"FPS: {1.0 / (current_time - last_prediction_time + 0.001):.1f}"
                cv2.putText(frame_with_overlay, fps_text, (500, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 显示画面
                cv2.imshow('Real-time Trash Classification', frame_with_overlay)
                
                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存截图
                    timestamp = int(time.time())
                    filename = f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(filename, frame_with_overlay)
                    print(f"📸 截图已保存: {filename}")
        
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断")
        
        except Exception as e:
            print(f"❌ 运行时错误: {e}")
        
        finally:
            self.stop_camera()

def main():
    """主函数"""
    print("🚀 启动实时垃圾分类识别系统...")
    
    try:
        # 创建分类器
        classifier = RealTimeTrashClassifier(confidence_threshold=0.5)
        
        # 运行实时识别
        classifier.run_realtime()
        
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
