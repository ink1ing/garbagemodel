#!/usr/bin/env python3
"""
垃圾桶二维码管理系统
功能：生成、管理、存储垃圾桶绑定的二维码
支持批量生成、单独操作（获取/作废/更新）等功能
"""

import qrcode
import json
import os
import uuid
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import shutil

class QRCodeManager:
    def __init__(self, storage_dir="qr_codes"):
        """
        初始化二维码管理器
        
        Args:
            storage_dir: 二维码存储目录
        """
        self.storage_dir = storage_dir
        self.config_file = os.path.join(storage_dir, "qr_config.json")
        self.active_dir = os.path.join(storage_dir, "active")      # 活跃二维码
        self.archived_dir = os.path.join(storage_dir, "archived")  # 已作废二维码
        self.captured_dir = os.path.join(storage_dir, "captured")  # 获取的二维码
        
        # 创建目录结构
        self._create_directories()
        
        # 加载配置
        self.config = self._load_config()
        
    def _create_directories(self):
        """创建必要的目录结构"""
        for directory in [self.storage_dir, self.active_dir, 
                         self.archived_dir, self.captured_dir]:
            os.makedirs(directory, exist_ok=True)
        print(f"✅ 目录结构创建完成: {self.storage_dir}")
        
    def _load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # 初始化空配置
            config = {
                "trash_bins": {},
                "total_generated": 0,
                "last_update": None
            }
            self._save_config(config)
            return config
    
    def _save_config(self, config=None):
        """保存配置文件"""
        if config is None:
            config = self.config
        config["last_update"] = datetime.now().isoformat()
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        self.config = config
        
    def _generate_unique_id(self):
        """生成唯一ID"""
        return str(uuid.uuid4())
    
    def _create_qr_data(self, bin_id, unique_id):
        """
        创建二维码数据
        
        Args:
            bin_id: 垃圾桶编号 (如 "no.1")
            unique_id: 唯一标识符
            
        Returns:
            dict: 二维码数据
        """
        qr_data = {
            "bin_id": bin_id,
            "unique_id": unique_id,
            "created_time": datetime.now().isoformat(),
            "type": "trash_bin_qr",
            "status": "active"
        }
        return qr_data
        
    def _generate_qr_image(self, qr_data, bin_id):
        """
        生成二维码图片
        
        Args:
            qr_data: 二维码数据
            bin_id: 垃圾桶编号
            
        Returns:
            str: 保存的图片路径
        """
        # 将数据转换为JSON字符串
        qr_content = json.dumps(qr_data, ensure_ascii=False)
        
        # 创建二维码
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_content)
        qr.make(fit=True)
        
        # 创建二维码图像
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # 转换为RGB模式确保兼容性
        if qr_img.mode != 'RGB':
            qr_img = qr_img.convert('RGB')
        
        # 创建带标签的图像 (添加垃圾桶编号)
        img_width, img_height = qr_img.size
        label_height = 80
        final_img = Image.new('RGB', (img_width, img_height + label_height), 'white')
        
        # 粘贴二维码到指定位置
        final_img.paste(qr_img, (0, 0, img_width, img_height))
        
        # 添加文字标签
        draw = ImageDraw.Draw(final_img)
        try:
            # 尝试使用系统字体
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
        except:
            # 使用默认字体
            font = ImageFont.load_default()
            
        # 绘制垃圾桶编号
        text = f"垃圾桶 {bin_id.upper()}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (img_width - text_width) // 2
        text_y = img_height + (label_height - 24) // 2
        
        draw.text((text_x, text_y), text, fill="black", font=font)
        
        # 保存图片
        filename = f"{bin_id}_qr_{int(time.time())}.png"
        filepath = os.path.join(self.active_dir, filename)
        final_img.save(filepath)
        
        return filepath
        
    def generate_qr_codes(self, count):
        """
        批量生成二维码
        
        Args:
            count: 生成数量
            
        Returns:
            list: 生成的二维码信息列表
        """
        if count <= 0:
            print("❌ 生成数量必须大于0")
            return []
            
        generated_codes = []
        
        print(f"🔄 开始生成 {count} 个垃圾桶二维码...")
        
        for i in range(1, count + 1):
            bin_id = f"no.{i}"
            unique_id = self._generate_unique_id()
            
            # 如果该垃圾桶已有活跃二维码，先作废
            if bin_id in self.config["trash_bins"]:
                old_qr = self.config["trash_bins"][bin_id]
                if old_qr["status"] == "active":
                    print(f"⚠️  垃圾桶 {bin_id} 已有活跃二维码，自动作废旧码")
                    self._invalidate_qr(bin_id)
            
            # 创建新二维码数据
            qr_data = self._create_qr_data(bin_id, unique_id)
            
            # 生成二维码图片
            filepath = self._generate_qr_image(qr_data, bin_id)
            
            # 更新配置
            self.config["trash_bins"][bin_id] = {
                "unique_id": unique_id,
                "created_time": qr_data["created_time"],
                "filepath": filepath,
                "status": "active"
            }
            
            generated_codes.append({
                "bin_id": bin_id,
                "filepath": filepath,
                "unique_id": unique_id
            })
            
            print(f"✅ 生成完成: {bin_id} -> {os.path.basename(filepath)}")
        
        # 更新总数
        self.config["total_generated"] += count
        self._save_config()
        
        print(f"🎉 批量生成完成！共生成 {count} 个二维码")
        print(f"📁 存储位置: {self.active_dir}")
        
        return generated_codes
    
    def get_qr_code(self, bin_id):
        """
        获取（保存）指定垃圾桶的二维码到captured目录
        
        Args:
            bin_id: 垃圾桶编号 (如 "no.5")
            
        Returns:
            str: 保存的文件路径，失败返回None
        """
        if bin_id not in self.config["trash_bins"]:
            print(f"❌ 垃圾桶 {bin_id} 不存在")
            return None
            
        qr_info = self.config["trash_bins"][bin_id]
        
        if qr_info["status"] != "active":
            print(f"❌ 垃圾桶 {bin_id} 的二维码已被作废")
            return None
            
        # 检查原文件是否存在
        if not os.path.exists(qr_info["filepath"]):
            print(f"❌ 原二维码文件不存在: {qr_info['filepath']}")
            return None
            
        # 复制到captured目录
        timestamp = int(time.time())
        captured_filename = f"{bin_id}_captured_{timestamp}.png"
        captured_path = os.path.join(self.captured_dir, captured_filename)
        
        try:
            shutil.copy2(qr_info["filepath"], captured_path)
            print(f"✅ 二维码已保存到: {captured_path}")
            
            # 记录获取历史
            if "capture_history" not in self.config:
                self.config["capture_history"] = []
            
            self.config["capture_history"].append({
                "bin_id": bin_id,
                "captured_time": datetime.now().isoformat(),
                "captured_path": captured_path
            })
            
            self._save_config()
            return captured_path
            
        except Exception as e:
            print(f"❌ 保存二维码失败: {e}")
            return None
    
    def _invalidate_qr(self, bin_id):
        """
        内部方法：作废二维码
        
        Args:
            bin_id: 垃圾桶编号
        """
        if bin_id not in self.config["trash_bins"]:
            return False
            
        qr_info = self.config["trash_bins"][bin_id]
        
        # 移动到archived目录
        if os.path.exists(qr_info["filepath"]):
            archived_filename = f"{bin_id}_invalid_{int(time.time())}.png"
            archived_path = os.path.join(self.archived_dir, archived_filename)
            
            try:
                shutil.move(qr_info["filepath"], archived_path)
                print(f"📦 已作废的二维码移至: {archived_path}")
            except Exception as e:
                print(f"⚠️  移动文件失败: {e}")
        
        # 更新状态
        qr_info["status"] = "invalid"
        qr_info["invalid_time"] = datetime.now().isoformat()
        
        return True
    
    def invalidate_qr_code(self, bin_id):
        """
        作废指定垃圾桶的二维码
        
        Args:
            bin_id: 垃圾桶编号 (如 "no.5")
            
        Returns:
            bool: 操作是否成功
        """
        if bin_id not in self.config["trash_bins"]:
            print(f"❌ 垃圾桶 {bin_id} 不存在")
            return False
            
        qr_info = self.config["trash_bins"][bin_id]
        
        if qr_info["status"] != "active":
            print(f"⚠️  垃圾桶 {bin_id} 的二维码已经是非活跃状态")
            return False
            
        # 执行作废操作
        success = self._invalidate_qr(bin_id)
        
        if success:
            self._save_config()
            print(f"✅ 垃圾桶 {bin_id} 的二维码已作废")
            return True
        else:
            print(f"❌ 作废垃圾桶 {bin_id} 的二维码失败")
            return False
    
    def update_qr_code(self, bin_id):
        """
        更新指定垃圾桶的二维码（旧码自动作废）
        
        Args:
            bin_id: 垃圾桶编号 (如 "no.5")
            
        Returns:
            str: 新二维码文件路径，失败返回None
        """
        print(f"🔄 开始更新垃圾桶 {bin_id} 的二维码...")
        
        # 如果存在旧的活跃二维码，先作废
        if bin_id in self.config["trash_bins"]:
            old_qr = self.config["trash_bins"][bin_id]
            if old_qr["status"] == "active":
                print(f"📦 作废旧二维码...")
                self._invalidate_qr(bin_id)
        
        # 生成新二维码
        unique_id = self._generate_unique_id()
        qr_data = self._create_qr_data(bin_id, unique_id)
        
        try:
            filepath = self._generate_qr_image(qr_data, bin_id)
            
            # 更新配置
            self.config["trash_bins"][bin_id] = {
                "unique_id": unique_id,
                "created_time": qr_data["created_time"],
                "filepath": filepath,
                "status": "active"
            }
            
            self._save_config()
            
            print(f"✅ 垃圾桶 {bin_id} 二维码更新完成")
            print(f"📁 新二维码路径: {filepath}")
            
            return filepath
            
        except Exception as e:
            print(f"❌ 更新二维码失败: {e}")
            return None
    
    def list_qr_codes(self, status_filter=None):
        """
        列出所有二维码信息
        
        Args:
            status_filter: 状态过滤器 ("active", "invalid", None为全部)
            
        Returns:
            list: 二维码信息列表
        """
        qr_list = []
        
        for bin_id, qr_info in self.config["trash_bins"].items():
            if status_filter is None or qr_info["status"] == status_filter:
                qr_list.append({
                    "bin_id": bin_id,
                    "status": qr_info["status"],
                    "created_time": qr_info["created_time"],
                    "filepath": qr_info.get("filepath", ""),
                    "unique_id": qr_info["unique_id"]
                })
        
        return qr_list
    
    def get_statistics(self):
        """
        获取统计信息
        
        Returns:
            dict: 统计信息
        """
        total_bins = len(self.config["trash_bins"])
        active_bins = len([qr for qr in self.config["trash_bins"].values() 
                          if qr["status"] == "active"])
        invalid_bins = total_bins - active_bins
        
        capture_count = len(self.config.get("capture_history", []))
        
        return {
            "total_bins": total_bins,
            "active_bins": active_bins,
            "invalid_bins": invalid_bins,
            "total_generated": self.config.get("total_generated", 0),
            "capture_count": capture_count,
            "last_update": self.config.get("last_update")
        }


def main():
    """主函数：命令行交互界面"""
    qr_manager = QRCodeManager()
    
    print("=" * 50)
    print("🤖 垃圾桶二维码管理系统")
    print("=" * 50)
    
    while True:
        print("\n📋 请选择操作:")
        print("1. 批量生成二维码")
        print("2. 操作单个垃圾桶 (获取/作废/更新)")
        print("3. 查看二维码列表")
        print("4. 查看统计信息")
        print("5. 退出")
        
        choice = input("\n请输入选项 (1-5): ").strip()
        
        if choice == "1":
            # 批量生成
            try:
                count = int(input("请输入生成数量: "))
                qr_manager.generate_qr_codes(count)
            except ValueError:
                print("❌ 请输入有效的数字")
                
        elif choice == "2":
            # 单个操作
            bin_id = input("请输入垃圾桶编号 (如 no.5): ").strip()
            
            print(f"\n对垃圾桶 {bin_id} 可执行的操作:")
            print("1. 获取（保存二维码）")
            print("2. 作废二维码")
            print("3. 更新二维码")
            
            op_choice = input("请选择操作 (1-3): ").strip()
            
            if op_choice == "1":
                qr_manager.get_qr_code(bin_id)
            elif op_choice == "2":
                qr_manager.invalidate_qr_code(bin_id)
            elif op_choice == "3":
                qr_manager.update_qr_code(bin_id)
            else:
                print("❌ 无效选项")
                
        elif choice == "3":
            # 查看列表
            print("\n📋 二维码列表:")
            qr_list = qr_manager.list_qr_codes()
            if qr_list:
                for qr in qr_list:
                    status_emoji = "✅" if qr["status"] == "active" else "❌"
                    print(f"{status_emoji} {qr['bin_id']} - {qr['status']} "
                          f"- 创建时间: {qr['created_time'][:19]}")
            else:
                print("暂无二维码")
                
        elif choice == "4":
            # 统计信息
            stats = qr_manager.get_statistics()
            print("\n📊 统计信息:")
            print(f"总垃圾桶数: {stats['total_bins']}")
            print(f"活跃二维码: {stats['active_bins']}")
            print(f"已作废二维码: {stats['invalid_bins']}")
            print(f"累计生成数: {stats['total_generated']}")
            print(f"累计获取次数: {stats['capture_count']}")
            print(f"最后更新: {stats['last_update'][:19] if stats['last_update'] else '无'}")
            
        elif choice == "5":
            print("👋 感谢使用，再见！")
            break
            
        else:
            print("❌ 无效选项，请重新选择")


# ============================================================================
# 与现有Python文件交互的思路（设计注释，无需实际实现）
# ============================================================================

"""
🔗 与现有垃圾分类系统集成的设计思路：

1. 与 web_realtime.py 集成：
   - 在Web界面添加二维码扫描功能
   - 用户扫描垃圾桶二维码后，系统记录垃圾投放位置信息
   - 实现垃圾分类结果与具体垃圾桶的绑定
   
   集成方式：
   @app.route('/scan_qr')
   def scan_qr():
       # 接收扫描的二维码数据
       # 解析垃圾桶信息
       # 结合当前分类结果记录投放行为
       pass

2. 与 realtime_classifier.py 集成：
   - 在实时分类时，如果检测到二维码，自动识别垃圾桶编号
   - 将分类结果与特定垃圾桶关联
   - 实现智能投放建议（根据垃圾桶类型提醒用户）
   
   集成方式：
   class RealTimeTrashClassifier:
       def detect_qr_in_frame(self, frame):
           # OpenCV + pyzbar 检测二维码
           # 解析垃圾桶信息
           # 返回垃圾桶编号和位置
           pass

3. 数据统计与分析：
   - 记录每个垃圾桶的使用频率
   - 分析不同垃圾桶的分类准确率
   - 生成垃圾投放行为报告
   
   数据结构设计：
   {
       "bin_id": "no.1",
       "usage_logs": [
           {
               "timestamp": "2024-01-01T10:00:00",
               "predicted_class": "plastic",
               "confidence": 0.95,
               "user_confirmed": true
           }
       ]
   }

4. 智能提醒系统：
   - 当用户扫描二维码后，根据垃圾桶类型提供投放建议
   - 如果分类结果与垃圾桶类型不匹配，发出警告
   - 支持用户反馈功能，持续优化模型
   
   API设计：
   /api/smart_suggestion
   {
       "bin_id": "no.1",
       "predicted_class": "paper",
       "bin_type": "plastic_bin",
       "suggestion": "建议投放到纸张垃圾桶"
   }

5. 管理后台集成：
   - 在generate_training_charts.py中添加垃圾桶使用统计图表
   - 在generate_confusion_matrix.py中按垃圾桶分析分类准确率
   - 实现垃圾桶布局优化建议
   
   报告生成：
   def generate_bin_usage_report():
       # 生成垃圾桶使用热力图
       # 分析高频使用时段
       # 提供垃圾桶布局优化建议
       pass

6. 移动端支持：
   - 生成带有垃圾桶位置信息的二维码
   - 支持GPS定位，实现垃圾桶导航功能
   - 离线模式下的二维码识别
   
   二维码数据扩展：
   {
       "bin_id": "no.1",
       "location": {"lat": 39.9042, "lng": 116.4074},
       "bin_type": "recyclable",
       "capacity": "80%",
       "last_maintenance": "2024-01-01"
   }

7. IoT设备集成：
   - 垃圾桶传感器数据集成（满溢检测、重量测量）
   - 自动更新垃圾桶状态信息
   - 预测性维护提醒
   
   设备数据接口：
   class IoTBinManager:
       def update_bin_status(self, bin_id, sensor_data):
           # 更新垃圾桶状态
           # 触发满溢警报
           # 记录使用数据
           pass

通过以上集成，可以实现：
✅ 完整的垃圾投放追踪链路
✅ 智能化的投放建议系统  
✅ 数据驱动的管理优化
✅ 用户行为分析与反馈
✅ 设备状态监控与维护
"""

if __name__ == "__main__":
    main()
