#!/usr/bin/env python3
"""
二维码管理系统测试脚本
快速测试二维码生成、管理功能
"""

from qr_manager import QRCodeManager
import os

def test_qr_manager():
    """测试二维码管理系统的主要功能"""
    print("🧪 开始测试二维码管理系统...")
    
    # 1. 初始化管理器
    qr_manager = QRCodeManager("test_qr_codes")
    print("✅ 管理器初始化完成")
    
    # 2. 测试批量生成
    print("\n📝 测试批量生成 3 个二维码...")
    generated = qr_manager.generate_qr_codes(3)
    print(f"生成结果: {len(generated)} 个二维码")
    
    # 3. 测试获取二维码
    print("\n💾 测试获取 no.2 的二维码...")
    captured_path = qr_manager.get_qr_code("no.2")
    if captured_path:
        print(f"保存成功: {captured_path}")
    
    # 4. 测试作废二维码
    print("\n❌ 测试作废 no.3 的二维码...")
    success = qr_manager.invalidate_qr_code("no.3")
    print(f"作废结果: {'成功' if success else '失败'}")
    
    # 5. 测试更新二维码
    print("\n🔄 测试更新 no.1 的二维码...")
    new_path = qr_manager.update_qr_code("no.1")
    if new_path:
        print(f"更新成功: {new_path}")
    
    # 6. 查看统计信息
    print("\n📊 统计信息:")
    stats = qr_manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 7. 列出所有二维码
    print("\n📋 所有二维码:")
    qr_list = qr_manager.list_qr_codes()
    for qr in qr_list:
        status_emoji = "✅" if qr["status"] == "active" else "❌"
        print(f"  {status_emoji} {qr['bin_id']} - {qr['status']}")
    
    print("\n🎉 测试完成！")
    print(f"📁 查看生成的文件: {qr_manager.storage_dir}")

if __name__ == "__main__":
    test_qr_manager()
