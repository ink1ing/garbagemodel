#!/usr/bin/env python3
"""
二维码快速操作工具
提供简化的命令行接口
"""

import sys
import argparse
from qr_manager import QRCodeManager

def main():
    """主函数：解析命令行参数并执行相应操作"""
    parser = argparse.ArgumentParser(description='垃圾桶二维码管理工具')
    parser.add_argument('--generate', '-g', type=int, metavar='COUNT',
                       help='生成指定数量的二维码 (例如: -g 5)')
    parser.add_argument('--get', metavar='BIN_ID',
                       help='获取指定垃圾桶的二维码 (例如: --get no.5)')
    parser.add_argument('--invalidate', '-i', metavar='BIN_ID',
                       help='作废指定垃圾桶的二维码 (例如: -i no.5)')
    parser.add_argument('--update', '-u', metavar='BIN_ID',
                       help='更新指定垃圾桶的二维码 (例如: -u no.5)')
    parser.add_argument('--list', '-l', action='store_true',
                       help='列出所有二维码')
    parser.add_argument('--stats', '-s', action='store_true',
                       help='显示统计信息')
    parser.add_argument('--storage-dir', default='qr_codes',
                       help='二维码存储目录 (默认: qr_codes)')
    
    # 如果没有参数，显示帮助
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()
    
    # 初始化管理器
    qr_manager = QRCodeManager(args.storage_dir)
    
    # 执行相应操作
    if args.generate:
        if args.generate <= 0:
            print("❌ 生成数量必须大于0")
            return
        print(f"🔄 生成 {args.generate} 个二维码...")
        results = qr_manager.generate_qr_codes(args.generate)
        print(f"✅ 成功生成 {len(results)} 个二维码")
        
    elif args.get:
        print(f"💾 获取垃圾桶 {args.get} 的二维码...")
        result = qr_manager.get_qr_code(args.get)
        if result:
            print(f"✅ 已保存到: {result}")
        
    elif args.invalidate:
        print(f"❌ 作废垃圾桶 {args.invalidate} 的二维码...")
        result = qr_manager.invalidate_qr_code(args.invalidate)
        if result:
            print("✅ 作废成功")
            
    elif args.update:
        print(f"🔄 更新垃圾桶 {args.update} 的二维码...")
        result = qr_manager.update_qr_code(args.update)
        if result:
            print(f"✅ 更新成功: {result}")
            
    elif args.list:
        print("📋 所有二维码:")
        qr_list = qr_manager.list_qr_codes()
        if qr_list:
            print(f"{'状态':<6} {'垃圾桶编号':<10} {'创建时间':<20} {'唯一ID':<10}")
            print("-" * 55)
            for qr in qr_list:
                status_emoji = "✅" if qr["status"] == "active" else "❌"
                created_time = qr["created_time"][:19] if qr["created_time"] else "未知"
                unique_id = qr["unique_id"][:8] + "..." if len(qr["unique_id"]) > 8 else qr["unique_id"]
                print(f"{status_emoji:<6} {qr['bin_id']:<10} {created_time:<20} {unique_id:<10}")
        else:
            print("暂无二维码")
            
    elif args.stats:
        print("📊 统计信息:")
        stats = qr_manager.get_statistics()
        print(f"  总垃圾桶数: {stats['total_bins']}")
        print(f"  活跃二维码: {stats['active_bins']}")
        print(f"  已作废二维码: {stats['invalid_bins']}")
        print(f"  累计生成数: {stats['total_generated']}")
        print(f"  累计获取次数: {stats['capture_count']}")
        if stats['last_update']:
            print(f"  最后更新: {stats['last_update'][:19]}")

if __name__ == "__main__":
    main()
