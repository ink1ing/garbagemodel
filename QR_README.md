# 🏷️ 垃圾桶二维码管理系统

## 📋 功能概述

这是一个专为垃圾分类项目设计的二维码管理系统，支持：

- ✅ **批量生成**：一次性生成多个垃圾桶二维码
- ✅ **单独操作**：获取、作废、更新特定垃圾桶的二维码  
- ✅ **分类存储**：自动分类存储活跃、已作废、已获取的二维码
- ✅ **统计分析**：提供详细的使用统计信息
- ✅ **命令行工具**：便捷的批量操作接口

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活虚拟环境
source trash_classifier_env/bin/activate

# 安装依赖（如果尚未安装）
pip install qrcode pillow
```

### 2. 批量生成二维码

```bash
# 生成5个垃圾桶二维码 (no.1 到 no.5)
python qr_tool.py -g 5

# 或者使用完整参数
python qr_tool.py --generate 5
```

### 3. 单独操作垃圾桶

```bash
# 获取no.5的二维码（保存到captured目录）
python qr_tool.py --get no.5

# 作废no.3的二维码
python qr_tool.py --invalidate no.3

# 更新no.1的二维码（旧码自动作废）
python qr_tool.py --update no.1
```

### 4. 查看系统状态

```bash
# 列出所有二维码
python qr_tool.py --list

# 查看统计信息
python qr_tool.py --stats
```

## 📁 目录结构

```
qr_codes/                    # 二维码存储根目录
├── qr_config.json          # 配置文件（记录所有二维码信息）
├── active/                  # 活跃的二维码
│   ├── no.1_qr_xxx.png     
│   ├── no.2_qr_xxx.png
│   └── ...
├── captured/                # 用户获取的二维码
│   ├── no.5_captured_xxx.png
│   └── ...
└── archived/                # 已作废的二维码
    ├── no.3_invalid_xxx.png
    └── ...
```

## 💻 编程接口

### 基本使用

```python
from qr_manager import QRCodeManager

# 初始化管理器
qr_manager = QRCodeManager("my_qr_codes")

# 生成3个二维码
qr_manager.generate_qr_codes(3)

# 获取no.2的二维码
qr_manager.get_qr_code("no.2")

# 作废no.1的二维码
qr_manager.invalidate_qr_code("no.1")

# 更新no.3的二维码
qr_manager.update_qr_code("no.3")

# 查看统计
stats = qr_manager.get_statistics()
print(stats)
```

### 高级功能

```python
# 列出所有活跃的二维码
active_qrs = qr_manager.list_qr_codes(status_filter="active")

# 列出所有已作废的二维码
invalid_qrs = qr_manager.list_qr_codes(status_filter="invalid")

# 获取详细统计信息
stats = qr_manager.get_statistics()
# {
#   "total_bins": 5,
#   "active_bins": 4,
#   "invalid_bins": 1,
#   "total_generated": 8,
#   "capture_count": 2
# }
```

## 🔧 命令行工具选项

```bash
usage: qr_tool.py [-h] [--generate COUNT] [--get BIN_ID] 
                  [--invalidate BIN_ID] [--update BIN_ID] 
                  [--list] [--stats] [--storage-dir STORAGE_DIR]

选项说明:
  -g, --generate COUNT     生成指定数量的二维码
  --get BIN_ID            获取指定垃圾桶的二维码  
  -i, --invalidate BIN_ID 作废指定垃圾桶的二维码
  -u, --update BIN_ID     更新指定垃圾桶的二维码
  -l, --list              列出所有二维码
  -s, --stats             显示统计信息
  --storage-dir DIR       指定存储目录（默认：qr_codes）
```

## 📊 二维码数据格式

每个二维码包含以下JSON数据：

```json
{
  "bin_id": "no.5",
  "unique_id": "f3d69802-4c8e-4b5a-9f2a-8d7e6c5b4a3",
  "created_time": "2025-07-30T20:02:04.123456",
  "type": "trash_bin_qr",
  "status": "active"
}
```

## 🔗 与垃圾分类系统集成

### 设计思路

1. **Web界面集成**
   - 在`web_realtime.py`中添加二维码扫描功能
   - 用户扫描垃圾桶二维码后记录投放位置信息
   - 实现垃圾分类结果与具体垃圾桶的绑定

2. **实时识别集成**  
   - 在`realtime_classifier.py`中添加二维码检测
   - 自动识别画面中的垃圾桶二维码
   - 提供智能投放建议

3. **数据统计分析**
   - 记录每个垃圾桶的使用频率
   - 分析不同垃圾桶的分类准确率
   - 生成垃圾投放行为报告

### 集成示例

```python
# 在web_realtime.py中添加
@app.route('/scan_qr')
def scan_qr():
    qr_data = request.json.get('qr_data')
    bin_info = json.loads(qr_data)
    
    # 记录扫描事件
    log_bin_scan(bin_info['bin_id'])
    
    return jsonify({
        'bin_id': bin_info['bin_id'],
        'suggestions': get_disposal_suggestions(bin_info['bin_id'])
    })
```

## 🧪 测试

运行测试脚本验证功能：

```bash
# 运行完整功能测试
python test_qr_manager.py

# 测试命令行工具
python qr_tool.py -g 3        # 生成3个二维码
python qr_tool.py --list      # 查看列表
python qr_tool.py --stats     # 查看统计
```

## 📝 配置文件说明

`qr_config.json`文件结构：

```json
{
  "trash_bins": {
    "no.1": {
      "unique_id": "uuid-string",
      "created_time": "2025-07-30T20:02:04.123456",
      "filepath": "qr_codes/active/no.1_qr_xxx.png",
      "status": "active"
    }
  },
  "total_generated": 5,
  "capture_history": [
    {
      "bin_id": "no.5",
      "captured_time": "2025-07-30T20:02:29.123456", 
      "captured_path": "qr_codes/captured/no.5_captured_xxx.png"
    }
  ],
  "last_update": "2025-07-30T20:02:29.123456"
}
```

## 🔮 未来扩展

- 🌐 **Web管理界面**：图形化的二维码管理面板
- 📱 **移动端支持**：iOS/Android二维码扫描应用
- 🔄 **批量更新**：一键更新所有过期二维码
- 📊 **使用分析**：二维码扫描热力图和使用统计
- 🔒 **安全增强**：二维码加密和防伪功能

---

**🎯 总结：** 这个二维码管理系统为垃圾分类项目提供了完整的二维码生命周期管理，支持批量操作、状态跟踪、数据统计等功能，可以无缝集成到现有的AI识别系统中，实现垃圾桶与分类结果的精确绑定。
