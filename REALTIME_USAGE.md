# SenseVoice 实时语音识别使用指南

## 功能特性

本项目已成功改造为支持实时语音识别，具备以下新功能：

### 🎯 核心功能
- **实时流式识别**：支持连续音频流的实时语音识别
- **置信度输出**：为每个识别的token提供置信度分数
- **声纹识别**：自动识别和区分不同的说话人
- **多语言支持**：支持中文、英文、粤语、日语、韩语等
- **情感识别**：识别语音中的情感状态
- **事件检测**：检测语音中的特殊事件（笑声、掌声等）

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 设置设备（可选）
export SENSEVOICE_DEVICE=cuda:0
```

### 2. 启动服务

```bash
# 使用启动脚本
./start_realtime.sh

# 或手动启动
python3 realtime_demo.py  # 演示界面
python3 api.py           # API服务
python3 realtime_api.py  # 实时API服务
```

## 使用方法

### 1. 基本实时识别

```python
from realtime_asr import RealtimeASR, RecognitionResult

def on_result(result: RecognitionResult):
    print(f"说话人: {result.speaker_id}")
    print(f"置信度: {result.confidence:.3f}")
    print(f"文本: {result.text}")

# 创建ASR实例
asr = RealtimeASR(
    model_dir="iic/SenseVoiceSmall",
    device="cuda:0",
    confidence_threshold=0.6,
    enable_speaker_id=True
)

# 开始识别
asr.start_recording(callback=on_result)

# 停止识别
asr.stop_recording()
```

### 2. WebSocket API

```javascript
// 连接WebSocket
const ws = new WebSocket('ws://localhost:50000/ws/realtime');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    if (data.type === 'result') {
        console.log('识别结果:', data.data);
    }
};
```

### 3. HTTP API

```bash
# 初始化实时ASR
curl -X POST "http://localhost:50000/api/v1/asr/realtime/init" \
     -H "Content-Type: application/json" \
     -d '{"confidence_threshold": 0.6, "enable_speaker_id": true}'

# 开始识别
curl -X POST "http://localhost:50000/api/v1/asr/realtime/start"

# 获取结果
curl -X GET "http://localhost:50000/api/v1/asr/realtime/results"

# 停止识别
curl -X POST "http://localhost:50000/api/v1/asr/realtime/stop"
```

## 配置参数

### RealtimeASR 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_dir` | str | "iic/SenseVoiceSmall" | 模型路径 |
| `device` | str | "cuda:0" | 计算设备 |
| `sample_rate` | int | 16000 | 采样率 |
| `chunk_duration` | float | 1.0 | 音频块时长（秒） |
| `overlap_duration` | float | 0.5 | 重叠时长（秒） |
| `confidence_threshold` | float | 0.5 | 置信度阈值 |
| `enable_speaker_id` | bool | True | 是否启用声纹识别 |

### 识别结果结构

```python
@dataclass
class RecognitionResult:
    text: str                    # 识别文本
    confidence: float           # 置信度 (0-1)
    speaker_id: Optional[str]   # 说话人ID
    timestamp: float            # 时间戳
    language: str               # 语言
    emotion: Optional[str]      # 情感
    event: Optional[str]        # 事件
```

## 服务端口

- **API服务**: http://localhost:50000
- **实时API**: http://localhost:8000
- **演示界面**: http://localhost:7860

## 测试

```bash
# 运行完整测试
python3 test_realtime.py

# 测试特定功能
python3 -c "from realtime_asr import SpeakerIdentifier; print('声纹识别测试通过')"
```

## 注意事项

1. **设备要求**: 建议使用GPU加速，CPU模式可能较慢
2. **音频格式**: 支持16kHz采样率的音频
3. **内存使用**: 实时识别会占用较多内存，建议8GB以上
4. **网络延迟**: WebSocket连接可能有网络延迟影响

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查网络连接
   - 确认模型路径正确
   - 检查设备可用性

2. **识别效果差**
   - 调整置信度阈值
   - 检查音频质量
   - 确认语言设置

3. **声纹识别不准确**
   - 增加音频时长
   - 调整相似度阈值
   - 确保音频清晰

### 日志调试

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 性能优化

1. **减少延迟**: 调整 `chunk_duration` 参数
2. **提高精度**: 增加 `confidence_threshold`
3. **节省内存**: 禁用声纹识别或减少音频缓冲区

## 扩展功能

项目支持以下扩展：

- 自定义声纹识别模型
- 多语言实时切换
- 实时情感分析
- 音频质量检测
- 说话人聚类分析

## 技术支持

如有问题，请查看：
- 项目文档: README.md
- 测试用例: test_realtime.py
- 演示代码: realtime_demo.py