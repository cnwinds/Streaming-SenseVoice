# SenseVoice 实时语音识别增强版

## 🎯 新增功能

本项目在原有 SenseVoice 基础上增加了以下强大功能：

### 1. **实时流式语音识别**
- 支持音频流实时处理
- 基于 VAD（语音活动检测）的智能分段
- WebSocket 实时通信
- 低延迟响应

### 2. **Token 级别置信度输出**
- 每个识别的 token 都附带置信度分数
- 平均置信度计算
- 帮助评估识别质量

### 3. **说话人声纹识别**
- 自动提取说话人特征嵌入向量
- 说话人唯一标识生成
- 说话人相似度匹配
- 多说话人自动区分

## 📦 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 如果使用 GPU
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 快速开始

### 方法一：使用 WebSocket 实时 API

1. **启动实时识别服务**

```bash
# 使用 GPU
export SENSEVOICE_DEVICE=cuda:0
python realtime_api.py

# 使用 CPU
export SENSEVOICE_DEVICE=cpu
python realtime_api.py
```

2. **访问 Web 演示界面**

打开浏览器访问：`http://localhost:8000/realtime-demo`

3. **API 文档**

访问：`http://localhost:8000/docs` 查看完整 API 文档

### 方法二：使用命令行演示程序

#### 简单模式（非流式）

```bash
python demo_realtime.py \
    --audio_file path/to/your/audio.wav \
    --mode simple \
    --language auto
```

#### 实时流式模式

```bash
python demo_realtime.py \
    --audio_file path/to/your/audio.wav \
    --mode realtime \
    --language auto
```

### 方法三：在代码中使用

```python
from model import SenseVoiceSmall
from speaker_embedding import SpeakerIdentifier

# 加载模型
model, kwargs = SenseVoiceSmall.from_pretrained(
    model="iic/SenseVoiceSmall",
    device="cuda:0"
)
model.eval()

# 创建说话人识别器
speaker_identifier = SpeakerIdentifier(
    similarity_threshold=0.75,
    embedding_dim=256
)

# 进行识别
result = model.inference(
    data_in="path/to/audio.wav",
    language="auto",
    use_itn=True,
    output_confidence=True,              # 输出置信度
    output_speaker_embedding=True,       # 输出声纹
    speaker_identifier=speaker_identifier,  # 说话人识别
    **kwargs
)

# 获取结果
if result and result[0]:
    res = result[0][0]
    
    # 识别文本
    print(f"文本: {res['text']}")
    
    # Token 置信度
    if 'token_confidences' in res:
        for tc in res['token_confidences']:
            print(f"Token: {tc['token']}, 置信度: {tc['confidence']:.2%}")
    
    # 说话人信息
    if 'speaker_info' in res:
        print(f"说话人ID: {res['speaker_info']['speaker_id']}")
        if res['speaker_info']['similarity']:
            print(f"相似度: {res['speaker_info']['similarity']:.2%}")
```

## 🔧 核心模块说明

### 1. realtime_stream.py - 实时流处理模块

提供了音频流缓冲、VAD 检测和实时处理能力。

**主要类：**
- `AudioStreamBuffer`: 音频流缓冲区
- `RealtimeASRProcessor`: 实时语音识别处理器
- `RealtimeStreamManager`: 流管理器

### 2. speaker_embedding.py - 声纹识别模块

提供说话人特征提取和识别功能。

**主要类：**
- `SpeakerEmbeddingExtractor`: 说话人嵌入提取器
- `SpeakerIdentifier`: 说话人识别器
- `StatisticsPooling`: 统计池化层

### 3. realtime_api.py - WebSocket API

基于 FastAPI 和 WebSocket 的实时识别 API。

**主要端点：**
- `GET /`: 主页
- `GET /realtime-demo`: 实时识别演示页面
- `WebSocket /ws/{client_id}`: WebSocket 连接端点
- `GET /health`: 健康检查

### 4. model.py - 模型增强

在原有 SenseVoiceSmall 模型基础上增加了：
- Token 置信度计算
- 说话人嵌入提取接口
- 增强的 inference 方法

## 📊 输出格式

### 识别结果示例

```json
{
  "key": "audio_001",
  "text": "<|zh|><|Speech|><|NEUTRAL|>你好世界",
  "token_confidences": [
    {"token": "你", "confidence": 0.95},
    {"token": "好", "confidence": 0.93},
    {"token": "世", "confidence": 0.97},
    {"token": "界", "confidence": 0.96}
  ],
  "average_confidence": 0.9525,
  "speaker_info": {
    "speaker_id": "speaker_0001_a3f5e7b92c4d",
    "similarity": 0.89
  },
  "speaker_embedding": [0.123, -0.456, 0.789, ...]
}
```

## 🎛️ 配置参数

### 模型推理参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `language` | str | "auto" | 语言：auto/zh/en/yue/ja/ko |
| `use_itn` | bool | False | 是否使用逆文本正则化 |
| `output_confidence` | bool | False | 是否输出置信度 |
| `output_speaker_embedding` | bool | False | 是否输出说话人嵌入 |
| `output_timestamp` | bool | False | 是否输出时间戳 |
| `speaker_identifier` | SpeakerIdentifier | None | 说话人识别器实例 |

### 实时流处理参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sample_rate` | int | 16000 | 采样率（Hz） |
| `chunk_duration_ms` | int | 100 | 音频块时长（毫秒） |
| `vad_threshold` | float | 0.5 | VAD 阈值 |
| `min_speech_duration_ms` | int | 250 | 最小语音时长（毫秒） |

### 说话人识别参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `similarity_threshold` | float | 0.75 | 相似度阈值（0-1） |
| `embedding_dim` | int | 256 | 嵌入向量维度 |

## 🔌 WebSocket API 使用示例

### Python 客户端

```python
import asyncio
import websockets
import json
import numpy as np

async def realtime_recognize():
    uri = "ws://localhost:8000/ws/test_client"
    
    async with websockets.connect(uri) as websocket:
        # 发送配置
        await websocket.send(json.dumps({
            "type": "config",
            "config": {
                "language": "zh",
                "output_confidence": True,
                "output_speaker_embedding": True,
                "sample_rate": 16000
            }
        }))
        
        # 读取音频文件
        audio_data = load_audio("test.wav")  # 你的音频加载函数
        
        # 分块发送音频
        chunk_size = 1600
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            
            await websocket.send(json.dumps({
                "type": "audio",
                "data": chunk.tolist()
            }))
            
            # 接收结果
            try:
                response = await asyncio.wait_for(
                    websocket.recv(), 
                    timeout=0.1
                )
                result = json.loads(response)
                if result.get("type") == "result":
                    print(f"识别结果: {result['text']}")
            except asyncio.TimeoutError:
                pass

asyncio.run(realtime_recognize())
```

### JavaScript 客户端

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/client_123');

// 发送配置
ws.onopen = () => {
    ws.send(JSON.stringify({
        type: 'config',
        config: {
            language: 'auto',
            output_confidence: true,
            output_speaker_embedding: true
        }
    }));
};

// 接收结果
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'result') {
        console.log('识别结果:', data.text);
        console.log('说话人:', data.speaker_info);
        console.log('置信度:', data.average_confidence);
    }
};

// 发送音频数据
function sendAudio(audioData) {
    ws.send(JSON.stringify({
        type: 'audio',
        data: Array.from(audioData)
    }));
}
```

## 🌟 高级功能

### 1. 多说话人识别

```python
from speaker_embedding import SpeakerIdentifier

# 创建说话人识别器
speaker_id = SpeakerIdentifier(similarity_threshold=0.75)

# 处理多个音频
for audio_file in audio_files:
    result = model.inference(
        data_in=audio_file,
        output_speaker_embedding=True,
        speaker_identifier=speaker_id,
        **kwargs
    )
    
    speaker_info = result[0][0]['speaker_info']
    print(f"说话人: {speaker_info['speaker_id']}")

# 获取所有检测到的说话人
all_speakers = speaker_id.get_all_speakers()
print(f"共检测到 {len(all_speakers)} 个不同的说话人")
```

### 2. 保存和加载说话人数据库

```python
# 保存
speaker_identifier.save_database("speakers.pkl")

# 加载
speaker_identifier.load_database("speakers.pkl")
```

### 3. 置信度过滤

```python
result = model.inference(
    data_in=audio_file,
    output_confidence=True,
    **kwargs
)

# 过滤低置信度的 token
high_conf_tokens = [
    tc for tc in result[0][0]['token_confidences']
    if tc['confidence'] > 0.8
]
```

## 🐛 常见问题

### Q: WebSocket 连接失败？
A: 检查防火墙设置，确保端口 8000 开放。尝试使用 `0.0.0.0` 作为 host。

### Q: 识别结果中没有置信度？
A: 确保在调用时设置了 `output_confidence=True`。

### Q: 说话人识别不准确？
A: 尝试调整 `similarity_threshold` 参数，或增加训练数据。

### Q: GPU 内存不足？
A: 尝试使用 CPU 模式，或减小 batch_size。

## 📝 注意事项

1. **音频格式**：支持 16kHz 采样率，推荐单声道
2. **实时性**：WebSocket 模式延迟约 100-300ms
3. **声纹识别**：需要至少 1 秒以上的音频才能准确提取声纹
4. **并发限制**：单个服务实例建议不超过 10 个并发连接

## 📄 许可证

本项目遵循原 SenseVoice 项目的许可证。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题，请通过 GitHub Issues 联系我们。

---

**享受强大的实时语音识别体验！** 🎤✨
