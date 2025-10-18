# -*- encoding: utf-8 -*-
"""
流式语音识别 WebSocket API
支持真正的实时流式识别，边说边出结果
"""

import os
import json
import asyncio
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from typing import Dict, Optional
import logging

from model import SenseVoiceSmall
from speaker_embedding import SpeakerIdentifier
from streaming_asr import StreamingASRManager, ChunkedStreamProcessor
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_FS = 16000

app = FastAPI(title="SenseVoice流式语音识别API")

# 全局模型实例
model_dir = "iic/SenseVoiceSmall"
device = os.getenv("SENSEVOICE_DEVICE", "cuda:0")
logger.info(f"加载模型到设备: {device}")
model, model_kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=device)
model.eval()

# 流式ASR管理器
streaming_manager = StreamingASRManager(model, model_kwargs)


class StreamingConnectionManager:
    """管理WebSocket连接和流式识别会话"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.stream_processors: Dict[str, ChunkedStreamProcessor] = {}
        self.speaker_identifiers: Dict[str, SpeakerIdentifier] = {}
        self.result_queues: Dict[str, asyncio.Queue] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.result_queues[client_id] = asyncio.Queue()
        logger.info(f"客户端 {client_id} 已连接")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.stream_processors:
            del self.stream_processors[client_id]
        if client_id in self.speaker_identifiers:
            del self.speaker_identifiers[client_id]
        if client_id in self.result_queues:
            del self.result_queues[client_id]
        streaming_manager.close_session(client_id)
        logger.info(f"客户端 {client_id} 已断开")
        
    def create_stream_processor(self, client_id: str, config: dict):
        """创建流式处理器"""
        # 创建说话人识别器
        self.speaker_identifiers[client_id] = SpeakerIdentifier(
            similarity_threshold=config.get("speaker_threshold", 0.75),
            embedding_dim=256
        )
        
        # 准备模型参数
        kwargs = {
            **model_kwargs,
            "language": config.get("language", "auto"),
            "use_itn": config.get("use_itn", False),
            "output_confidence": config.get("output_confidence", True),
            "output_speaker_embedding": config.get("output_speaker_embedding", True),
            "speaker_identifier": self.speaker_identifiers[client_id]
        }
        
        # 创建回调函数
        async def result_callback(result):
            if client_id in self.result_queues:
                await self.result_queues[client_id].put(result)
        
        # 同步回调包装
        def sync_callback(result):
            try:
                loop = asyncio.get_event_loop()
                asyncio.run_coroutine_threadsafe(result_callback(result), loop)
            except Exception as e:
                logger.error(f"回调错误: {e}")
        
        # 创建流式识别器
        recognizer = streaming_manager.create_session(
            session_id=client_id,
            callback=sync_callback,
            sample_rate=config.get("sample_rate", TARGET_FS),
            chunk_size=config.get("chunk_size", 960),  # 60ms
            stride_size=config.get("stride_size", 640),  # 40ms
            min_chunk_threshold=config.get("min_chunk_threshold", 10)
        )
        
        # 创建块处理器
        self.stream_processors[client_id] = ChunkedStreamProcessor(
            recognizer=recognizer,
            chunk_duration_ms=config.get("chunk_duration_ms", 100),
            sample_rate=config.get("sample_rate", TARGET_FS)
        )


manager = StreamingConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="utf-8">
            <title>SenseVoice流式语音识别</title>
        </head>
        <body>
            <h1>🎤 SenseVoice流式语音识别API</h1>
            <ul>
                <li><a href='/docs'>📚 API文档</a></li>
                <li><a href='/streaming-demo'>🎬 流式识别演示</a></li>
            </ul>
            <h2>✨ 核心特性：</h2>
            <ul>
                <li>🚀 真正的实时流式识别（边说边出结果）</li>
                <li>📊 Token级别置信度</li>
                <li>👤 说话人声纹识别</li>
                <li>🌍 多语言支持</li>
                <li>⚡ 超低延迟（< 200ms）</li>
            </ul>
        </body>
    </html>
    """


@app.get("/streaming-demo", response_class=HTMLResponse)
async def streaming_demo():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="utf-8">
            <title>流式语音识别演示</title>
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }
                .container { 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background: white;
                    border-radius: 20px;
                    padding: 30px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                }
                h1 { 
                    color: #667eea; 
                    margin-bottom: 30px;
                    text-align: center;
                    font-size: 2.5em;
                }
                .controls { 
                    display: flex; 
                    gap: 15px; 
                    align-items: center; 
                    margin-bottom: 30px;
                    flex-wrap: wrap;
                    justify-content: center;
                }
                button { 
                    padding: 12px 24px; 
                    font-size: 16px; 
                    cursor: pointer;
                    border: none;
                    border-radius: 10px;
                    transition: all 0.3s;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }
                button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.2); }
                button:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
                
                #connectBtn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
                #startBtn { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; }
                #stopBtn { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); color: white; }
                #clearBtn { background: linear-gradient(135deg, #30cfd0 0%, #330867 100%); color: white; }
                
                select, input[type="checkbox"] { 
                    padding: 10px; 
                    font-size: 16px;
                    border: 2px solid #667eea;
                    border-radius: 8px;
                }
                label { 
                    display: flex; 
                    align-items: center; 
                    gap: 8px;
                    font-weight: 500;
                }
                
                .status { 
                    padding: 15px; 
                    border-radius: 10px; 
                    margin-bottom: 20px;
                    font-weight: 600;
                    text-align: center;
                    font-size: 18px;
                }
                .connected { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #155724; }
                .disconnected { background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); color: #721c24; }
                .recording { 
                    background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); 
                    color: #856404;
                    animation: pulse 1.5s infinite;
                }
                
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.7; }
                }
                
                .results { 
                    border: 2px solid #667eea; 
                    padding: 20px; 
                    border-radius: 15px; 
                    max-height: 500px; 
                    overflow-y: auto;
                    background: #f8f9fa;
                }
                .result-item { 
                    margin: 15px 0; 
                    padding: 15px; 
                    border-left: 4px solid #667eea; 
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    animation: slideIn 0.3s ease-out;
                }
                
                @keyframes slideIn {
                    from { opacity: 0; transform: translateX(-20px); }
                    to { opacity: 1; transform: translateX(0); }
                }
                
                .partial-result {
                    border-left-color: #ffc107;
                    background: #fffef7;
                }
                .final-result {
                    border-left-color: #28a745;
                    background: #f0fff4;
                }
                
                .result-text {
                    font-size: 18px;
                    margin-bottom: 10px;
                    line-height: 1.6;
                }
                .result-meta {
                    display: flex;
                    gap: 15px;
                    flex-wrap: wrap;
                    font-size: 14px;
                    color: #666;
                }
                .meta-item {
                    display: flex;
                    align-items: center;
                    gap: 5px;
                }
                .badge {
                    padding: 4px 8px;
                    border-radius: 5px;
                    font-size: 12px;
                    font-weight: 600;
                }
                .badge-partial { background: #fff3cd; color: #856404; }
                .badge-final { background: #d4edda; color: #155724; }
                .speaker-info { color: #667eea; font-weight: 600; }
                .confidence { color: #28a745; font-weight: 600; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🎤 实时流式语音识别</h1>
                
                <div class="controls">
                    <button id="connectBtn">🔌 连接</button>
                    <button id="startBtn" disabled>🎙️ 开始录音</button>
                    <button id="stopBtn" disabled>⏹️ 停止录音</button>
                    <button id="clearBtn">🗑️ 清空结果</button>
                    <select id="language">
                        <option value="auto">🌐 自动检测</option>
                        <option value="zh">🇨🇳 中文</option>
                        <option value="en">🇺🇸 英文</option>
                        <option value="yue">🇭🇰 粤语</option>
                        <option value="ja">🇯🇵 日语</option>
                        <option value="ko">🇰🇷 韩语</option>
                    </select>
                    <label>
                        <input type="checkbox" id="showConfidence" checked> 
                        显示置信度
                    </label>
                    <label>
                        <input type="checkbox" id="showSpeaker" checked> 
                        显示说话人
                    </label>
                </div>
                
                <div id="status" class="status disconnected">🔴 未连接</div>
                
                <div class="results" id="results">
                    <p style="text-align: center; color: #999;">识别结果将显示在这里...</p>
                </div>
            </div>
            
            <script>
                let ws = null;
                let mediaRecorder = null;
                let audioContext = null;
                let sessionId = null;
                let isRecording = false;
                
                const connectBtn = document.getElementById('connectBtn');
                const startBtn = document.getElementById('startBtn');
                const stopBtn = document.getElementById('stopBtn');
                const clearBtn = document.getElementById('clearBtn');
                const statusDiv = document.getElementById('status');
                const resultsDiv = document.getElementById('results');
                const languageSelect = document.getElementById('language');
                const showConfidenceCheckbox = document.getElementById('showConfidence');
                const showSpeakerCheckbox = document.getElementById('showSpeaker');
                
                connectBtn.addEventListener('click', () => {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.close();
                    } else {
                        connect();
                    }
                });
                
                startBtn.addEventListener('click', startRecording);
                stopBtn.addEventListener('click', stopRecording);
                clearBtn.addEventListener('click', clearResults);
                
                function connect() {
                    sessionId = 'session_' + Date.now();
                    ws = new WebSocket(`ws://${window.location.host}/ws/streaming/${sessionId}`);
                    
                    ws.onopen = () => {
                        statusDiv.textContent = '🟢 已连接';
                        statusDiv.className = 'status connected';
                        connectBtn.textContent = '🔌 断开';
                        startBtn.disabled = false;
                        
                        // 发送配置
                        ws.send(JSON.stringify({
                            type: 'config',
                            config: {
                                language: languageSelect.value,
                                output_confidence: showConfidenceCheckbox.checked,
                                output_speaker_embedding: showSpeakerCheckbox.checked,
                                sample_rate: 16000,
                                chunk_size: 960,
                                stride_size: 640,
                                min_chunk_threshold: 8
                            }
                        }));
                    };
                    
                    ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        if (data.type === 'result') {
                            displayResult(data);
                        } else if (data.type === 'status') {
                            console.log('状态:', data.message);
                        }
                    };
                    
                    ws.onclose = () => {
                        statusDiv.textContent = '🔴 未连接';
                        statusDiv.className = 'status disconnected';
                        connectBtn.textContent = '🔌 连接';
                        startBtn.disabled = true;
                        stopBtn.disabled = true;
                        if (isRecording) {
                            stopRecording();
                        }
                    };
                    
                    ws.onerror = (error) => {
                        console.error('WebSocket错误:', error);
                        statusDiv.textContent = '❌ 连接错误';
                        statusDiv.className = 'status disconnected';
                    };
                }
                
                async function startRecording() {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ 
                            audio: {
                                channelCount: 1,
                                sampleRate: 16000,
                                echoCancellation: true,
                                noiseSuppression: true,
                                autoGainControl: true
                            } 
                        });
                        
                        audioContext = new AudioContext({ sampleRate: 16000 });
                        const source = audioContext.createMediaStreamSource(stream);
                        const processor = audioContext.createScriptProcessor(4096, 1, 1);
                        
                        processor.onaudioprocess = (e) => {
                            if (ws && ws.readyState === WebSocket.OPEN && isRecording) {
                                const audioData = e.inputBuffer.getChannelData(0);
                                const int16Array = new Int16Array(audioData.length);
                                for (let i = 0; i < audioData.length; i++) {
                                    int16Array[i] = Math.max(-32768, Math.min(32767, audioData[i] * 32768));
                                }
                                ws.send(JSON.stringify({
                                    type: 'audio',
                                    data: Array.from(int16Array)
                                }));
                            }
                        };
                        
                        source.connect(processor);
                        processor.connect(audioContext.destination);
                        
                        isRecording = true;
                        statusDiv.textContent = '🔴 录音中... 边说边出结果';
                        statusDiv.className = 'status recording';
                        startBtn.disabled = true;
                        stopBtn.disabled = false;
                        
                        mediaRecorder = { stream, processor, source };
                    } catch (error) {
                        console.error('录音错误:', error);
                        alert('无法访问麦克风：' + error.message);
                    }
                }
                
                function stopRecording() {
                    if (mediaRecorder) {
                        isRecording = false;
                        mediaRecorder.processor.disconnect();
                        mediaRecorder.source.disconnect();
                        mediaRecorder.stream.getTracks().forEach(track => track.stop());
                        if (audioContext) {
                            audioContext.close();
                        }
                        mediaRecorder = null;
                        
                        // 发送结束信号
                        if (ws && ws.readyState === WebSocket.OPEN) {
                            ws.send(JSON.stringify({ type: 'finalize' }));
                        }
                        
                        statusDiv.textContent = '🟢 已连接';
                        statusDiv.className = 'status connected';
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                    }
                }
                
                function clearResults() {
                    resultsDiv.innerHTML = '<p style="text-align: center; color: #999;">识别结果将显示在这里...</p>';
                }
                
                let lastResultElement = null;
                let lastChunkId = -1;
                
                function displayResult(data) {
                    const isFinal = data.is_final;
                    const chunkId = data.chunk_id || 0;
                    
                    // 清理首次显示的提示文本
                    if (resultsDiv.children.length === 1 && resultsDiv.children[0].tagName === 'P') {
                        resultsDiv.innerHTML = '';
                    }
                    
                    // 如果是同一个chunk的更新，替换之前的结果
                    if (!isFinal && lastChunkId === chunkId && lastResultElement) {
                        lastResultElement.remove();
                    }
                    
                    const resultDiv = document.createElement('div');
                    resultDiv.className = 'result-item ' + (isFinal ? 'final-result' : 'partial-result');
                    
                    // 处理文本
                    let displayText = data.text || '';
                    try {
                        displayText = displayText.replace(/<\|[^|]+\|>/g, '').trim();
                    } catch (e) {
                        console.error('文本处理错误:', e);
                    }
                    
                    let html = '<div class="result-text">';
                    html += '<span class="badge ' + (isFinal ? 'badge-final' : 'badge-partial') + '">';
                    html += isFinal ? '✓ 最终' : '⏳ 识别中';
                    html += '</span> ';
                    html += displayText;
                    html += '</div>';
                    
                    html += '<div class="result-meta">';
                    
                    if (data.speaker_info && showSpeakerCheckbox.checked) {
                        html += '<div class="meta-item speaker-info">';
                        html += '👤 ' + data.speaker_info.speaker_id;
                        if (data.speaker_info.similarity !== null && data.speaker_info.similarity !== undefined) {
                            html += ' (' + (data.speaker_info.similarity * 100).toFixed(1) + '%)';
                        }
                        html += '</div>';
                    }
                    
                    if (data.average_confidence !== undefined && showConfidenceCheckbox.checked) {
                        html += '<div class="meta-item confidence">';
                        html += '📊 置信度: ' + (data.average_confidence * 100).toFixed(1) + '%';
                        html += '</div>';
                    }
                    
                    html += '<div class="meta-item">🕒 ' + new Date().toLocaleTimeString() + '</div>';
                    html += '</div>';
                    
                    resultDiv.innerHTML = html;
                    resultsDiv.insertBefore(resultDiv, resultsDiv.firstChild);
                    
                    if (!isFinal) {
                        lastResultElement = resultDiv;
                        lastChunkId = chunkId;
                    } else {
                        lastResultElement = null;
                        lastChunkId = -1;
                    }
                }
            </script>
        </body>
    </html>
    """


@app.websocket("/ws/streaming/{client_id}")
async def websocket_streaming_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket端点，用于流式语音识别"""
    await manager.connect(websocket, client_id)
    
    config = {}
    
    try:
        # 创建结果发送任务
        async def send_results():
            while client_id in manager.result_queues:
                try:
                    result = await asyncio.wait_for(
                        manager.result_queues[client_id].get(),
                        timeout=0.1
                    )
                    
                    # 处理文本
                    if 'text' in result:
                        result['text'] = rich_transcription_postprocess(result['text'])
                    
                    await websocket.send_text(json.dumps({
                        "type": "result",
                        **result
                    }))
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"发送结果错误: {e}")
                    break
        
        # 启动结果发送任务
        send_task = asyncio.create_task(send_results())
        
        while True:
            # 接收消息
            message = await websocket.receive_text()
            data = json.loads(message)
            
            msg_type = data.get("type")
            
            if msg_type == "config":
                # 配置消息
                config = data.get("config", {})
                manager.create_stream_processor(client_id, config)
                
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "status": "ready",
                    "message": "流式识别已启动"
                }))
                
            elif msg_type == "audio":
                # 音频数据
                if client_id not in manager.stream_processors:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "请先发送配置消息"
                    }))
                    continue
                
                # 解析音频数据
                audio_data = np.array(data.get("data", []), dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32768.0
                
                # 处理音频
                manager.stream_processors[client_id].process_audio(audio_float)
                
            elif msg_type == "finalize":
                # 完成识别
                if client_id in manager.stream_processors:
                    final_result = manager.stream_processors[client_id].flush()
                    if final_result:
                        final_result['text'] = rich_transcription_postprocess(final_result['text'])
                        await websocket.send_text(json.dumps({
                            "type": "result",
                            **final_result
                        }))
                
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "status": "finalized",
                    "message": "识别已完成"
                }))
                
            elif msg_type == "reset":
                # 重置识别
                if client_id in manager.stream_processors:
                    manager.stream_processors[client_id].reset()
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "status": "reset",
                    "message": "已重置"
                }))
                
    except WebSocketDisconnect:
        send_task.cancel()
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket错误: {e}", exc_info=True)
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e)
            }))
        except:
            pass
        manager.disconnect(client_id)


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model": model_dir,
        "device": device,
        "active_connections": len(manager.active_connections),
        "mode": "streaming"
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8001))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"启动流式语音识别API服务: {host}:{port}")
    logger.info("特性: 真正的实时流式识别，边说边出结果")
    uvicorn.run(app, host=host, port=port)
