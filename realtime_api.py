# -*- encoding: utf-8 -*-
"""
实时语音识别 WebSocket API
支持流式音频输入、实时识别、token置信度和声纹识别
"""

import os
import json
import asyncio
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from typing import Dict, Optional
import base64
import logging

from model import SenseVoiceSmall
from speaker_embedding import SpeakerIdentifier
from realtime_stream import RealtimeStreamManager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_FS = 16000

app = FastAPI(title="SenseVoice实时语音识别API")

# 全局模型实例
model_dir = "iic/SenseVoiceSmall"
device = os.getenv("SENSEVOICE_DEVICE", "cuda:0")
logger.info(f"加载模型到设备: {device}")
model, model_kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=device)
model.eval()

# 说话人识别器（全局共享或每个连接一个）
global_speaker_identifier = SpeakerIdentifier(
    similarity_threshold=0.75,
    embedding_dim=256
)


class ConnectionManager:
    """管理WebSocket连接"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.stream_managers: Dict[str, RealtimeStreamManager] = {}
        self.speaker_identifiers: Dict[str, SpeakerIdentifier] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"客户端 {client_id} 已连接")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.stream_managers:
            self.stream_managers[client_id].stop_stream()
            del self.stream_managers[client_id]
        if client_id in self.speaker_identifiers:
            del self.speaker_identifiers[client_id]
        logger.info(f"客户端 {client_id} 已断开")
        
    def get_or_create_stream_manager(self, client_id: str, config: dict) -> RealtimeStreamManager:
        if client_id not in self.stream_managers:
            # 为每个连接创建独立的说话人识别器
            self.speaker_identifiers[client_id] = SpeakerIdentifier(
                similarity_threshold=config.get("speaker_threshold", 0.75),
                embedding_dim=256
            )
            
            # 创建流管理器
            kwargs = {
                **model_kwargs,
                "language": config.get("language", "auto"),
                "use_itn": config.get("use_itn", False),
                "output_confidence": config.get("output_confidence", True),
                "output_speaker_embedding": config.get("output_speaker_embedding", True),
                "speaker_identifier": self.speaker_identifiers[client_id]
            }
            
            self.stream_managers[client_id] = RealtimeStreamManager(
                model=model,
                model_kwargs=kwargs
            )
            
        return self.stream_managers[client_id]


manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="utf-8">
            <title>SenseVoice实时语音识别API</title>
        </head>
        <body>
            <h1>SenseVoice实时语音识别API</h1>
            <ul>
                <li><a href='/docs'>API文档</a></li>
                <li><a href='/realtime-demo'>实时识别演示</a></li>
            </ul>
            <h2>功能特性：</h2>
            <ul>
                <li>实时流式语音识别</li>
                <li>Token级别置信度输出</li>
                <li>说话人声纹识别和唯一标识</li>
                <li>支持多语言识别</li>
            </ul>
        </body>
    </html>
    """


@app.get("/realtime-demo", response_class=HTMLResponse)
async def realtime_demo():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset="utf-8">
            <title>实时语音识别演示</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 1200px; margin: 50px auto; padding: 20px; }
                .container { display: flex; flex-direction: column; gap: 20px; }
                .controls { display: flex; gap: 10px; align-items: center; }
                button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
                .status { padding: 10px; border-radius: 5px; }
                .connected { background-color: #d4edda; color: #155724; }
                .disconnected { background-color: #f8d7da; color: #721c24; }
                .recording { background-color: #fff3cd; color: #856404; }
                .results { border: 1px solid #ddd; padding: 20px; border-radius: 5px; max-height: 500px; overflow-y: auto; }
                .result-item { margin: 10px 0; padding: 10px; border-left: 3px solid #007bff; background: #f8f9fa; }
                .token { display: inline-block; margin: 2px; padding: 2px 5px; background: #e9ecef; border-radius: 3px; }
                .confidence { font-size: 0.8em; color: #666; }
                .speaker-info { color: #28a745; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>实时语音识别演示</h1>
            <div class="container">
                <div class="controls">
                    <button id="connectBtn">连接</button>
                    <button id="startBtn" disabled>开始录音</button>
                    <button id="stopBtn" disabled>停止录音</button>
                    <select id="language">
                        <option value="auto">自动检测</option>
                        <option value="zh">中文</option>
                        <option value="en">英文</option>
                        <option value="yue">粤语</option>
                        <option value="ja">日语</option>
                        <option value="ko">韩语</option>
                    </select>
                    <label><input type="checkbox" id="showConfidence" checked> 显示置信度</label>
                    <label><input type="checkbox" id="showSpeaker" checked> 显示说话人</label>
                </div>
                <div id="status" class="status disconnected">未连接</div>
                <div class="results" id="results"></div>
            </div>
            
            <script>
                let ws = null;
                let mediaRecorder = null;
                let audioContext = null;
                let sessionId = null;
                
                const connectBtn = document.getElementById('connectBtn');
                const startBtn = document.getElementById('startBtn');
                const stopBtn = document.getElementById('stopBtn');
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
                
                function connect() {
                    sessionId = 'session_' + Date.now();
                    ws = new WebSocket(`ws://${window.location.host}/ws/${sessionId}`);
                    
                    ws.onopen = () => {
                        statusDiv.textContent = '已连接';
                        statusDiv.className = 'status connected';
                        connectBtn.textContent = '断开';
                        startBtn.disabled = false;
                        
                        // 发送配置
                        ws.send(JSON.stringify({
                            type: 'config',
                            config: {
                                language: languageSelect.value,
                                output_confidence: showConfidenceCheckbox.checked,
                                output_speaker_embedding: showSpeakerCheckbox.checked,
                                sample_rate: 16000
                            }
                        }));
                    };
                    
                    ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        displayResult(data);
                    };
                    
                    ws.onclose = () => {
                        statusDiv.textContent = '未连接';
                        statusDiv.className = 'status disconnected';
                        connectBtn.textContent = '连接';
                        startBtn.disabled = true;
                        stopBtn.disabled = true;
                        if (mediaRecorder && mediaRecorder.state === 'recording') {
                            stopRecording();
                        }
                    };
                    
                    ws.onerror = (error) => {
                        console.error('WebSocket错误:', error);
                        statusDiv.textContent = '连接错误';
                        statusDiv.className = 'status disconnected';
                    };
                }
                
                async function startRecording() {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                        audioContext = new AudioContext({ sampleRate: 16000 });
                        const source = audioContext.createMediaStreamSource(stream);
                        const processor = audioContext.createScriptProcessor(4096, 1, 1);
                        
                        processor.onaudioprocess = (e) => {
                            if (ws && ws.readyState === WebSocket.OPEN) {
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
                        
                        statusDiv.textContent = '录音中...';
                        statusDiv.className = 'status recording';
                        startBtn.disabled = true;
                        stopBtn.disabled = false;
                        
                        mediaRecorder = { stream, processor, source };
                    } catch (error) {
                        console.error('录音错误:', error);
                        alert('无法访问麦克风');
                    }
                }
                
                function stopRecording() {
                    if (mediaRecorder) {
                        mediaRecorder.processor.disconnect();
                        mediaRecorder.source.disconnect();
                        mediaRecorder.stream.getTracks().forEach(track => track.stop());
                        if (audioContext) {
                            audioContext.close();
                        }
                        mediaRecorder = null;
                        
                        statusDiv.textContent = '已连接';
                        statusDiv.className = 'status connected';
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                    }
                }
                
                function displayResult(data) {
                    if (data.type === 'result') {
                        const resultDiv = document.createElement('div');
                        resultDiv.className = 'result-item';
                        
                        let html = '<div><strong>识别结果:</strong> ' + data.text + '</div>';
                        
                        if (data.speaker_info) {
                            html += '<div class="speaker-info">说话人: ' + data.speaker_info.speaker_id;
                            if (data.speaker_info.similarity !== null) {
                                html += ' (相似度: ' + (data.speaker_info.similarity * 100).toFixed(1) + '%)';
                            }
                            html += '</div>';
                        }
                        
                        if (data.token_confidences && showConfidenceCheckbox.checked) {
                            html += '<div>Token置信度: ';
                            data.token_confidences.forEach(tc => {
                                const confidence = (tc.confidence * 100).toFixed(1);
                                html += `<span class="token">${tc.token} <span class="confidence">${confidence}%</span></span>`;
                            });
                            html += '</div>';
                        }
                        
                        if (data.average_confidence !== undefined) {
                            html += '<div>平均置信度: ' + (data.average_confidence * 100).toFixed(1) + '%</div>';
                        }
                        
                        resultDiv.innerHTML = html;
                        resultsDiv.insertBefore(resultDiv, resultsDiv.firstChild);
                    }
                }
            </script>
        </body>
    </html>
    """


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket端点，用于实时语音识别"""
    await manager.connect(websocket, client_id)
    
    stream_manager = None
    config = {}
    
    try:
        while True:
            # 接收消息
            message = await websocket.receive_text()
            data = json.loads(message)
            
            msg_type = data.get("type")
            
            if msg_type == "config":
                # 配置消息
                config = data.get("config", {})
                stream_manager = manager.get_or_create_stream_manager(client_id, config)
                
                # 启动流处理
                stream_manager.start_stream(
                    sample_rate=config.get("sample_rate", TARGET_FS),
                    chunk_duration_ms=config.get("chunk_duration_ms", 100),
                    vad_threshold=config.get("vad_threshold", 0.5),
                    min_speech_duration_ms=config.get("min_speech_duration_ms", 250)
                )
                
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "status": "ready",
                    "message": "流处理已启动"
                }))
                
            elif msg_type == "audio":
                # 音频数据
                if stream_manager is None:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "请先发送配置消息"
                    }))
                    continue
                
                # 解析音频数据
                audio_data = np.array(data.get("data", []), dtype=np.int16)
                audio_float = audio_data.astype(np.float32) / 32768.0
                
                # 添加到流处理器
                stream_manager.add_audio(audio_float)
                
                # 检查是否有识别结果
                result = stream_manager.get_result(timeout=0.01)
                if result:
                    # 发送结果
                    for res_item in result[0]:
                        await websocket.send_text(json.dumps({
                            "type": "result",
                            **res_item
                        }))
                        
            elif msg_type == "stop":
                # 停止识别
                if stream_manager:
                    stream_manager.stop_stream()
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "status": "stopped",
                    "message": "识别已停止"
                }))
                
            elif msg_type == "reset_speaker":
                # 重置说话人数据库
                if client_id in manager.speaker_identifiers:
                    manager.speaker_identifiers[client_id].clear_database()
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "status": "speaker_reset",
                    "message": "说话人数据库已重置"
                }))
                
    except WebSocketDisconnect:
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
        "active_connections": len(manager.active_connections)
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"启动实时语音识别API服务: {host}:{port}")
    uvicorn.run(app, host=host, port=port)
