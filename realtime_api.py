#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# 实时语音识别API接口

import asyncio
import json
import base64
import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import io
import wave

from realtime_asr import RealtimeASR, RecognitionResult

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="实时语音识别API", version="1.0.0")

# 全局ASR实例
asr_instance: Optional[RealtimeASR] = None

class ASRConfig(BaseModel):
    """ASR配置"""
    model_dir: str = "iic/SenseVoiceSmall"
    device: str = "cuda:0"
    sample_rate: int = 16000
    chunk_duration: float = 1.0
    overlap_duration: float = 0.5
    confidence_threshold: float = 0.5
    enable_speaker_id: bool = True
    language: str = "auto"

class RecognitionResponse(BaseModel):
    """识别响应"""
    text: str
    confidence: float
    speaker_id: Optional[str] = None
    timestamp: float
    language: str
    emotion: Optional[str] = None
    event: Optional[str] = None
    tokens: Optional[list] = None

class ConnectionManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.asr_instances: Dict[str, RealtimeASR] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"客户端 {client_id} 已连接")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.asr_instances:
            self.asr_instances[client_id].stop_recording()
            del self.asr_instances[client_id]
        logger.info(f"客户端 {client_id} 已断开")
    
    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except Exception as e:
                logger.error(f"发送消息失败: {e}")
    
    def get_asr_instance(self, client_id: str) -> Optional[RealtimeASR]:
        return self.asr_instances.get(client_id)
    
    def set_asr_instance(self, client_id: str, asr: RealtimeASR):
        self.asr_instances[client_id] = asr

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """返回Web界面"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>实时语音识别</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .controls { margin: 20px 0; }
            button { padding: 10px 20px; margin: 5px; font-size: 16px; }
            .start { background-color: #4CAF50; color: white; }
            .stop { background-color: #f44336; color: white; }
            .results { border: 1px solid #ccc; padding: 20px; margin: 20px 0; min-height: 200px; }
            .result-item { margin: 10px 0; padding: 10px; background-color: #f9f9f9; }
            .speaker { font-weight: bold; color: #2196F3; }
            .confidence { color: #666; font-size: 12px; }
            .text { margin: 5px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>实时语音识别系统</h1>
            <div class="controls">
                <button id="startBtn" class="start" onclick="startRecording()">开始录音</button>
                <button id="stopBtn" class="stop" onclick="stopRecording()" disabled>停止录音</button>
                <button onclick="clearResults()">清空结果</button>
            </div>
            <div class="results" id="results">
                <p>点击"开始录音"开始实时语音识别...</p>
            </div>
        </div>

        <script>
            let ws = null;
            let mediaRecorder = null;
            let audioChunks = [];

            function startRecording() {
                if (ws) {
                    ws.close();
                }

                ws = new WebSocket('ws://localhost:8000/ws/asr');
                
                ws.onopen = function(event) {
                    console.log('WebSocket连接已建立');
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    
                    // 开始录音
                    navigator.mediaDevices.getUserMedia({ audio: true })
                        .then(stream => {
                            mediaRecorder = new MediaRecorder(stream, {
                                mimeType: 'audio/webm;codecs=opus'
                            });
                            
                            mediaRecorder.ondataavailable = function(event) {
                                if (event.data.size > 0) {
                                    const reader = new FileReader();
                                    reader.onload = function() {
                                        const arrayBuffer = reader.result;
                                        const base64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
                                        ws.send(JSON.stringify({
                                            type: 'audio',
                                            data: base64
                                        }));
                                    };
                                    reader.readAsArrayBuffer(event.data);
                                }
                            };
                            
                            mediaRecorder.start(1000); // 每秒发送一次数据
                        })
                        .catch(err => {
                            console.error('无法访问麦克风:', err);
                            alert('无法访问麦克风，请检查权限设置');
                        });
                };
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    if (data.type === 'result') {
                        displayResult(data.result);
                    }
                };
                
                ws.onclose = function(event) {
                    console.log('WebSocket连接已关闭');
                    document.getElementById('startBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                    if (mediaRecorder && mediaRecorder.state === 'recording') {
                        mediaRecorder.stop();
                    }
                };
            }

            function stopRecording() {
                if (ws) {
                    ws.close();
                }
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop();
                }
            }

            function displayResult(result) {
                const resultsDiv = document.getElementById('results');
                const resultItem = document.createElement('div');
                resultItem.className = 'result-item';
                
                let html = '';
                if (result.speaker_id) {
                    html += `<div class="speaker">说话人: ${result.speaker_id}</div>`;
                }
                html += `<div class="confidence">置信度: ${(result.confidence * 100).toFixed(1)}%</div>`;
                html += `<div class="text">${result.text}</div>`;
                
                resultItem.innerHTML = html;
                resultsDiv.appendChild(resultItem);
                resultsDiv.scrollTop = resultsDiv.scrollHeight;
            }

            function clearResults() {
                document.getElementById('results').innerHTML = '<p>结果已清空</p>';
            }
        </script>
    </body>
    </html>
    """

@app.post("/api/v1/asr/init")
async def init_asr(config: ASRConfig):
    """初始化ASR实例"""
    try:
        global asr_instance
        asr_instance = RealtimeASR(
            model_dir=config.model_dir,
            device=config.device,
            sample_rate=config.sample_rate,
            chunk_duration=config.chunk_duration,
            overlap_duration=config.overlap_duration,
            confidence_threshold=config.confidence_threshold,
            enable_speaker_id=config.enable_speaker_id
        )
        return {"status": "success", "message": "ASR实例初始化成功"}
    except Exception as e:
        logger.error(f"ASR初始化失败: {e}")
        raise HTTPException(status_code=500, detail=f"ASR初始化失败: {str(e)}")

@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    """WebSocket实时语音识别"""
    client_id = f"client_{id(websocket)}"
    await manager.connect(websocket, client_id)
    
    try:
        # 为每个客户端创建ASR实例
        asr = RealtimeASR(
            model_dir="iic/SenseVoiceSmall",
            device="cuda:0",
            confidence_threshold=0.6,
            enable_speaker_id=True
        )
        manager.set_asr_instance(client_id, asr)
        
        def on_result(result: RecognitionResult):
            """结果回调函数"""
            response = RecognitionResponse(
                text=result.text,
                confidence=result.confidence,
                speaker_id=result.speaker_id,
                timestamp=result.timestamp,
                language=result.language,
                emotion=result.emotion,
                event=result.event
            )
            
            # 异步发送结果
            asyncio.create_task(manager.send_message(
                json.dumps({
                    "type": "result",
                    "result": response.dict()
                }),
                client_id
            ))
        
        # 开始录音
        asr.start_recording(callback=on_result)
        
        # 等待消息
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message["type"] == "audio":
                    # 处理音频数据
                    audio_data = base64.b64decode(message["data"])
                    # 这里需要将音频数据转换为numpy数组并处理
                    # 简化处理，实际应用中需要更复杂的音频格式转换
                    pass
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket错误: {e}")
                break
                
    finally:
        # 清理资源
        asr = manager.get_asr_instance(client_id)
        if asr:
            asr.stop_recording()
        manager.disconnect(client_id)

@app.post("/api/v1/asr/recognize")
async def recognize_audio(audio_data: str, config: ASRConfig):
    """单次音频识别"""
    try:
        if not asr_instance:
            raise HTTPException(status_code=400, detail="ASR实例未初始化")
        
        # 解码base64音频数据
        audio_bytes = base64.b64decode(audio_data)
        
        # 转换为numpy数组（简化处理）
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
        
        # 进行识别
        result = asr_instance._process_audio_chunk(audio_array)
        
        if result:
            response = RecognitionResponse(
                text=result.text,
                confidence=result.confidence,
                speaker_id=result.speaker_id,
                timestamp=result.timestamp,
                language=result.language,
                emotion=result.emotion,
                event=result.event
            )
            return {"status": "success", "result": response.dict()}
        else:
            return {"status": "success", "result": None}
            
    except Exception as e:
        logger.error(f"音频识别失败: {e}")
        raise HTTPException(status_code=500, detail=f"音频识别失败: {str(e)}")

@app.get("/api/v1/asr/status")
async def get_status():
    """获取ASR状态"""
    return {
        "status": "running" if asr_instance else "stopped",
        "model_dir": asr_instance.model_dir if asr_instance else None,
        "device": asr_instance.device if asr_instance else None,
        "active_connections": len(manager.active_connections)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)