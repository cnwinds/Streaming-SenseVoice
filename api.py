# Set the device with environment, default is cuda:0
# export SENSEVOICE_DEVICE=cuda:1

import os, re
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from typing_extensions import Annotated
from typing import List, Optional
from enum import Enum
import torchaudio
import asyncio
import json
import base64
import numpy as np
from model import SenseVoiceSmall
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from io import BytesIO
from realtime_asr import RealtimeASR, RecognitionResult

TARGET_FS = 16000


class Language(str, Enum):
    auto = "auto"
    zh = "zh"
    en = "en"
    yue = "yue"
    ja = "ja"
    ko = "ko"
    nospeech = "nospeech"


model_dir = "iic/SenseVoiceSmall"
m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=os.getenv("SENSEVOICE_DEVICE", "cuda:0"))
m.eval()

regex = r"<\|.*\|>"

app = FastAPI(title="SenseVoice API", version="2.0.0")

# 全局ASR实例
realtime_asr: Optional[RealtimeASR] = None


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """


@app.post("/api/v1/asr")
async def turn_audio_to_text(
    files: Annotated[List[UploadFile], File(description="wav or mp3 audios in 16KHz")],
    keys: Annotated[str, Form(description="name of each audio joined with comma")] = None,
    lang: Annotated[Language, Form(description="language of audio content")] = "auto",
):
    audios = []
    for file in files:
        file_io = BytesIO(await file.read())
        data_or_path_or_list, audio_fs = torchaudio.load(file_io)

        # transform to target sample
        if audio_fs != TARGET_FS:
            resampler = torchaudio.transforms.Resample(orig_freq=audio_fs, new_freq=TARGET_FS)
            data_or_path_or_list = resampler(data_or_path_or_list)

        data_or_path_or_list = data_or_path_or_list.mean(0)
        audios.append(data_or_path_or_list)

    if lang == "":
        lang = "auto"

    if not keys:
        key = [f.filename for f in files]
    else:
        key = keys.split(",")

    res = m.inference(
        data_in=audios,
        language=lang,  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=False,
        ban_emo_unk=False,
        key=key,
        fs=TARGET_FS,
        **kwargs,
    )
    if len(res) == 0:
        return {"result": []}
    for it in res[0]:
        it["raw_text"] = it["text"]
        it["clean_text"] = re.sub(regex, "", it["text"], 0, re.MULTILINE)
        it["text"] = rich_transcription_postprocess(it["text"])
    return {"result": res[0]}


@app.post("/api/v1/asr/realtime/init")
async def init_realtime_asr(
    confidence_threshold: float = 0.6,
    enable_speaker_id: bool = True
):
    """初始化实时语音识别"""
    global realtime_asr
    try:
        realtime_asr = RealtimeASR(
            model_dir=model_dir,
            device=os.getenv("SENSEVOICE_DEVICE", "cuda:0"),
            confidence_threshold=confidence_threshold,
            enable_speaker_id=enable_speaker_id
        )
        return {"status": "success", "message": "实时ASR初始化成功"}
    except Exception as e:
        return {"status": "error", "message": f"实时ASR初始化失败: {str(e)}"}

@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """WebSocket实时语音识别"""
    await websocket.accept()
    
    if not realtime_asr:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "实时ASR未初始化，请先调用 /api/v1/asr/realtime/init"
        }))
        await websocket.close()
        return
    
    def on_result(result: RecognitionResult):
        """结果回调函数"""
        response = {
            "type": "result",
            "data": {
                "text": result.text,
                "confidence": result.confidence,
                "speaker_id": result.speaker_id,
                "timestamp": result.timestamp,
                "language": result.language,
                "emotion": result.emotion,
                "event": result.event
            }
        }
        asyncio.create_task(websocket.send_text(json.dumps(response)))
    
    try:
        # 开始录音
        realtime_asr.start_recording(callback=on_result)
        await websocket.send_text(json.dumps({
            "type": "status",
            "message": "开始实时语音识别"
        }))
        
        # 等待消息
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message["type"] == "stop":
                    break
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"处理错误: {str(e)}"
                }))
                
    finally:
        # 停止录音
        if realtime_asr:
            realtime_asr.stop_recording()
        await websocket.close()

@app.post("/api/v1/asr/realtime/start")
async def start_realtime_recognition():
    """开始实时识别"""
    global realtime_asr
    if not realtime_asr:
        return {"status": "error", "message": "实时ASR未初始化"}
    
    try:
        realtime_asr.start_recording()
        return {"status": "success", "message": "开始实时识别"}
    except Exception as e:
        return {"status": "error", "message": f"开始识别失败: {str(e)}"}

@app.post("/api/v1/asr/realtime/stop")
async def stop_realtime_recognition():
    """停止实时识别"""
    global realtime_asr
    if not realtime_asr:
        return {"status": "error", "message": "实时ASR未初始化"}
    
    try:
        realtime_asr.stop_recording()
        return {"status": "success", "message": "停止实时识别"}
    except Exception as e:
        return {"status": "error", "message": f"停止识别失败: {str(e)}"}

@app.get("/api/v1/asr/realtime/results")
async def get_realtime_results():
    """获取实时识别结果"""
    global realtime_asr
    if not realtime_asr:
        return {"status": "error", "message": "实时ASR未初始化"}
    
    try:
        results = realtime_asr.get_all_results()
        return {
            "status": "success",
            "results": [
                {
                    "text": r.text,
                    "confidence": r.confidence,
                    "speaker_id": r.speaker_id,
                    "timestamp": r.timestamp,
                    "language": r.language,
                    "emotion": r.emotion,
                    "event": r.event
                }
                for r in results
            ]
        }
    except Exception as e:
        return {"status": "error", "message": f"获取结果失败: {str(e)}"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=50000)
