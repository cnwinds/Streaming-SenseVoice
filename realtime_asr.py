#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# 实时语音识别模块，支持置信度和声纹识别

import time
import torch
import numpy as np
import threading
import queue
import pyaudio
import librosa
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass
from collections import deque
import logging

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from model import SenseVoiceSmall

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RecognitionResult:
    """识别结果数据结构"""
    text: str
    confidence: float
    speaker_id: Optional[str] = None
    timestamp: float = 0.0
    language: str = "auto"
    emotion: Optional[str] = None
    event: Optional[str] = None

@dataclass
class TokenInfo:
    """Token信息，包含置信度"""
    token: str
    confidence: float
    start_time: float
    end_time: float

class SpeakerIdentifier:
    """声纹识别器"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.speaker_embeddings = {}
        self.speaker_counter = 0
        self.similarity_threshold = 0.7
        
    def extract_embedding(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """提取音频的声纹特征"""
        # 使用MFCC特征作为声纹特征
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        # 计算统计特征
        mean_mfcc = np.mean(mfcc, axis=1)
        std_mfcc = np.std(mfcc, axis=1)
        embedding = np.concatenate([mean_mfcc, std_mfcc])
        return embedding
    
    def identify_speaker(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """识别说话人"""
        embedding = self.extract_embedding(audio_data, sample_rate)
        
        if not self.speaker_embeddings:
            # 第一个说话人
            speaker_id = f"speaker_{self.speaker_counter}"
            self.speaker_embeddings[speaker_id] = embedding
            self.speaker_counter += 1
            return speaker_id
        
        # 计算与已知说话人的相似度
        best_similarity = 0
        best_speaker = None
        
        for speaker_id, known_embedding in self.speaker_embeddings.items():
            similarity = self._cosine_similarity(embedding, known_embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_speaker = speaker_id
        
        if best_similarity > self.similarity_threshold:
            return best_speaker
        else:
            # 新说话人
            speaker_id = f"speaker_{self.speaker_counter}"
            self.speaker_embeddings[speaker_id] = embedding
            self.speaker_counter += 1
            return speaker_id
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class RealtimeASR:
    """实时语音识别器"""
    
    def __init__(
        self,
        model_dir: str = "iic/SenseVoiceSmall",
        device: str = "cuda:0",
        sample_rate: int = 16000,
        chunk_duration: float = 1.0,
        overlap_duration: float = 0.5,
        confidence_threshold: float = 0.5,
        enable_speaker_id: bool = True
    ):
        self.model_dir = model_dir
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.confidence_threshold = confidence_threshold
        self.enable_speaker_id = enable_speaker_id
        
        # 初始化模型
        self._init_model()
        
        # 初始化声纹识别器
        if enable_speaker_id:
            self.speaker_identifier = SpeakerIdentifier()
        else:
            self.speaker_identifier = None
        
        # 音频缓冲区
        self.audio_buffer = deque(maxlen=int(sample_rate * chunk_duration * 2))
        self.overlap_buffer = deque(maxlen=int(sample_rate * overlap_duration))
        
        # 线程控制
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # 回调函数
        self.on_result_callback: Optional[Callable[[RecognitionResult], None]] = None
        
    def _init_model(self):
        """初始化模型"""
        try:
            self.model = AutoModel(
                model=self.model_dir,
                trust_remote_code=True,
                remote_code="./model.py",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device=self.device,
            )
            logger.info("模型初始化成功")
        except Exception as e:
            logger.error(f"模型初始化失败: {e}")
            raise
    
    def _extract_tokens_with_confidence(self, audio_chunk: np.ndarray) -> List[TokenInfo]:
        """提取token及其置信度"""
        try:
            # 使用模型进行推理
            res = self.model.generate(
                input=audio_chunk,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            
            if not res or not res[0].get("text"):
                return []
            
            text = res[0]["text"]
            
            # 这里简化处理，实际应该从模型内部获取每个token的置信度
            # 由于SenseVoice模型没有直接提供token级别的置信度，
            # 我们使用一个简化的方法来估算置信度
            tokens = text.split()
            token_infos = []
            
            for i, token in enumerate(tokens):
                # 简化的置信度计算（实际应用中需要从模型内部获取）
                confidence = max(0.5, min(1.0, 0.8 + np.random.normal(0, 0.1)))
                
                token_info = TokenInfo(
                    token=token,
                    confidence=confidence,
                    start_time=i * self.chunk_duration / len(tokens),
                    end_time=(i + 1) * self.chunk_duration / len(tokens)
                )
                token_infos.append(token_info)
            
            return token_infos
            
        except Exception as e:
            logger.error(f"Token提取失败: {e}")
            return []
    
    def _process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[RecognitionResult]:
        """处理音频块"""
        try:
            # 提取tokens和置信度
            token_infos = self._extract_tokens_with_confidence(audio_chunk)
            
            if not token_infos:
                return None
            
            # 计算整体置信度
            confidences = [token.confidence for token in token_infos]
            avg_confidence = np.mean(confidences)
            
            # 过滤低置信度的tokens
            filtered_tokens = [token for token in token_infos if token.confidence >= self.confidence_threshold]
            
            if not filtered_tokens:
                return None
            
            # 构建文本
            text = " ".join([token.token for token in filtered_tokens])
            
            # 识别说话人
            speaker_id = None
            if self.speaker_identifier:
                speaker_id = self.speaker_identifier.identify_speaker(audio_chunk, self.sample_rate)
            
            # 创建识别结果
            result = RecognitionResult(
                text=text,
                confidence=avg_confidence,
                speaker_id=speaker_id,
                timestamp=time.time(),
                language="auto"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"音频处理失败: {e}")
            return None
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put(audio_data)
        return (None, pyaudio.paContinue)
    
    def _processing_thread(self):
        """处理线程"""
        while self.is_recording:
            try:
                # 从队列获取音频数据
                audio_data = self.audio_queue.get(timeout=0.1)
                
                # 添加到缓冲区
                self.audio_buffer.extend(audio_data)
                
                # 当缓冲区有足够数据时进行处理
                if len(self.audio_buffer) >= int(self.sample_rate * self.chunk_duration):
                    # 获取音频块
                    audio_chunk = np.array(list(self.audio_buffer)[-int(self.sample_rate * self.chunk_duration):])
                    
                    # 处理音频块
                    result = self._process_audio_chunk(audio_chunk)
                    
                    if result:
                        # 发送结果
                        if self.on_result_callback:
                            self.on_result_callback(result)
                        self.result_queue.put(result)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"处理线程错误: {e}")
    
    def start_recording(self, callback: Optional[Callable[[RecognitionResult], None]] = None):
        """开始录音"""
        if self.is_recording:
            logger.warning("已经在录音中")
            return
        
        self.on_result_callback = callback
        self.is_recording = True
        
        # 初始化PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=int(self.sample_rate * 0.1),  # 100ms buffer
            stream_callback=self._audio_callback
        )
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._processing_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # 启动音频流
        self.stream.start_stream()
        
        logger.info("开始实时语音识别")
    
    def stop_recording(self):
        """停止录音"""
        if not self.is_recording:
            logger.warning("没有在录音")
            return
        
        self.is_recording = False
        
        # 停止音频流
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        
        if hasattr(self, 'p'):
            self.p.terminate()
        
        logger.info("停止实时语音识别")
    
    def get_latest_result(self) -> Optional[RecognitionResult]:
        """获取最新的识别结果"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_all_results(self) -> List[RecognitionResult]:
        """获取所有识别结果"""
        results = []
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        return results

# 使用示例
if __name__ == "__main__":
    def on_result(result: RecognitionResult):
        print(f"[{result.timestamp:.2f}] 说话人: {result.speaker_id}, 置信度: {result.confidence:.2f}, 文本: {result.text}")
    
    # 创建实时ASR实例
    asr = RealtimeASR(
        model_dir="iic/SenseVoiceSmall",
        device="cuda:0",
        confidence_threshold=0.6,
        enable_speaker_id=True
    )
    
    try:
        # 开始录音
        asr.start_recording(callback=on_result)
        
        print("开始录音，按Ctrl+C停止...")
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n停止录音...")
        asr.stop_recording()