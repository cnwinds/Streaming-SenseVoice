# -*- encoding: utf-8 -*-
"""
实时语音流处理模块
支持流式音频输入和实时识别
"""

import numpy as np
import torch
import queue
import threading
from typing import Optional, Callable, Dict, List
from collections import deque


class AudioStreamBuffer:
    """音频流缓冲区，用于处理实时音频流"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1600,  # 100ms at 16kHz
        max_buffer_size: int = 48000  # 3秒最大缓冲
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.max_buffer_size = max_buffer_size
        self.buffer = deque(maxlen=max_buffer_size)
        self.lock = threading.Lock()
        
    def add_audio(self, audio_chunk: np.ndarray):
        """添加音频块到缓冲区"""
        with self.lock:
            self.buffer.extend(audio_chunk.flatten())
            
    def get_audio(self, num_samples: Optional[int] = None) -> np.ndarray:
        """从缓冲区获取音频数据"""
        with self.lock:
            if num_samples is None:
                num_samples = len(self.buffer)
            
            if len(self.buffer) < num_samples:
                return np.array([])
            
            audio_data = np.array([self.buffer.popleft() for _ in range(num_samples)])
            return audio_data
    
    def peek_audio(self, num_samples: int) -> np.ndarray:
        """查看缓冲区数据但不移除"""
        with self.lock:
            if len(self.buffer) < num_samples:
                return np.array([])
            return np.array(list(self.buffer)[:num_samples])
    
    def clear(self):
        """清空缓冲区"""
        with self.lock:
            self.buffer.clear()
            
    def size(self) -> int:
        """返回缓冲区大小"""
        with self.lock:
            return len(self.buffer)


class RealtimeASRProcessor:
    """实时语音识别处理器"""
    
    def __init__(
        self,
        model,
        model_kwargs: dict,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
        vad_threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        callback: Optional[Callable] = None
    ):
        """
        Args:
            model: SenseVoice模型实例
            model_kwargs: 模型推理参数
            sample_rate: 采样率
            chunk_duration_ms: 音频块时长(毫秒)
            vad_threshold: VAD阈值
            min_speech_duration_ms: 最小语音时长(毫秒)
            callback: 识别结果回调函数
        """
        self.model = model
        self.model_kwargs = model_kwargs
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.vad_threshold = vad_threshold
        self.min_speech_samples = int(sample_rate * min_speech_duration_ms / 1000)
        self.callback = callback
        
        self.buffer = AudioStreamBuffer(
            sample_rate=sample_rate,
            chunk_size=self.chunk_size
        )
        self.is_running = False
        self.processing_thread = None
        
        # 语音活动检测状态
        self.is_speaking = False
        self.speech_buffer = []
        
    def start(self):
        """启动实时处理"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop(self):
        """停止实时处理"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            
    def add_audio(self, audio_chunk: np.ndarray):
        """添加音频数据"""
        self.buffer.add_audio(audio_chunk)
        
    def _simple_vad(self, audio: np.ndarray) -> bool:
        """简单的VAD检测"""
        if len(audio) == 0:
            return False
        energy = np.sqrt(np.mean(audio ** 2))
        return energy > self.vad_threshold
        
    def _process_loop(self):
        """处理循环"""
        while self.is_running:
            # 检查缓冲区是否有足够数据
            if self.buffer.size() < self.chunk_size:
                threading.Event().wait(0.01)  # 等待10ms
                continue
                
            # 获取音频块
            audio_chunk = self.buffer.peek_audio(self.chunk_size)
            if len(audio_chunk) == 0:
                continue
                
            # VAD检测
            has_speech = self._simple_vad(audio_chunk)
            
            if has_speech:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_buffer = []
                    
                # 从缓冲区移除已处理的数据
                self.buffer.get_audio(self.chunk_size)
                self.speech_buffer.append(audio_chunk)
            else:
                if self.is_speaking:
                    # 语音结束，处理累积的语音
                    if len(self.speech_buffer) > 0:
                        speech_audio = np.concatenate(self.speech_buffer)
                        if len(speech_audio) >= self.min_speech_samples:
                            self._process_speech(speech_audio)
                    
                    self.is_speaking = False
                    self.speech_buffer = []
                else:
                    # 没有语音，移除静音数据
                    self.buffer.get_audio(self.chunk_size)
                    
    def _process_speech(self, audio: np.ndarray):
        """处理语音片段"""
        try:
            # 准备输入
            audio_tensor = torch.from_numpy(audio).float()
            
            # 调用模型推理，包含置信度和声纹
            result = self.model.inference(
                data_in=[audio_tensor],
                **self.model_kwargs
            )
            
            if self.callback and result:
                self.callback(result)
                
        except Exception as e:
            print(f"处理语音时出错: {e}")


class RealtimeStreamManager:
    """实时流管理器"""
    
    def __init__(
        self,
        model,
        model_kwargs: dict,
        result_queue: Optional[queue.Queue] = None
    ):
        """
        Args:
            model: 模型实例
            model_kwargs: 模型参数
            result_queue: 结果队列
        """
        self.model = model
        self.model_kwargs = model_kwargs
        self.result_queue = result_queue or queue.Queue()
        self.processor = None
        
    def start_stream(self, **kwargs):
        """开始流式处理"""
        def callback(result):
            self.result_queue.put(result)
            
        self.processor = RealtimeASRProcessor(
            model=self.model,
            model_kwargs=self.model_kwargs,
            callback=callback,
            **kwargs
        )
        self.processor.start()
        
    def stop_stream(self):
        """停止流式处理"""
        if self.processor:
            self.processor.stop()
            
    def add_audio(self, audio_chunk: np.ndarray):
        """添加音频数据"""
        if self.processor:
            self.processor.add_audio(audio_chunk)
            
    def get_result(self, timeout: Optional[float] = None) -> Optional[Dict]:
        """获取识别结果"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None
