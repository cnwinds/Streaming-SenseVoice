# -*- encoding: utf-8 -*-
"""
流式语音识别模块 - 真正的实时识别，边说边出结果
参考 streaming-sensevoice 实现
"""

import numpy as np
import torch
import threading
import queue
from typing import Optional, Callable, Dict, List
from collections import deque
import time


class StreamingSpeechRecognizer:
    """流式语音识别器 - 支持真正的实时识别"""
    
    def __init__(
        self,
        model,
        model_kwargs: dict,
        sample_rate: int = 16000,
        chunk_size: int = 960,  # 60ms chunks for lower latency
        stride_size: int = 640,  # 40ms stride
        context_size: int = 5,  # Keep 5 chunks of context
        min_chunk_threshold: int = 10,  # Minimum chunks before recognition
        callback: Optional[Callable] = None
    ):
        """
        Args:
            model: SenseVoice模型实例
            model_kwargs: 模型推理参数
            sample_rate: 采样率 (Hz)
            chunk_size: 每个音频块大小 (样本数)
            stride_size: 步长 (样本数)，用于重叠处理
            context_size: 保留的上下文块数量
            min_chunk_threshold: 开始识别前的最小块数
            callback: 识别结果回调函数
        """
        self.model = model
        self.model_kwargs = model_kwargs
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stride_size = stride_size
        self.context_size = context_size
        self.min_chunk_threshold = min_chunk_threshold
        self.callback = callback
        
        # 音频缓冲区
        self.audio_buffer = deque(maxlen=chunk_size * context_size)
        self.chunk_count = 0
        self.last_text = ""
        
        # 线程控制
        self.is_running = False
        self.process_thread = None
        self.audio_queue = queue.Queue()
        
        # 锁
        self.lock = threading.Lock()
        
    def start(self):
        """启动流式识别"""
        if self.is_running:
            return
            
        self.is_running = True
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()
        
    def stop(self):
        """停止流式识别"""
        self.is_running = False
        if self.process_thread:
            self.process_thread.join(timeout=2.0)
            
    def add_audio(self, audio_chunk: np.ndarray):
        """添加音频数据"""
        self.audio_queue.put(audio_chunk)
        
    def reset(self):
        """重置识别状态"""
        with self.lock:
            self.audio_buffer.clear()
            self.chunk_count = 0
            self.last_text = ""
            # 清空队列
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
                    
    def _process_loop(self):
        """处理循环 - 持续处理音频"""
        while self.is_running:
            try:
                # 获取音频块（非阻塞，超时100ms）
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                with self.lock:
                    # 添加到缓冲区
                    self.audio_buffer.extend(audio_chunk.flatten())
                    self.chunk_count += 1
                    
                    # 当有足够的音频数据时进行识别
                    if self.chunk_count >= self.min_chunk_threshold and len(self.audio_buffer) >= self.chunk_size:
                        self._recognize_stream()
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"处理音频时出错: {e}")
                
    def _recognize_stream(self):
        """对当前缓冲区的音频进行流式识别"""
        try:
            # 获取当前缓冲区的所有音频
            audio_data = np.array(list(self.audio_buffer))
            
            if len(audio_data) < self.chunk_size:
                return
                
            # 转换为tensor
            audio_tensor = torch.from_numpy(audio_data).float()
            
            # 进行识别
            result = self.model.inference(
                data_in=[audio_tensor],
                **self.model_kwargs
            )
            
            if result and result[0] and len(result[0]) > 0:
                res_item = result[0][0]
                current_text = res_item.get('text', '')
                
                # 检查文本是否有变化
                if current_text != self.last_text:
                    # 添加流式标记
                    res_item['is_final'] = False
                    res_item['chunk_id'] = self.chunk_count
                    
                    # 回调
                    if self.callback:
                        self.callback(res_item)
                    
                    self.last_text = current_text
                    
        except Exception as e:
            print(f"识别出错: {e}")
            
    def finalize(self) -> Optional[Dict]:
        """完成识别，返回最终结果"""
        try:
            with self.lock:
                if len(self.audio_buffer) == 0:
                    return None
                    
                # 获取所有音频数据
                audio_data = np.array(list(self.audio_buffer))
                audio_tensor = torch.from_numpy(audio_data).float()
                
                # 最终识别
                result = self.model.inference(
                    data_in=[audio_tensor],
                    **self.model_kwargs
                )
                
                if result and result[0] and len(result[0]) > 0:
                    res_item = result[0][0]
                    res_item['is_final'] = True
                    res_item['chunk_id'] = self.chunk_count
                    return res_item
                    
        except Exception as e:
            print(f"最终识别出错: {e}")
            
        return None


class StreamingASRManager:
    """流式ASR管理器 - 管理多个并发的流式识别会话"""
    
    def __init__(self, model, model_kwargs: dict):
        """
        Args:
            model: SenseVoice模型实例
            model_kwargs: 模型参数字典
        """
        self.model = model
        self.model_kwargs = model_kwargs
        self.sessions: Dict[str, StreamingSpeechRecognizer] = {}
        self.lock = threading.Lock()
        
    def create_session(
        self,
        session_id: str,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> StreamingSpeechRecognizer:
        """创建新的识别会话"""
        with self.lock:
            if session_id in self.sessions:
                # 停止旧会话
                self.sessions[session_id].stop()
                
            # 创建新会话
            recognizer = StreamingSpeechRecognizer(
                model=self.model,
                model_kwargs=self.model_kwargs.copy(),
                callback=callback,
                **kwargs
            )
            recognizer.start()
            self.sessions[session_id] = recognizer
            return recognizer
            
    def get_session(self, session_id: str) -> Optional[StreamingSpeechRecognizer]:
        """获取会话"""
        with self.lock:
            return self.sessions.get(session_id)
            
    def close_session(self, session_id: str):
        """关闭会话"""
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id].stop()
                del self.sessions[session_id]
                
    def close_all(self):
        """关闭所有会话"""
        with self.lock:
            for session in self.sessions.values():
                session.stop()
            self.sessions.clear()


class ChunkedStreamProcessor:
    """分块流处理器 - 用于处理连续的音频流"""
    
    def __init__(
        self,
        recognizer: StreamingSpeechRecognizer,
        chunk_duration_ms: int = 100,
        sample_rate: int = 16000
    ):
        """
        Args:
            recognizer: 流式识别器
            chunk_duration_ms: 块时长(毫秒)
            sample_rate: 采样率
        """
        self.recognizer = recognizer
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.sample_rate = sample_rate
        self.buffer = []
        
    def process_audio(self, audio_data: np.ndarray):
        """处理音频数据"""
        # 添加到缓冲区
        self.buffer.extend(audio_data.flatten())
        
        # 当有足够数据时，分块发送
        while len(self.buffer) >= self.chunk_size:
            chunk = np.array(self.buffer[:self.chunk_size])
            self.buffer = self.buffer[self.chunk_size:]
            self.recognizer.add_audio(chunk)
            
    def flush(self) -> Optional[Dict]:
        """刷新缓冲区并获取最终结果"""
        # 发送剩余音频
        if len(self.buffer) > 0:
            chunk = np.array(self.buffer)
            self.recognizer.add_audio(chunk)
            self.buffer = []
            
        # 等待处理完成
        time.sleep(0.2)
        
        # 获取最终结果
        return self.recognizer.finalize()
        
    def reset(self):
        """重置处理器"""
        self.buffer = []
        self.recognizer.reset()
