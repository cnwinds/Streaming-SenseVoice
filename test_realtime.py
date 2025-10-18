#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# 实时语音识别测试脚本

import time
import numpy as np
import pyaudio
import threading
from realtime_asr import RealtimeASR, RecognitionResult

def test_realtime_asr():
    """测试实时语音识别"""
    
    def on_result(result: RecognitionResult):
        """结果回调函数"""
        print(f"\n[识别结果]")
        print(f"时间: {time.strftime('%H:%M:%S', time.localtime(result.timestamp))}")
        print(f"说话人: {result.speaker_id or '未知'}")
        print(f"置信度: {result.confidence:.3f}")
        print(f"文本: {result.text}")
        print(f"语言: {result.language}")
        if result.emotion:
            print(f"情感: {result.emotion}")
        if result.event:
            print(f"事件: {result.event}")
        print("-" * 50)
    
    # 创建ASR实例
    print("初始化实时语音识别...")
    asr = RealtimeASR(
        model_dir="iic/SenseVoiceSmall",
        device="cuda:0",
        confidence_threshold=0.6,
        enable_speaker_id=True
    )
    
    try:
        print("开始录音，按Ctrl+C停止...")
        asr.start_recording(callback=on_result)
        
        # 保持运行
        while True:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n停止录音...")
        asr.stop_recording()
        print("录音已停止")

def test_audio_processing():
    """测试音频处理功能"""
    print("测试音频处理功能...")
    
    # 生成测试音频数据
    sample_rate = 16000
    duration = 2.0
    frequency = 440  # A4音符
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # 创建ASR实例
    asr = RealtimeASR(
        model_dir="iic/SenseVoiceSmall",
        device="cuda:0",
        confidence_threshold=0.5,
        enable_speaker_id=True
    )
    
    # 处理音频
    result = asr._process_audio_chunk(audio_data)
    
    if result:
        print("音频处理结果:")
        print(f"文本: {result.text}")
        print(f"置信度: {result.confidence}")
        print(f"说话人: {result.speaker_id}")
    else:
        print("未识别到有效语音")

def test_speaker_identification():
    """测试声纹识别功能"""
    print("测试声纹识别功能...")
    
    from realtime_asr import SpeakerIdentifier
    
    # 创建声纹识别器
    speaker_id = SpeakerIdentifier()
    
    # 生成不同频率的测试音频
    sample_rate = 16000
    duration = 1.0
    
    # 说话人1 - 低频
    t1 = np.linspace(0, duration, int(sample_rate * duration), False)
    audio1 = np.sin(2 * np.pi * 220 * t1).astype(np.float32)
    
    # 说话人2 - 高频
    t2 = np.linspace(0, duration, int(sample_rate * duration), False)
    audio2 = np.sin(2 * np.pi * 880 * t2).astype(np.float32)
    
    # 识别说话人
    speaker1 = speaker_id.identify_speaker(audio1, sample_rate)
    speaker2 = speaker_id.identify_speaker(audio2, sample_rate)
    speaker1_again = speaker_id.identify_speaker(audio1, sample_rate)
    
    print(f"音频1识别为: {speaker1}")
    print(f"音频2识别为: {speaker2}")
    print(f"音频1再次识别为: {speaker1_again}")
    
    if speaker1 == speaker1_again and speaker1 != speaker2:
        print("✅ 声纹识别功能正常")
    else:
        print("❌ 声纹识别功能异常")

if __name__ == "__main__":
    print("=" * 60)
    print("实时语音识别测试")
    print("=" * 60)
    
    # 测试声纹识别
    test_speaker_identification()
    print()
    
    # 测试音频处理
    test_audio_processing()
    print()
    
    # 测试实时识别（需要麦克风）
    print("是否进行实时语音识别测试？(需要麦克风) [y/N]: ", end="")
    choice = input().strip().lower()
    if choice == 'y':
        test_realtime_asr()
    else:
        print("跳过实时语音识别测试")
    
    print("测试完成！")