#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
流式语音识别演示程序
真正的实时识别 - 边说边出结果
"""

import argparse
import numpy as np
import torch
import torchaudio
from pathlib import Path
import time

from model import SenseVoiceSmall
from speaker_embedding import SpeakerIdentifier
from streaming_asr import StreamingSpeechRecognizer, ChunkedStreamProcessor
from funasr.utils.postprocess_utils import rich_transcription_postprocess


def demo_streaming(
    audio_file: str,
    model_dir: str = "iic/SenseVoiceSmall",
    device: str = "cuda:0",
    language: str = "auto",
    chunk_duration_ms: int = 100,
    output_confidence: bool = True,
    output_speaker: bool = True,
    playback_speed: float = 1.0
):
    """
    从音频文件模拟流式识别
    
    Args:
        audio_file: 音频文件路径
        model_dir: 模型目录
        device: 设备
        language: 语言
        chunk_duration_ms: 音频块时长(毫秒)
        output_confidence: 是否输出置信度
        output_speaker: 是否输出说话人信息
        playback_speed: 播放速度（1.0=实时，0.5=0.5倍速）
    """
    
    print("=" * 80)
    print("流式语音识别演示 - 边说边出结果")
    print("=" * 80)
    
    # 加载模型
    print(f"\n正在加载模型: {model_dir}")
    model, model_kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=device)
    model.eval()
    print("✓ 模型加载完成")
    
    # 创建说话人识别器
    speaker_identifier = SpeakerIdentifier(
        similarity_threshold=0.75,
        embedding_dim=256
    ) if output_speaker else None
    
    # 准备模型参数
    kwargs = {
        **model_kwargs,
        "language": language,
        "use_itn": False,
        "output_confidence": output_confidence,
        "output_speaker_embedding": output_speaker,
        "speaker_identifier": speaker_identifier
    }
    
    # 加载音频
    print(f"\n正在加载音频: {audio_file}")
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # 转换为16kHz单声道
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    audio_data = waveform.squeeze().numpy()
    duration = len(audio_data) / 16000
    print(f"✓ 音频长度: {duration:.2f}秒")
    
    # 创建回调函数
    result_count = [0]
    start_time = [time.time()]
    
    def result_callback(result):
        result_count[0] += 1
        elapsed = time.time() - start_time[0]
        
        # 判断是否是最终结果
        is_final = result.get('is_final', False)
        chunk_id = result.get('chunk_id', 0)
        
        # 打印结果
        print(f"\n{'─' * 80}")
        print(f"[{elapsed:.2f}s] {'🔴 识别中' if not is_final else '✓ 最终结果'} (Chunk #{chunk_id})")
        print(f"{'─' * 80}")
        
        # 文本
        text = rich_transcription_postprocess(result.get('text', ''))
        print(f"📝 文本: {text}")
        
        # 说话人信息
        if output_speaker and 'speaker_info' in result:
            speaker_info = result['speaker_info']
            print(f"👤 说话人: {speaker_info['speaker_id']}", end="")
            if speaker_info.get('similarity') is not None:
                print(f" (相似度: {speaker_info['similarity']*100:.1f}%)")
            else:
                print(" (新说话人)")
        
        # 置信度
        if output_confidence and 'average_confidence' in result:
            conf = result['average_confidence']
            conf_bar = '█' * int(conf * 20) + '░' * (20 - int(conf * 20))
            print(f"📊 平均置信度: {conf*100:.1f}% [{conf_bar}]")
        
        # Token置信度（只显示前5个）
        if output_confidence and 'token_confidences' in result:
            tokens = result['token_confidences'][:5]
            if tokens:
                print("🔤 Token置信度:")
                for tc in tokens:
                    print(f"   {tc['token']}: {tc['confidence']*100:.1f}%")
                if len(result['token_confidences']) > 5:
                    print(f"   ... (还有{len(result['token_confidences'])-5}个)")
    
    # 创建流式识别器
    print("\n开始流式识别...")
    print("=" * 80)
    
    recognizer = StreamingSpeechRecognizer(
        model=model,
        model_kwargs=kwargs,
        sample_rate=16000,
        chunk_size=960,  # 60ms
        stride_size=640,  # 40ms
        min_chunk_threshold=8,  # 约0.5秒后开始识别
        callback=result_callback
    )
    recognizer.start()
    
    # 创建块处理器
    processor = ChunkedStreamProcessor(
        recognizer=recognizer,
        chunk_duration_ms=chunk_duration_ms,
        sample_rate=16000
    )
    
    # 模拟实时流式输入
    chunk_size = int(16000 * chunk_duration_ms / 1000)
    total_chunks = len(audio_data) // chunk_size
    
    print(f"模拟实时播放 (速度: {playback_speed}x)")
    print(f"总共 {total_chunks} 个音频块\n")
    
    for i in range(total_chunks):
        # 获取音频块
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        audio_chunk = audio_data[start_idx:end_idx]
        
        # 发送到处理器
        processor.process_audio(audio_chunk)
        
        # 模拟实时延迟
        time.sleep(chunk_duration_ms / 1000 / playback_speed)
    
    # 处理剩余音频
    if len(audio_data) % chunk_size > 0:
        remaining = audio_data[-(len(audio_data) % chunk_size):]
        processor.process_audio(remaining)
    
    # 等待处理完成并获取最终结果
    print("\n处理完成，获取最终结果...")
    time.sleep(0.5)
    final_result = processor.flush()
    
    if final_result and final_result.get('text'):
        print("\n" + "=" * 80)
        print("最终完整结果:")
        print("=" * 80)
        result_callback(final_result)
    
    # 停止识别器
    recognizer.stop()
    
    # 统计信息
    print("\n" + "=" * 80)
    print("识别统计:")
    print("=" * 80)
    print(f"总识别次数: {result_count[0]}")
    print(f"音频时长: {duration:.2f}秒")
    print(f"识别耗时: {time.time() - start_time[0]:.2f}秒")
    
    if output_speaker and speaker_identifier:
        speakers = speaker_identifier.get_all_speakers()
        print(f"检测到说话人: {len(speakers)}个")


def main():
    parser = argparse.ArgumentParser(
        description="流式语音识别演示 - 真正的实时识别，边说边出结果"
    )
    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="音频文件路径"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="iic/SenseVoiceSmall",
        help="模型目录"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="设备 (cuda:0 或 cpu)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="auto",
        choices=["auto", "zh", "en", "yue", "ja", "ko"],
        help="语言"
    )
    parser.add_argument(
        "--chunk_duration_ms",
        type=int,
        default=100,
        help="音频块时长(毫秒)，默认100ms"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="播放速度倍数，默认1.0(实时)"
    )
    parser.add_argument(
        "--no-confidence",
        action="store_true",
        help="不输出置信度"
    )
    parser.add_argument(
        "--no-speaker",
        action="store_true",
        help="不输出说话人信息"
    )
    
    args = parser.parse_args()
    
    if not Path(args.audio_file).exists():
        print(f"错误: 音频文件不存在: {args.audio_file}")
        return
    
    demo_streaming(
        audio_file=args.audio_file,
        model_dir=args.model_dir,
        device=args.device,
        language=args.language,
        chunk_duration_ms=args.chunk_duration_ms,
        output_confidence=not args.no_confidence,
        output_speaker=not args.no_speaker,
        playback_speed=args.speed
    )


if __name__ == "__main__":
    main()
