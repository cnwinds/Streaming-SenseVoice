#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
实时语音识别演示程序
支持从麦克风输入或音频文件进行实时识别
"""

import argparse
import numpy as np
import torch
from pathlib import Path
import time

from model import SenseVoiceSmall
from speaker_embedding import SpeakerIdentifier
from funasr.utils.postprocess_utils import rich_transcription_postprocess


def demo_realtime_from_file(
    audio_file: str,
    model_dir: str = "iic/SenseVoiceSmall",
    device: str = "cuda:0",
    language: str = "auto",
    chunk_size: int = 1600,  # 100ms at 16kHz
    output_confidence: bool = True,
    output_speaker: bool = True
):
    """从音频文件模拟实时识别"""
    
    print("=" * 80)
    print("实时语音识别演示 - 文件模式")
    print("=" * 80)
    
    # 加载模型
    print(f"\n正在加载模型: {model_dir}")
    m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=device)
    m.eval()
    print("模型加载完成")
    
    # 创建说话人识别器
    speaker_identifier = SpeakerIdentifier(
        similarity_threshold=0.75,
        embedding_dim=256
    )
    
    # 加载音频
    print(f"\n正在加载音频: {audio_file}")
    import torchaudio
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # 转换为16kHz单声道
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    audio_data = waveform.squeeze().numpy()
    print(f"音频长度: {len(audio_data) / 16000:.2f}秒")
    
    # 模拟实时处理
    print("\n开始实时识别...")
    print("-" * 80)
    
    total_chunks = len(audio_data) // chunk_size
    speech_buffer = []
    is_speaking = False
    vad_threshold = 0.01
    min_speech_chunks = 15  # 最少15个chunk（约1.5秒）才处理
    
    for i in range(total_chunks):
        # 获取音频块
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        audio_chunk = audio_data[start_idx:end_idx]
        
        # 简单VAD
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        has_speech = energy > vad_threshold
        
        if has_speech:
            if not is_speaking:
                is_speaking = True
                speech_buffer = []
            speech_buffer.append(audio_chunk)
        else:
            if is_speaking and len(speech_buffer) >= min_speech_chunks:
                # 处理语音段
                speech_audio = np.concatenate(speech_buffer)
                speech_tensor = torch.from_numpy(speech_audio).float()
                
                # 进行识别
                try:
                    result = m.inference(
                        data_in=[speech_tensor],
                        language=language,
                        use_itn=True,
                        output_confidence=output_confidence,
                        output_speaker_embedding=output_speaker,
                        speaker_identifier=speaker_identifier if output_speaker else None,
                        **kwargs
                    )
                    
                    if result and result[0]:
                        res_item = result[0][0]
                        
                        # 打印结果
                        timestamp = i * chunk_size / 16000
                        print(f"\n[{timestamp:.2f}s]")
                        print(f"文本: {rich_transcription_postprocess(res_item['text'])}")
                        
                        if output_speaker and 'speaker_info' in res_item:
                            speaker_info = res_item['speaker_info']
                            print(f"说话人: {speaker_info['speaker_id']}", end="")
                            if speaker_info.get('similarity') is not None:
                                print(f" (相似度: {speaker_info['similarity']*100:.1f}%)")
                            else:
                                print(" (新说话人)")
                        
                        if output_confidence and 'average_confidence' in res_item:
                            print(f"平均置信度: {res_item['average_confidence']*100:.1f}%")
                        
                        if output_confidence and 'token_confidences' in res_item:
                            print("Token置信度:")
                            for tc in res_item['token_confidences'][:10]:  # 只显示前10个
                                print(f"  {tc['token']}: {tc['confidence']*100:.1f}%")
                            if len(res_item['token_confidences']) > 10:
                                print(f"  ... (还有{len(res_item['token_confidences'])-10}个)")
                        
                        print("-" * 80)
                        
                except Exception as e:
                    print(f"识别错误: {e}")
                
                is_speaking = False
                speech_buffer = []
        
        # 模拟实时延迟
        time.sleep(chunk_size / 16000 * 0.5)  # 0.5倍速播放
    
    print("\n识别完成！")
    print(f"检测到 {len(speaker_identifier.get_all_speakers())} 个不同的说话人")


def demo_realtime_simple(
    audio_file: str,
    model_dir: str = "iic/SenseVoiceSmall",
    device: str = "cuda:0",
    language: str = "auto"
):
    """简单的识别演示（非流式）"""
    
    print("=" * 80)
    print("语音识别演示 - 简单模式")
    print("=" * 80)
    
    # 加载模型
    print(f"\n正在加载模型: {model_dir}")
    m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=device)
    m.eval()
    
    # 创建说话人识别器
    speaker_identifier = SpeakerIdentifier()
    
    # 识别
    print(f"\n正在识别: {audio_file}")
    start_time = time.time()
    
    res = m.inference(
        data_in=audio_file,
        language=language,
        use_itn=True,
        output_confidence=True,
        output_speaker_embedding=True,
        speaker_identifier=speaker_identifier,
        **kwargs,
    )
    
    end_time = time.time()
    
    # 显示结果
    if res and res[0]:
        result = res[0][0]
        
        print("\n" + "=" * 80)
        print("识别结果:")
        print("=" * 80)
        print(f"\n文本: {rich_transcription_postprocess(result['text'])}")
        
        if 'speaker_info' in result:
            speaker_info = result['speaker_info']
            print(f"\n说话人ID: {speaker_info['speaker_id']}")
            if speaker_info.get('similarity') is not None:
                print(f"相似度: {speaker_info['similarity']*100:.1f}%")
        
        if 'average_confidence' in result:
            print(f"\n平均置信度: {result['average_confidence']*100:.1f}%")
        
        if 'token_confidences' in result:
            print(f"\nToken数量: {len(result['token_confidences'])}")
            print("\n前10个Token的置信度:")
            for tc in result['token_confidences'][:10]:
                print(f"  {tc['token']}: {tc['confidence']*100:.1f}%")
        
        print(f"\n识别耗时: {end_time - start_time:.3f}秒")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="实时语音识别演示")
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
        "--mode",
        type=str,
        default="simple",
        choices=["simple", "realtime"],
        help="模式: simple=简单模式, realtime=实时流式模式"
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
    
    if args.mode == "realtime":
        demo_realtime_from_file(
            audio_file=args.audio_file,
            model_dir=args.model_dir,
            device=args.device,
            language=args.language,
            output_confidence=not args.no_confidence,
            output_speaker=not args.no_speaker
        )
    else:
        demo_realtime_simple(
            audio_file=args.audio_file,
            model_dir=args.model_dir,
            device=args.device,
            language=args.language
        )


if __name__ == "__main__":
    main()
